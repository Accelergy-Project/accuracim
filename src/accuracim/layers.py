"""Conv2d / Linear / Conv1D replacements that model a CIM array.

One forward pass does all of the analog computation. It clips, quantizes and slices the
inputs and weights, and adds DAC and cell noise. Then, for each (input slice, weight slice,
row group), it adds ADC noise, quantization and clipping error.
`NoisyLinear` and `NoisyConv1D` reshape to `NoisyConv2d`, so this logic exists only once.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For Hugging Face's Conv1D
from transformers.pytorch_utils import Conv1D

from accuracim.params import get, is_percentile_below_100
from accuracim.quantize import (
    add_noise_to_sliced_tensors,
    apply_adc_noise,
    dequantize_tensor,
    detach_cpu,
    get_quantization_scale,
    quantize_tensor,
    slice_and_add_noise_to_tensor,
    slice_weights,
)
from accuracim.sampling import nested_defaultdict, sample_fraction_tensor


class NoisyConv2d(nn.Conv2d):
    """Conv2d with noisy input, weights, and ADC output."""

    def __init__(
        self,
        weight_tensor,
        bias_tensor,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        input_percentiles: dict = None,
        output_percentiles: dict = None,
        probe_outputs: bool = False,
        name: str = None,
        **params,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        self.name = name
        self.params = params
        self.input_percentiles = input_percentiles
        self.output_percentiles = output_percentiles
        self.device = device if device is not None else weight_tensor.device

        # Record pre-ADC outputs (for ADC statistics)
        self.probe_outputs = probe_outputs
        if probe_outputs:
            self.outputs = nested_defaultdict()

        # Own copy: the original may share storage with a tied weight (GPT-2 `lm_head` is
        # `wte`), which in-place clipping, drift and noise below must not change
        self.weight.data = weight_tensor.clone()
        if bias:
            self.bias.data = bias_tensor

        input_bits = get(params, "input_bits")
        input_clip_percentile = get(params, "input_clip_percentile")
        differential_input = get(params, "differential_input")
        self.input_scale, self.input_zp, self.input_clip_range = get_quantization_scale(
            input_bits,
            input_percentiles,
            input_clip_percentile,
            symmetrical_range=differential_input,
        )

        # Weight bit-slices (CPU)
        (
            self.sliced_weights,
            self.weight_slice_lsb_to_msb,
            self.weight_slice_scales,
            self.weight_scale,
            self.weight_clip_range,
        ) = slice_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv with noisy input and weights, then per-slice ADC effects on the result."""
        params = self.params

        # Input
        input_bits = get(params, "input_bits")
        input_clip_percentile = get(params, "input_clip_percentile")
        enable_input_quantization = get(params, "enable_input_quantization")
        enable_input_clipping = get(params, "enable_input_clipping")
        input_physical_noise = get(params, "input_physical_noise")
        input_slice_lsb_to_msb = get(params, "input_slice_lsb_to_msb")
        differential_input = get(params, "differential_input")

        # Weight
        weight_bits = get(params, "weight_bits")
        weight_physical_noise = get(params, "weight_physical_noise")
        differential_weight = get(params, "differential_weight")

        # ADC
        adc_physical_noise = get(params, "adc_physical_noise")
        enable_adc_clipping = get(params, "enable_adc_clipping")
        enable_adc_quantization = get(params, "enable_adc_quantization")
        num_rows = get(params, "num_rows")

        N_batch, Cin, H_pad, W_pad = x.shape
        Cout, Cin, R, S = self.weight.shape

        with torch.no_grad():
            # --- Input: clip, quantize, slice + DAC noise ---
            low, high = self.input_clip_range
            if enable_input_clipping and not (low is None and high is None):
                if is_percentile_below_100(input_clip_percentile):
                    x = x.clamp(low, high)

            if input_bits is not None:
                x_q = quantize_tensor(x, self.input_scale, self.input_zp).to(dtype=torch.int)

                if enable_input_quantization:
                    x = dequantize_tensor(x_q, self.input_scale, self.input_zp)
                else:
                    x = x.clone()

                sliced_input, input_slice_lsb_to_msb, input_slice_scales = (
                    slice_and_add_noise_to_tensor(
                        tensor=x,
                        tensor_int=x_q,
                        bit_slice_lsb_to_msb=input_slice_lsb_to_msb,
                        num_bits=input_bits,
                        scale=self.input_scale,
                        physical_noise=input_physical_noise,
                        differential_mode=differential_input,
                        detach_cpu=detach_cpu,
                        device=self.device,
                        mode="input",
                    )
                )
                del x_q
            # --- Weights: noisy slices (written into self.weight; restored below) ---
            weight_ideal = self.weight.data.clone()
            sliced_weights_noisy = add_noise_to_sliced_tensors(
                tensor=self.weight.data,
                sliced_tensors=self.sliced_weights,
                bit_slice_lsb_to_msb=self.weight_slice_lsb_to_msb,
                slice_scales=self.weight_slice_scales,
                scale=self.weight_scale,
                num_bits=weight_bits,
                physical_noise=weight_physical_noise,
                differential_weight=differential_weight,
            )

            # --- Output with noisy input and weights ---
            result = super().forward(x)

            self.weight.data = weight_ideal
            del weight_ideal, x

            # --- ADC: per (input slice, weight slice, row group) ---
            if (
                input_bits is not None
                and weight_bits is not None
                and (
                    enable_adc_quantization
                    or enable_adc_clipping
                    or adc_physical_noise is not None
                    or self.probe_outputs
                )
            ):

                # Row groups: num_rows / (R*S) input channels per ADC
                max_Cin_per_adc = round(float(num_rows) / (R * S)) if num_rows else 0
                idx_range = int(np.ceil(Cin / max_Cin_per_adc)) if max_Cin_per_adc else 1
                adc_scale_cache = {}

                for weight_slice, w_s_s in zip(sliced_weights_noisy, self.weight_slice_scales):
                    # Single row group: move each slice to device once
                    if idx_range == 1:
                        weight_matrix = weight_slice.to(device=self.device, dtype=torch.float32)

                    for input_slice, i_s_s in zip(sliced_input, input_slice_scales):
                        if idx_range == 1:
                            input_matrix = input_slice.to(device=self.device, dtype=torch.float32)

                        slice_scale = i_s_s * w_s_s
                        total_scale = slice_scale * self.input_scale * self.weight_scale

                        for idx in range(idx_range):
                            start = idx * max_Cin_per_adc
                            end = min(start + max_Cin_per_adc, Cin) if max_Cin_per_adc else Cin

                            if idx_range != 1:
                                # (N_batch, max_Cin, H, W)
                                input_matrix = input_slice[:, start:end].to(
                                    device=self.device, dtype=torch.float32
                                )
                                # (Cout, max_Cin, R, S)
                                weight_matrix = weight_slice[:, start:end].to(
                                    device=self.device, dtype=torch.float32
                                )

                            pre_adc_result = F.conv2d(
                                input_matrix,
                                weight_matrix,
                                bias=None,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )

                            if idx_range == 1:
                                del input_matrix  # weight_matrix is reused
                            else:
                                del input_matrix, weight_matrix

                            if self.probe_outputs:
                                if idx not in self.outputs[i_s_s][w_s_s]:
                                    self.outputs[i_s_s][w_s_s][idx] = {"low": [], "high": []}
                                sample_fraction = get(params, "sample_fraction")
                                low, high = sample_fraction_tensor(pre_adc_result, sample_fraction)
                                self.outputs[i_s_s][w_s_s][idx]["low"].append(low.detach().cpu())
                                self.outputs[i_s_s][w_s_s][idx]["high"].append(high.detach().cpu())
                                del low, high

                            if self.output_percentiles is None:
                                continue

                            result = apply_adc_noise(
                                result,
                                pre_adc_result,
                                self.output_percentiles,
                                i_s_s,
                                w_s_s,
                                total_scale,
                                params,
                                adc_scale_cache,
                            )

        return result


class NoisyLinear(NoisyConv2d):
    """Linear as a 1x1 `NoisyConv2d`."""

    def __init__(
        self,
        module: nn.Linear,
        in_features,
        out_features,
        input_percentiles: dict = None,
        output_percentiles: dict = None,
        probe_outputs: bool = False,
        name: str = None,
        **params,
    ):
        weight = module.weight.data
        bias = module.bias.data if module.bias is not None else None

        # (out, in) -> (out, in, 1, 1)
        weight = weight.view(*weight.shape, 1, 1)

        super().__init__(
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=(bias is not None),
            padding_mode="zeros",
            device=module.weight.device,
            dtype=module.weight.dtype,
            input_percentiles=input_percentiles,
            output_percentiles=output_percentiles,
            probe_outputs=probe_outputs,
            name=name,
            **params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        result_shape = list(input_shape[:-1]) + [self.out_channels]
        assert input_shape[-1] == self.in_channels, "Input feature dimension must match in_features"
        # (..., in) -> (N, in, 1, 1)
        x = x.view(-1, self.in_channels, 1, 1)
        result = super().forward(x)
        return result.view(*result_shape)


class NoisyConv1D(NoisyConv2d):
    """HF Conv1D or nn.Conv1d as a `NoisyConv2d` with height 1."""

    def __init__(
        self,
        module: nn.Conv1d | Conv1D,
        in_channels,
        out_channels,
        input_percentiles: dict = None,
        output_percentiles: dict = None,
        probe_outputs: bool = False,
        name: str = None,
        **params,
    ):
        self.module_type = type(module)

        # HF Conv1D: weight (in, out) -> (out, in, 1, 1), 1x1 kernel
        if isinstance(module, Conv1D):
            weight = module.weight.data.T
            weight = weight.view(*weight.shape, 1, 1)
            bias = module.bias.data if module.bias is not None else None

            kernel_size = (1, 1)
            stride = (1, 1)
            padding = (0, 0)
            dilation = (1, 1)
            groups = 1
            padding_mode = "zeros"

        # nn.Conv1d: weight (out, in, k) -> (out, in, 1, k)
        elif isinstance(module, nn.Conv1d):
            weight = module.weight.data.unsqueeze(-2)
            bias = module.bias.data if module.bias is not None else None

            kernel_size = (1, module.kernel_size[0])
            stride = (1, module.stride[0])
            padding = (0, module.padding[0])
            dilation = (1, module.dilation[0])
            groups = module.groups
            padding_mode = module.padding_mode

        else:
            raise TypeError(f"Unsupported module type: {type(module)}")

        super().__init__(
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(bias is not None),
            padding_mode=padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            input_percentiles=input_percentiles,
            output_percentiles=output_percentiles,
            probe_outputs=probe_outputs,
            name=name,
            **params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF Conv1D: (B, T, Cin) -> (B, Cin, 1, T)
        if self.module_type == Conv1D:
            if x.dim() == 3:
                x = x.permute(0, 2, 1).unsqueeze(-2)
            elif x.dim() == 4:  # already Conv2d layout
                pass
            else:
                raise ValueError(f"Unexpected input shape for Conv1D: {x.shape}")

        # nn.Conv1d: (B, Cin, T) -> (B, Cin, 1, T)
        elif self.module_type == nn.Conv1d:
            if x.dim() == 3:
                x = x.unsqueeze(-2)
            elif x.dim() == 4:  # already Conv2d layout
                pass
            else:
                raise ValueError(f"Unexpected input shape for nn.Conv1d: {x.shape}")

        result = super().forward(x)

        # Back to Conv1D layout: (B, T, Cout) / (B, Cout, T)
        if self.module_type == Conv1D:
            result = result.squeeze(-2).permute(0, 2, 1)
        elif self.module_type == nn.Conv1d:
            result = result.squeeze(-2)

        return result
