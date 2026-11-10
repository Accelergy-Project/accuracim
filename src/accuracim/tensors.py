"""Attention path: `NoisyTensor`, a `torch.Tensor` subclass that intercepts `@`.

Attention matmuls (Q@K.T, attn@V) have no module to replace. Thus the projection outputs are
wrapped in `NoisyTensor`, which sends `@` through the same array model as `layers`.
`scaled_dot_product_attention` is reimplemented, so that fused SDPA is intercepted too.
SDPA does not pass the attention mask down, so a thread-local context carries it.
"""

import threading
import types
from contextlib import contextmanager
from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from accuracim.params import get, is_percentile_below_100
from accuracim.quantize import (
    apply_adc_noise,
    dequantize_tensor,
    detach_cpu,
    get_quantization_scale,
    quantize_tensor,
    slice_and_add_noise_to_tensor,
)
from accuracim.sampling import sample_fraction_tensor


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Reference SDPA built from `@`, so `NoisyTensor` can intercept its matmuls."""
    device = query.device
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


# Thread-local attention-mask context
_CONTEXT = threading.local()


@contextmanager
def attention_mask_context(mask: torch.Tensor):
    """Expose `mask` to `get_current_attention_mask` within the block."""
    try:
        _CONTEXT.attention_mask = mask
        yield
    finally:
        # Don't leak into later passes
        if hasattr(_CONTEXT, "attention_mask"):
            del _CONTEXT.attention_mask


def get_current_attention_mask() -> torch.Tensor | None:
    """Return the active attention mask, or None."""
    return getattr(_CONTEXT, "attention_mask", None)


def get_current_causal() -> bool:
    """Whether the running model's attention is causal (default True: decoder LMs)."""
    return getattr(_CONTEXT, "causal", True)


def attention_mask_context_patch_model_forward(model: nn.Module):
    """Patch `model.forward` to run inside `attention_mask_context` when `attention_mask` is passed.

    Also exposes whether attention is causal (`get_current_causal`): False if the model's
    attention modules declare `is_causal = False` (e.g. ViT), else True.
    """
    original_forward = model.forward
    causal = not any(getattr(m, "is_causal", None) is False for m in model.modules())

    def patched_forward(self, *args, **kwargs):
        mask = kwargs.get("attention_mask")
        previous = getattr(_CONTEXT, "causal", None)
        _CONTEXT.causal = causal
        try:
            if mask is not None:
                with attention_mask_context(mask):
                    return original_forward(*args, **kwargs)

            return original_forward(*args, **kwargs)
        finally:
            if previous is None:
                del _CONTEXT.causal
            else:
                _CONTEXT.causal = previous

    model.forward = types.MethodType(patched_forward, model)


class NoisyTensor(torch.Tensor):
    """Tensor whose `@` runs through the noisy CIM model; `layer_name` tracks provenance."""

    _enabled = True
    _custom_flags = [
        "layer_name",
        "params",
        "probe_inputs",
        "probe_outputs",
        "input_percentiles",
        "output_percentiles",
        "store_dict",
    ]  # keep in sync with __new__

    @staticmethod
    def __new__(cls, data: torch.Tensor, *args, **kwargs):
        instance = torch.Tensor._make_subclass(cls, data)

        instance.layer_name = kwargs.get("layer_name", None)
        instance.params = kwargs.get("params", {})
        instance.probe_inputs = kwargs.get("probe_inputs", False)
        instance.probe_outputs = kwargs.get("probe_outputs", False)
        instance.input_percentiles = kwargs.get("input_percentiles", None)
        instance.output_percentiles = kwargs.get("output_percentiles", None)
        instance.store_dict = kwargs.get("store_dict", {})

        return instance

    @classmethod
    def _attrs_from_sequence(cls, args) -> dict:
        """Attrs for an op whose `NoisyTensor`s sit inside a sequence arg, e.g. `torch.cat((a, b))`.

        Settings come from the first one. The name only if every part comes from one layer:
        a KV-cache `cat([empty, key])` stays unnamed, as the calibrated ranges are keyed.
        """
        for a in args:
            if not isinstance(a, (tuple, list)):
                continue
            tensors = [t for t in a if isinstance(t, torch.Tensor)]
            noisy = [t for t in tensors if isinstance(t, NoisyTensor)]
            if not noisy:
                continue
            attrs = {
                flag: getattr(noisy[0], flag)
                for flag in cls._custom_flags
                if flag != "layer_name" and getattr(noisy[0], flag, None) is not None
            }
            names = {getattr(t, "layer_name", None) for t in noisy}
            if len(noisy) == len(tensors) and len(names) == 1 and None not in names:
                attrs["layer_name"] = names.pop()
            return attrs
        return {}

    @classmethod
    def _wrap_tensors(cls, obj, args):
        """Recursively wrap output tensors as `NoisyTensor`, copying attrs from the first `NoisyTensor` arg."""
        if isinstance(obj, torch.Tensor):
            attrs = {}
            for a in args:
                if isinstance(a, NoisyTensor):
                    for flag in cls._custom_flags:
                        value = getattr(a, flag, None)
                        if value is not None:
                            attrs[flag] = value
                    break
            else:
                attrs = cls._attrs_from_sequence(args)
            # A `layer_name` set by `@` takes precedence
            obj_layer_name = getattr(obj, "layer_name", None)
            if obj_layer_name is not None:
                attrs["layer_name"] = obj_layer_name
            return NoisyTensor(obj, **attrs)
        elif isinstance(obj, (tuple, list)):
            return type(obj)(cls._wrap_tensors(o, args) for o in obj)
        elif isinstance(obj, dict):
            return {k: cls._wrap_tensors(v, args) for k, v in obj.items()}
        return obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if not cls._enabled:
            args = [a.as_subclass(torch.Tensor) if isinstance(a, cls) else a for a in args]
            return func(*args, **(kwargs or {}))

        # Decompose SDPA so its matmuls are intercepted
        if "scaled_dot_product_attention" in func.__name__:
            return scaled_dot_product_attention(*args, **(kwargs or {}))

        elif func in {torch.matmul}:
            a, b = args
            if isinstance(a, cls):
                return a.__matmul__(b)
            elif isinstance(b, cls):
                return b.__rmatmul__(a)
        else:
            ret = super().__torch_function__(func, types, args, kwargs)

        ret = cls._wrap_tensors(ret, args)
        return ret

    def __matmul__(self, other):
        """`weight_operand_loc` 'left': compute (B.T @ A.T).T so the weight is on the right."""
        params = getattr(self, "params", {})
        weight_operand_loc = getattr(params, "weight_operand_loc", "right")

        if weight_operand_loc == "left":
            self_T = self.transpose(-2, -1)
            other_T = other.transpose(-2, -1)

            result = other_T.matmul_ordered(self_T)

            return result.transpose(-2, -1)

        else:
            return self.matmul_ordered(other)

    def matmul_ordered(self, other):
        """Noisy `self @ other` (input @ weight); the result's `layer_name` is (self name, other name)."""
        self_name = getattr(self, "layer_name", None)
        other_name = getattr(other, "layer_name", None)
        out_name = (self_name, other_name)

        # attn @ V has an operand named by a tuple (from Q @ K.T); Q @ K.T does not
        is_attention_output_operation = isinstance(self_name, tuple) or isinstance(
            other_name, tuple
        )
        is_attention_weight_operation = not is_attention_output_operation

        if not isinstance(other, torch.Tensor):
            return NotImplemented

        params = getattr(self, "params", {})

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
        weight_slice_lsb_to_msb = get(params, "weight_slice_lsb_to_msb")
        enable_weight_quantization = get(params, "enable_weight_quantization")
        enable_weight_clipping = get(params, "enable_weight_clipping")
        weight_physical_noise = get(params, "weight_physical_noise")
        differential_weight = get(params, "differential_weight")
        weight_clip_percentile = get(params, "weight_clip_percentile")

        # ADC
        adc_physical_noise = get(params, "adc_physical_noise")
        enable_adc_clipping = get(params, "enable_adc_clipping")
        enable_adc_quantization = get(params, "enable_adc_quantization")
        num_rows = get(params, "num_rows")

        probe_inputs = getattr(self, "probe_inputs", False) or getattr(other, "probe_inputs", False)
        input_percentiles_total = getattr(self, "input_percentiles", None)

        x, weight = self, other
        if input_percentiles_total is not None:
            input_percentiles = input_percentiles_total[out_name]["input"]
            weight_percentiles = input_percentiles_total[out_name]["weight"]

        # Padding mask (see `attention_mask_context`)
        mask = get_current_attention_mask()

        # Probe mode: record operand tails (padding masked out), then a plain matmul
        if probe_inputs:
            store_dict = getattr(self, "store_dict", {})
            if out_name not in store_dict:
                store_dict[out_name] = {k: {"low": [], "high": []} for k in ["input", "weight"]}
            inputs = store_dict[out_name]

            input_sampled, weight_sampled = self, other
            if mask is not None:
                B, S = mask.shape
                if is_attention_weight_operation:  # (B, H, S(mask), D) @ (B, H, D, S(mask))
                    input_sampled = torch.masked_select(input_sampled, mask.view(B, 1, S, 1).bool())
                    weight_sampled = torch.masked_select(
                        weight_sampled, mask.view(B, 1, 1, S).bool()
                    )

                else:  # Attention output, (B, H, S(mask), S(mask)) @ (B, H, S(mask), D)
                    square_mask = mask.view(B, 1, S, 1) & mask.view(B, 1, 1, S)
                    input_sampled = torch.masked_select(input_sampled, square_mask.bool())
                    weight_sampled = torch.masked_select(
                        weight_sampled, mask.view(B, 1, S, 1).bool()
                    )

            for k, tensor in zip(["input", "weight"], [input_sampled, weight_sampled]):
                low, high = sample_fraction_tensor(tensor, params["sample_fraction"])
                inputs[k]["low"].append(low.detach().cpu())
                inputs[k]["high"].append(high.detach().cpu())

            result = super().__matmul__(other)
            return self._create_named_noisytensor(result, layer_name=out_name)

        input_scale, input_zp, input_clip_range = get_quantization_scale(
            input_bits,
            input_percentiles,
            input_clip_percentile,
            symmetrical_range=differential_input,
        )

        # Differential cells: symmetric weight range (-|Wmax|, |Wmax|)
        weight_scale, weight_zp, weight_clip_range = get_quantization_scale(
            weight_bits,
            weight_percentiles,
            weight_clip_percentile,
            symmetrical_range=differential_weight,
        )

        with torch.no_grad():
            # --- Input: clip, quantize, slice + DAC noise ---
            low, high = input_clip_range
            if enable_input_clipping and not (low is None and high is None):
                if is_percentile_below_100(input_clip_percentile):
                    x = x.clamp(low, high)

            if input_bits is not None:
                x_q = quantize_tensor(x, input_scale, input_zp).to(dtype=torch.int)
                if enable_input_quantization:
                    x = dequantize_tensor(x_q, input_scale, input_zp)
                else:
                    x = x.clone()

                sliced_inputs, input_slice_lsb_to_msb, input_slice_scales = (
                    slice_and_add_noise_to_tensor(
                        tensor=x,
                        tensor_int=x_q,
                        bit_slice_lsb_to_msb=input_slice_lsb_to_msb,
                        num_bits=input_bits,
                        scale=input_scale,
                        physical_noise=input_physical_noise,
                        differential_mode=differential_input,
                        detach_cpu=detach_cpu,
                        device=self.device,
                        mode="input",
                    )
                )
                del x_q

            # --- Weight: clip, quantize, slice + conductance noise ---
            low, high = weight_clip_range
            if enable_weight_clipping and not (low is None and high is None):
                if is_percentile_below_100(weight_clip_percentile):
                    weight = weight.clamp(low, high)

            if weight_bits is not None:
                weight_q = quantize_tensor(weight, weight_scale, weight_zp).to(dtype=torch.int)
                # Dequantize only if enabled, isolating conductance noise from quantization error
                if enable_weight_quantization:
                    weight = dequantize_tensor(weight_q, weight_scale, weight_zp)
                else:
                    weight = weight.clone()

                sliced_weights, weight_slice_lsb_to_msb, weight_slice_scales = (
                    slice_and_add_noise_to_tensor(
                        tensor=weight,
                        tensor_int=weight_q,
                        bit_slice_lsb_to_msb=weight_slice_lsb_to_msb,
                        num_bits=weight_bits,
                        scale=weight_scale,
                        physical_noise=weight_physical_noise,
                        differential_mode=differential_weight,
                        detach_cpu=detach_cpu,
                        device=self.device,
                        mode="weight",
                    )
                )
                del weight_q

            result = super(NoisyTensor, x).__matmul__(weight)

            # --- ADC: per (input slice, weight slice, row group) ---
            if (
                input_bits is not None
                and weight_bits is not None
                and (
                    enable_adc_quantization
                    or enable_adc_clipping
                    or adc_physical_noise is not None
                    or getattr(self, "probe_outputs", False)
                )
            ):

                # vvm_len: reduction length
                if is_attention_weight_operation:
                    B, H, S, D = self.shape
                    vvm_len = D
                    assert other.shape == (
                        B,
                        H,
                        D,
                        S,
                    ), f"Incompatible matrix shapes for multiplication. Input {self.shape}, Weight {other.shape}"
                else:  # Attention output
                    B, H, S, S = self.shape
                    assert other.shape[:-1] == (
                        B,
                        H,
                        S,
                    ), f"Incompatible matrix shapes for multiplication. Input {self.shape}, Weight {other.shape}"
                    D = other.shape[-1]
                    vvm_len = S

                # Row groups of `num_rows`
                idx_range = int(np.ceil(vvm_len / num_rows)) if num_rows else 1
                adc_scale_cache = {}

                for weight_slice, w_s_s in zip(sliced_weights, weight_slice_scales):
                    # Single row group: move each slice to device once
                    if idx_range == 1:
                        weight_matrix = weight_slice.to(device=self.device, dtype=torch.float32)

                    for input_slice, i_s_s in zip(sliced_inputs, input_slice_scales):
                        slice_scale = i_s_s * w_s_s
                        total_scale = float(slice_scale * input_scale * weight_scale)

                        if idx_range == 1:
                            input_matrix = input_slice.to(dtype=torch.float32, device=self.device)

                        for idx in range(idx_range):
                            start = idx * num_rows if num_rows else 0
                            end = min(start + num_rows, vvm_len) if num_rows else vvm_len

                            if idx_range != 1:
                                input_matrix = input_slice[..., start:end].to(
                                    dtype=torch.float32, device=self.device
                                )
                                weight_matrix = weight_slice[..., start:end, :].to(
                                    dtype=torch.float32, device=self.device
                                )

                            pre_adc_result = super(NoisyTensor, input_matrix).__matmul__(
                                weight_matrix
                            )

                            # weight_matrix is reused when idx_range == 1
                            if idx_range != 1:
                                del weight_matrix
                            del input_matrix

                            if getattr(self, "probe_outputs", False):
                                store_dict = getattr(self, "store_dict", {})
                                if idx not in store_dict[out_name][i_s_s][w_s_s]:
                                    store_dict[out_name][i_s_s][w_s_s][idx] = {
                                        "low": [],
                                        "high": [],
                                    }
                                outputs = store_dict[out_name][i_s_s][w_s_s][idx]
                                sample_fraction = get(params, "sample_fraction")

                                final_mask = None
                                # Padding mask
                                if mask is not None:
                                    if is_attention_weight_operation:
                                        # (B, H, S, S)
                                        final_mask = (
                                            mask.view(B, 1, S, 1) & mask.view(B, 1, 1, S)
                                        ).bool()
                                    else:
                                        # (B, H, S, D)
                                        final_mask = mask.view(B, 1, S, 1).bool()

                                # Q @ K.T of a causal model: lower-triangular mask too
                                if is_attention_weight_operation and get_current_causal():
                                    causal_mask = torch.tril(
                                        torch.ones((S, S), device=pre_adc_result.device)
                                    ).bool()
                                    causal_mask = causal_mask.view(1, 1, S, S)
                                    final_mask = (
                                        causal_mask
                                        if final_mask is None
                                        else causal_mask & final_mask
                                    )

                                # No mask (e.g. ViT A·V): every value counts
                                sampled_pre_adc_result = pre_adc_result
                                if final_mask is not None:
                                    sampled_pre_adc_result = torch.masked_select(
                                        pre_adc_result, final_mask
                                    )

                                if sampled_pre_adc_result.numel() > 0:
                                    low, high = sample_fraction_tensor(
                                        sampled_pre_adc_result, sample_fraction
                                    )
                                    outputs["low"].append(low.detach().cpu())
                                    outputs["high"].append(high.detach().cpu())
                                    del low, high
                                del sampled_pre_adc_result

                            output_percentiles_total = getattr(self, "output_percentiles", None)
                            if output_percentiles_total is None:
                                continue
                            output_percentiles = output_percentiles_total[out_name]

                            result = apply_adc_noise(
                                result,
                                pre_adc_result,
                                output_percentiles,
                                i_s_s,
                                w_s_s,
                                total_scale,
                                params,
                                adc_scale_cache,
                            )

        return self._create_named_noisytensor(result, layer_name=out_name)

    def _create_named_noisytensor(self, tensor, layer_name: str = None):
        """Wrap `tensor` as a `NoisyTensor` with this tensor's attrs, overriding `layer_name` if given."""
        constructor_kwargs = {}
        for flag in self._custom_flags:
            if flag == "layer_name" and layer_name is not None:
                constructor_kwargs[flag] = layer_name
            else:
                constructor_kwargs[flag] = getattr(self, flag, None)
        return NoisyTensor(tensor, **constructor_kwargs)

    def __rmatmul__(self, other):
        raise NotImplementedError(
            "Reverse matmul shouldn't be triggered since Q & K should be tracked normally"
        )


def return_noisy_tensor_hook(layer_name=None, **kwargs):
    """Return a forward hook that wraps the layer output as a `NoisyTensor` named `layer_name`."""

    def hook(module, input, output):
        if isinstance(output, NoisyTensor):
            # Settings may have been lost on the way in (e.g. `torch.cat` of a tuple)
            output.layer_name = layer_name
            for flag, value in kwargs.items():
                setattr(output, flag, value)
            return output
        elif isinstance(output, torch.Tensor):
            return NoisyTensor(output, layer_name=layer_name, **kwargs)
        return output

    return hook
