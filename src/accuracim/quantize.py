"""Fixed-point model of a CIM array.

Each matmul goes through these steps: clipping range (from calibrated percentiles),
quantization scale, quantization, bit slices (one for each physical write), analog noise.
The `add_*_error` functions add quantization and clipping error directly to the result,
so that the pre-ADC partial sums are not copied.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from accuracim.noise import apply_noise
from accuracim.params import close_eq, get, is_percentile_below_100
from accuracim.sampling import torch_quantile

logger = logging.getLogger(__name__)


def get_clipping_range(
    output_percentiles: dict = None,
    clip_percentile: float | tuple[float] = 100.0,
    interpolation: str = "linear",  # 'quadratic', 'cubic', 'nearest'
):
    """Return (low, high) at `clip_percentile`, interpolated over log tail percentiles.

    Scalar percentile: split evenly between tails (ReLU outputs: high tail only).
    Tuple: (low, high) percentiles.
    """
    check_relu = True
    for key, value in output_percentiles.items():
        if key <= 50 and not close_eq(value, 0.0):
            # Not ReLU output
            check_relu = False

    if clip_percentile is None:
        return (None, None)
    elif isinstance(clip_percentile, tuple):
        lp, hp = clip_percentile
    # ReLU: lower bound is 0
    elif check_relu:
        lp = 0
        hp = clip_percentile
    else:
        lp = (100.0 - clip_percentile) / 2
        hp = 100.0 - lp

    def interpolate_percentile_values(p):
        # None: unbounded
        if p == None:
            return None
        elif float(p) in output_percentiles:
            return output_percentiles[float(p)]

        # Work in tail percentiles (99.5 -> 0.5)
        if p <= 50:
            filtered = {k: v for k, v in output_percentiles.items() if 0 < k <= 50}
        else:
            p = 100 - p
            filtered = {100 - k: v for k, v in output_percentiles.items() if 100 > k > 50}

        percentiles = sorted(filtered.keys())
        values = [filtered[k] for k in percentiles]
        log_percentiles = [np.log10(k) for k in percentiles]
        target_p = np.log10(p)

        f = interp1d(
            log_percentiles,
            values,
            kind=interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )
        output = f(target_p).item()
        # Don't extrapolate past the observed min/max
        if p <= 50:
            min_value = output_percentiles[min(output_percentiles.keys())]
            output = max(min_value, output)
        else:
            max_value = output_percentiles[max(output_percentiles.keys())]
            output = min(max_value, output)
        return output

    return tuple(interpolate_percentile_values(p) for p in (lp, hp))


def get_symmetrical_clipping_range(
    output_percentiles: dict,
    clip_percentile: float | tuple[float] = 100.0,
    interpolation: str = "linear",  # 'quadratic', 'cubic', 'nearest'
):
    """Return (-v, v) clipping `100 - clip_percentile` total tail mass across both tails.

    ReLU outputs fall back to `get_clipping_range`.
    """
    if clip_percentile is None:
        return (None, None)

    check_relu = True
    for key, value in output_percentiles.items():
        if key <= 50 and not close_eq(value, 0.0):
            # Not ReLU output
            check_relu = False
    if check_relu:
        return get_clipping_range(output_percentiles, clip_percentile, interpolation)

    # Tuple: use its total width
    if isinstance(clip_percentile, tuple):
        lp, hp = clip_percentile
        if lp is None or hp is None:
            return (None, None)
        clip_percentile = hp - lp

    functions = {}
    min_value, max_value = 0, 0

    # Per tail: |value| -> log10(tail percentile), e.g. 99.5 -> 0.5
    for loc in ["low", "high"]:
        if loc == "low":
            filtered = {k: v for k, v in output_percentiles.items() if 0 < k <= 50}
            percentiles = sorted(filtered.keys())
            values = [-filtered[k] for k in percentiles]
        else:
            filtered = {100 - k: v for k, v in output_percentiles.items() if 100 > k > 50}
            percentiles = sorted(filtered.keys())
            values = [filtered[k] for k in percentiles]

        log_percentiles = [np.log10(k) for k in percentiles]

        f = interp1d(
            values,
            log_percentiles,
            kind=interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )
        functions[loc] = f

        # Search bracket bound
        if loc == "low":
            min_value = max(values) if values else 0
        else:
            max_value = max(values) if values else 0

    # Solve tail_low(v) + tail_high(v) = 100 - clip_percentile
    combined_function = (
        lambda value, target: (10 ** functions["low"](value) + 10 ** functions["high"](value))
        - target
    )
    target = 100 - clip_percentile

    try:
        # Pinned tails (e.g. GELU min, saturated pixels) repeat a value: the interpolation
        # slope divides by zero and extrapolates to 10**huge = inf. Harmless for the root.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            solution = root_scalar(
                combined_function,
                args=(target,),
                method="brentq",
                bracket=[0, max(min_value, max_value) * 1.1],
            )
        result = solution.root
    except:
        result = max(min_value, max_value)
    return (-result, result)


def get_quantization_scale(
    num_bits: int = None,
    output_percentiles: dict = None,
    clip_percentile: float | tuple[float] = 100.0,
    symmetrical_range: bool = False,
):
    """Return (scale, zero_point, (low, high)); scale/zero_point are None if bits or range are unset."""
    if symmetrical_range:
        clip_range = get_symmetrical_clipping_range(output_percentiles, clip_percentile)
    else:
        clip_range = get_clipping_range(output_percentiles, clip_percentile)
    low, high = clip_range
    zero_point = 0

    if low is None or high is None or num_bits is None:
        return (None, None, clip_range)
    else:
        min_max_range = high - low
        scale = float(min_max_range) / (1 << num_bits)

        if not symmetrical_range and not close_eq(low, 0.0):
            # Zero point keeps the low bound exact (e.g. GELU min ~ -0.17)
            zero_point = round(-low / scale)

        return (scale, zero_point, (low, high))


def get_adc_scale(
    adc_bits: int = None,
    output_percentiles: dict = None,
    adc_clip_percentile: float | tuple[float] = 100.0,
    adc_reference_mode: str = "layer",
    i_s_s: float | int = None,
    w_s_s: float | int = None,
    cash: dict = None,
):
    """Return (adc_scale, (low, high)) from the percentile table `adc_reference_mode` selects; memoized in `cash`."""
    # Slice indices that key the percentile table
    if adc_reference_mode == "layer":
        i_s_s, w_s_s = (None, None)
    elif adc_reference_mode == "input_slice":
        i_s_s, w_s_s = (i_s_s, None)
    elif adc_reference_mode == "weight_slice":
        i_s_s, w_s_s = (None, w_s_s)
    elif adc_reference_mode == "input_weight_slice":
        i_s_s, w_s_s = (i_s_s, w_s_s)
    else:
        raise ValueError(
            f"adc_reference_mode={adc_reference_mode} not supported--use 'layer', 'input_slice', 'weight_slice', 'input_weight_slice'"
        )

    if cash is None:
        cash = {}

    key = (i_s_s, w_s_s)
    if key in cash:
        return cash[key]
    else:
        adc_percentiles = output_percentiles[i_s_s][w_s_s]
        clip_range = get_clipping_range(adc_percentiles, adc_clip_percentile)
        low, high = clip_range

        if low is None or high is None or adc_bits is None:
            cash[key] = (None, clip_range)
            return cash[key]
        else:
            adc_range = high - low
            adc_scale = float(adc_range) / (1 << adc_bits)
            cash[key] = (adc_scale, clip_range)

            return cash[key]


def quantize_tensor(tensor, scale, zp=0):
    """Return round(tensor / scale) + zp; a tensor `scale` is per-channel (dim 0)."""
    if isinstance(scale, (float, int, np.number)):
        return torch.round(tensor / scale) + zp
    else:
        scale_inv = 1 / scale

        while zp.dim() < tensor.dim():
            zp = zp.unsqueeze(-1)

        return (torch.round(torch.einsum("i...,i->i...", tensor, scale_inv)) + zp).to(
            dtype=torch.int
        )


def dequantize_tensor(tensor, scale, zp=0):
    """Inverse of `quantize_tensor`: (tensor - zp) * scale."""
    if isinstance(scale, float):
        return (tensor - zp) * scale
    else:
        scale_inv = 1 / scale

        while zp.dim() < tensor.dim():
            zp = zp.unsqueeze(-1)

        return torch.einsum("i...,i->i...", tensor - zp, scale_inv)


def slice_tensor_lsb_to_msb(
    tensor, bit_slice_lsb_to_msb: int | list[int], num_bits: int, detach: bool = False, device=None
):
    """Split an integer tensor into bit slices, LSB first.

    Returns:
    tuple: (slices, slice widths, slice scales 2^offset)
    """
    # Int width: repeat up to num_bits
    if isinstance(bit_slice_lsb_to_msb, int):
        slicing_pattern = []
        current_bits = 0

        while current_bits < num_bits:
            slice_size = min(num_bits - current_bits, bit_slice_lsb_to_msb)
            slicing_pattern.append(slice_size)
            current_bits += slice_size

        # Remainder goes to the LSB slice; MSB slices stay full width
        bit_slice_lsb_to_msb = list(reversed(slicing_pattern))

    elif bit_slice_lsb_to_msb is None:
        bit_slice_lsb_to_msb = [num_bits]

    sliced_weights = []
    slice_scales = []
    bit_offset = 0
    tensor = tensor.to(dtype=torch.int)

    for bits in bit_slice_lsb_to_msb:
        mask = (1 << bits) - 1
        sliced = (tensor >> bit_offset) & mask

        if detach:
            sliced = sliced.detach()
        if device is not None:
            sliced = sliced.to(device)

        sliced_weights.append(sliced)
        slice_scales.append(1 << bit_offset)
        bit_offset += bits

    return sliced_weights, bit_slice_lsb_to_msb, slice_scales


def slice_weights(module: nn.Module):
    """Clip and quantize `module.weight` in place, bit-slice it, then apply drift.

    Returns:
    tuple: (slices, slice widths, slice scales, weight_scale, weight_range); all None if unquantized.
    """
    params = module.params

    weight_bits = get(params, "weight_bits")
    enable_weight_quantization = get(params, "enable_weight_quantization")
    weight_clip_percentile = get(params, "weight_clip_percentile")
    enable_weight_clipping = get(params, "enable_weight_clipping")
    weight_slice_lsb_to_msb = get(params, "weight_slice_lsb_to_msb")
    differential_weight = get(params, "differential_weight")
    weight = module.weight.data

    with torch.no_grad():
        # Symmetric range: 2 * max|w|, or 2 * geometric mean of the percentile bounds
        if close_eq(weight_clip_percentile, 100.0):
            weight_range = abs(weight).max() * 2
        else:
            if isinstance(weight_clip_percentile, tuple):
                low, high = weight_clip_percentile
            else:
                low = (100.0 - weight_clip_percentile) / 2
                high = 100.0 - low
            low_w = abs(torch_quantile(weight, low / 100.0))
            high_w = abs(torch_quantile(weight, high / 100.0))
            weight_range = ((low_w * high_w) ** 0.5) * 2
            # TODO: Non-symmetric weights - is it physically meaningful to implement?

        if enable_weight_clipping and not close_eq(weight_clip_percentile, 100):
            module.weight.data.clamp_(-weight_range / 2, weight_range / 2)

        if weight_bits is not None:
            weight_scale = (weight_range / (1 << weight_bits)).item()
            weight_zp = 0
            weight = quantize_tensor(module.weight.data, weight_scale, weight_zp)

            # Dequantize only if enabled, isolating conductance noise from quantization error
            if enable_weight_quantization:
                module.weight.data = dequantize_tensor(weight, weight_scale, weight_zp)

            if not differential_weight:
                weight += 1 << (weight_bits - 1)  # offset to non-negative

            # Slice before conductance noise
            if weight_slice_lsb_to_msb is not None:
                assert (
                    weight_bits is not None
                ), f"weight_bits must be specified if weight_slice_lsb_to_msb = {weight_slice_lsb_to_msb}"
                sliced_weights, weight_slice_lsb_to_msb, weight_slice_scales = (
                    slice_tensor_lsb_to_msb(abs(weight), weight_slice_lsb_to_msb, weight_bits)
                )
                # Restore sign: differential cells encode negatives as negative current
                weight_neg_loc = module.weight.data < 0
                sliced_weights = [
                    torch.where(weight_neg_loc, -weight_slice, weight_slice).detach().cpu()
                    for weight_slice in sliced_weights
                ]

            else:
                sliced_weights = [weight.detach().cpu()]
                weight_slice_lsb_to_msb = [weight_bits]
                weight_slice_scales = [1]

            sliced_weights = _apply_weight_drift(module, sliced_weights)
            sliced_weights = [_compact_slice(s) for s in sliced_weights]
            return (
                sliced_weights,
                weight_slice_lsb_to_msb,
                weight_slice_scales,
                weight_scale,
                weight_range,
            )

        else:
            # Unquantized: no cells to slice, but the weights still decay.
            _apply_weight_drift(module, [])
            return None, None, None, None, None


def _compact_slice(tensor: torch.Tensor) -> torch.Tensor:
    """Return an integer slice in the smallest integer dtype that holds it exactly.

    Float slices (drifted, or an unsliced quantized weight) are returned unchanged.
    """
    if tensor.is_floating_point() or tensor.dtype == torch.bool or tensor.numel() == 0:
        return tensor
    bound = tensor.abs().max().item()
    for dtype in (torch.int8, torch.int16, torch.int32):
        if bound <= torch.iinfo(dtype).max:
            return tensor.to(dtype)
    return tensor


# Where weight slices live between forward passes: "auto", "cpu", or "gpu" (`run.weight_slices`)
_WEIGHT_SLICE_PLACEMENT = "auto"


def set_weight_slice_placement(placement: str) -> None:
    """Set where `place_weight_slices` puts weight slices: "auto", "cpu", or "gpu"."""
    global _WEIGHT_SLICE_PLACEMENT
    if placement not in ("auto", "cpu", "gpu"):
        raise ValueError(f"run.weight_slices must be auto, cpu, or gpu (got {placement!r})")
    _WEIGHT_SLICE_PLACEMENT = placement


def place_weight_slices(model: nn.Module) -> None:
    """Keep the noisy layers' weight slices on CPU, or move them to the layers' GPU.

    "auto": GPU if all slices, plus the largest layer's float copies made each forward pass,
    fit in half the free GPU memory. The other half is left for activations, which scale
    with the batch: input slices and every input x weight slice pre-ADC output.
    """
    layers = [m for m in model.modules() if getattr(m, "sliced_weights", None)]
    if not layers or _WEIGHT_SLICE_PLACEMENT == "cpu":
        return
    device = torch.device(layers[0].device)
    if device.type != "cuda":
        return

    stored = sum(s.numel() * s.element_size() for m in layers for s in m.sliced_weights)
    # Noisy float32 slices, and their float32 copies in the ADC loop
    per_pass = max(sum(s.numel() for s in m.sliced_weights) * 4 * 2 for m in layers)
    if _WEIGHT_SLICE_PLACEMENT == "auto":
        free, _ = torch.cuda.mem_get_info(device)
        on_gpu = stored + per_pass <= 0.5 * free
        logger.info(
            f"weight slices on {'gpu' if on_gpu else 'cpu'} · {stored / 2**30:.2f} GB "
            f"(+{per_pass / 2**30:.2f} GB per pass) · {free / 2**30:.1f} GB free"
        )
        if not on_gpu:
            return

    for m in layers:
        m.sliced_weights = [s.to(m.device) for s in m.sliced_weights]
    model._weight_slices_on_gpu = True


def _apply_weight_drift(
    module: nn.Module, sliced_weights: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Decay the programmed cells once, in place.

    Runs at layer construction (via `slice_weights`), when cells are written: after
    quantization, and not re-quantized (drift is analog). Slices and dense weight
    both decay, keeping the pre-ADC path and `super().forward(x)` consistent;
    `forward` restores this decayed weight each pass, so decay does not compound.
    """
    drift = get(module.params, "weight_drift")
    if drift is None:
        return sliced_weights

    assert getattr(
        drift, "stationary", False
    ), f"weight_drift must be a stationary profile, got {type(drift).__name__}"
    with torch.no_grad():
        # Slices carry the weight's sign, so this shrinks toward zero.
        module.weight.data.add_(drift(module.weight.data))
        return [s + drift(s) for s in sliced_weights]


def add_noise_to_sliced_tensors(
    tensor: torch.Tensor,
    sliced_tensors: list[torch.Tensor],
    bit_slice_lsb_to_msb: list[int],
    slice_scales: list,
    scale: float,
    num_bits: int,
    physical_noise: callable = None,
    differential_weight: bool = False,
):
    """Add per-slice physical noise to `tensor` in place; return the noisy slices (None if no bits)."""
    device = tensor.device
    sliced_tensors_noisy = []

    with torch.no_grad():
        if num_bits is None:
            return None

        elif physical_noise is not None:
            for sliced_tensor, slice_scale, bit_slice in zip(
                sliced_tensors, slice_scales, bit_slice_lsb_to_msb
            ):
                low, high = 0, (1 << bit_slice) - 1
                home = sliced_tensor.device  # noisy slice stays where its slice lives
                sliced_tensor = sliced_tensor.to(dtype=tensor.dtype, device=device)
                sliced_error = apply_noise(physical_noise, sliced_tensor, low=low, high=high)

                # Differential: the paired cell adds noise too
                if differential_weight:
                    sliced_error = sliced_error + apply_noise(
                        physical_noise, torch.zeros_like(sliced_tensor), low=low, high=high
                    )

                sliced_tensors_noisy.append((sliced_tensor + sliced_error).detach().to(home))
                tensor.add_(sliced_error * (slice_scale * scale))

        else:
            sliced_tensors_noisy = sliced_tensors

    return sliced_tensors_noisy


def _expand_slice_pattern(bit_slice_lsb_to_msb, num_bits):
    """Return slice widths LSB->MSB: an int width repeats up to `num_bits`; None -> one slice."""
    if isinstance(bit_slice_lsb_to_msb, int):
        slicing_pattern = []
        current_bits = 0
        while current_bits < num_bits:
            slice_size = min(num_bits - current_bits, bit_slice_lsb_to_msb)
            slicing_pattern.append(slice_size)
            current_bits += slice_size
        # Remainder goes to the LSB slice
        return list(reversed(slicing_pattern))
    return [num_bits] if bit_slice_lsb_to_msb is None else list(bit_slice_lsb_to_msb)


# Move slices to CPU to save GPU memory
detach_cpu = False


def slice_and_add_noise_to_tensor(
    tensor: torch.Tensor,
    tensor_int: torch.Tensor,
    bit_slice_lsb_to_msb: int | list[int],
    num_bits: int,
    scale: float = 1.0,
    physical_noise: callable = None,
    differential_mode: bool = False,
    detach_cpu: bool = True,
    device: torch.device = None,
    mode: str = "weight",  # "weight" or "input"
):
    """Bit-slice `tensor_int`, adding per-slice physical noise to the slices and to `tensor` (in place).

    Signed values take two cycles per slice, positive then negative (slice scale -s),
    for inputs in differential mode or weights in non-differential mode.

    Returns:
    tuple: (slices, slice widths, slice scales); slices on CPU if `detach_cpu`.
    """

    if device is None:
        device = tensor.device

    pattern = _expand_slice_pattern(bit_slice_lsb_to_msb, num_bits)

    result_slices = []
    slice_scales = []

    neg_loc = tensor_int < 0
    neg_exist = neg_loc.any()
    base = abs(tensor_int.to(dtype=torch.int))
    bit_offset = 0

    for bits in pattern:
        mask = (1 << bits) - 1
        sliced_base = (base >> bit_offset) & mask
        slice_scale_base = 1 << bit_offset

        # Two cycles: positive, then negative
        if (mode == "input" and neg_exist and differential_mode) or (
            mode == "weight" and neg_exist and not differential_mode
        ):
            sliced_list = [
                torch.where(~neg_loc, sliced_base, torch.zeros_like(sliced_base)),
                torch.where(neg_loc, sliced_base, torch.zeros_like(sliced_base)),
            ]
            slice_scale_list = [slice_scale_base, -slice_scale_base]
        else:
            sliced_list = [sliced_base]
            slice_scale_list = [slice_scale_base]

        for sliced, slice_scale in zip(sliced_list, slice_scale_list):
            if physical_noise is not None:
                sliced_f = sliced.to(dtype=torch.float32)
                error = apply_noise(physical_noise, sliced_f, low=0, high=mask)

                # Differential: the paired cell adds noise too
                if differential_mode:
                    error = error + apply_noise(
                        physical_noise, torch.zeros_like(sliced_f), low=0, high=mask
                    )

                # Noise into the slice and the dense tensor
                sliced_f = sliced_f + error
                tensor.add_(error * abs(slice_scale * scale))
            else:
                sliced_f = sliced.to(dtype=torch.float32)

            if detach_cpu:
                result_slices.append(sliced_f.detach().cpu())
            else:
                result_slices.append(sliced_f.detach())
            slice_scales.append(slice_scale)

            # Free early
            del sliced
            if "sliced_f" in locals():
                del sliced_f
        bit_offset += bits

    return result_slices, pattern, slice_scales


# Error kernels (TorchScript-compiled below)
def add_quantization_error_no_jit_(
    result_tensor: torch.Tensor, tensor: torch.Tensor, scale: float, error_scale: float = 1.0
):
    """Return result + (rounding error of `tensor` at `scale`) * error_scale."""
    tensor = tensor.to(dtype=result_tensor.dtype)
    quantized_tensor = torch.round(tensor / scale) * scale
    if error_scale == 1.0:
        return result_tensor + quantized_tensor - tensor
    return result_tensor + (quantized_tensor - tensor) * error_scale


def add_clipping_error_no_jit_(
    result_tensor: torch.Tensor,
    tensor: torch.Tensor,
    low: float,
    high: float,
    error_scale: float = 1.0,
):
    """Return result + (clipping error of `tensor` to [low, high]) * error_scale."""
    tensor = tensor.to(dtype=result_tensor.dtype)
    if error_scale == 1.0:
        return result_tensor + torch.clamp(tensor, min=low, max=high) - tensor
    return result_tensor + (torch.clamp(tensor, min=low, max=high) - tensor) * error_scale


def add_independent_noises_no_jit_(std_1: torch.Tensor, std_2: torch.Tensor):
    """Combined std of two independent noises."""
    return torch.sqrt(std_1**2 + std_2**2)


add_quantization_error: callable = torch.jit.script(add_quantization_error_no_jit_)
add_clipping_error: callable = torch.jit.script(add_clipping_error_no_jit_)
add_independent_noises: callable = torch.jit.script(add_independent_noises_no_jit_)


def apply_adc_noise(
    result, pre_adc_result, output_percentiles, i_s_s, w_s_s, total_scale, params, adc_scale_cache
):
    """Add one ADC's error for one partial-sum tile into `result`.

    Physical noise, quantization, and clipping at the calibrated range are added
    directly, scaled by `total_scale`, so the full-precision tile is never materialized.
    Shared by `layers` (Conv2d/Linear) and `tensors` (attention matmuls).
    """
    adc_scale, (low, high) = get_adc_scale(
        get(params, "adc_bits"),
        output_percentiles,
        get(params, "adc_clip_percentile"),
        get(params, "adc_reference_mode"),
        i_s_s,
        w_s_s,
        adc_scale_cache,
    )

    adc_physical_noise = get(params, "adc_physical_noise")
    if adc_physical_noise is not None:
        error = apply_noise(adc_physical_noise, pre_adc_result, low=low, high=high)
        result += error * abs(total_scale)

    if get(params, "enable_adc_quantization") and adc_scale is not None:
        result = add_quantization_error(
            result_tensor=result, tensor=pre_adc_result, scale=adc_scale, error_scale=total_scale
        )

    adc_clip_percentile = get(params, "adc_clip_percentile")
    if get(params, "enable_adc_clipping") and not (low is None and high is None):
        if is_percentile_below_100(adc_clip_percentile):
            result = add_clipping_error(
                result_tensor=result,
                tensor=pre_adc_result,
                low=low,
                high=high,
                error_scale=total_scale,
            )

    return result
