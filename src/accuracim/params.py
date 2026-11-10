"""The params dict: defaults, predicates, and canonical string and cache keys.

One flat `params` dict carries all hardware settings through the simulator: bit widths,
slicing, clipping percentiles, ADC settings and noise objects.
"""

import hashlib
import os

import numpy as np

from accuracim.config import project_root

DEFAULT_PARAMS = {
    "input_zp": 0,
    "weight_zp": 0,
    "enable_input_quantization": True,
    "enable_input_clipping": True,
    "differential_input": True,
    "enable_weight_quantization": True,
    "enable_weight_clipping": True,
    "differential_weight": True,
    "enable_adc_quantization": True,
    "enable_adc_clipping": True,
    "adc_reference_mode": "layer",
    "input_clip_percentile": 100.0,
    "adc_clip_percentile": 100.0,
    "weight_clip_percentile": 100.0,
    # LLM only
    "weight_operand_loc": "right",
}


def get(params, key):
    """Return `params[key]`, else its default, else None."""
    if key in params:
        return params[key]
    elif key in DEFAULT_PARAMS:
        return DEFAULT_PARAMS[key]
    return None


def close_eq(a, b, epsilon=1e-10):
    return abs(a - b) < epsilon


def is_percentile_below_100(x):
    """True if clipping is active: percentile (or sum of a pair) is not 100."""
    if isinstance(x, int) or isinstance(x, float):
        return not close_eq(x, 100)
    else:
        return not close_eq(sum(x), 100)


# Params that determine the ADC output distribution
ADC_CONFIG_KEYS = {
    "weight_bits",
    "weight_slice_lsb_to_msb",
    "weight_clip_percentile",
    "differential_weight",
    "input_bits",
    "input_slice_lsb_to_msb",
    "input_clip_percentile",
    "differential_input",
    "num_rows",
}


#: Every key a `params_grid_list` entry may set; others are rejected as typos (pipeline)
KNOWN_PARAMS = set(DEFAULT_PARAMS) | ADC_CONFIG_KEYS | {
    "adc_bits", "input_physical_noise", "weight_physical_noise", "adc_physical_noise",
    "weight_drift", "sample_fraction",
}  # fmt: skip


#: Keys a programmed model reads on every forward pass (never when its cells are
#: programmed), plus cache tags the model never reads. Runs differing only in these can
#: share one programmed model (`model_cache`); any other key, including future ones,
#: reprograms it.
FORWARD_PARAMS = frozenset({
    "input_physical_noise", "weight_physical_noise", "adc_physical_noise",
    "enable_input_quantization", "enable_input_clipping", "input_slice_lsb_to_msb",
    "enable_adc_quantization", "enable_adc_clipping", "adc_bits", "adc_clip_percentile",
    "adc_reference_mode", "num_rows", "sample_fraction",
    "quantization", "baseline",  # cache tags (`sources.tag_baseline`, estimate)
})  # fmt: skip


def _exact(value):
    """Return a hashable stand-in for `value` that is equal only for equal values (no rounding)."""
    if value is None or isinstance(value, (bool, str)):
        return (type(value).__name__, value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("number", float(value))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_exact(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _exact(v)) for k, v in value.items())))
    if hasattr(value, "detach") and hasattr(value, "cpu"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return ("array", str(value.dtype), value.shape, digest)
    return (type(value).__qualname__, repr(value))


def programming_key(params: dict) -> tuple:
    """Return an exact key of everything that programs a model: all params but `FORWARD_PARAMS`."""
    keys = (set(params) | set(DEFAULT_PARAMS)) - FORWARD_PARAMS
    # A key set to None reads the same as a missing one (`get`)
    values = {k: get(params, k) for k in keys}
    return tuple(sorted((k, _exact(v)) for k, v in values.items() if v is not None))


def extract_adc_config_params(params: dict):
    """Return the `ADC_CONFIG_KEYS` subset of `params`; None without input/weight bits."""
    extracted_params = {key: get(params, key) for key in ADC_CONFIG_KEYS}
    if extracted_params["input_bits"] is None or extracted_params["weight_bits"] is None:
        return None
    return extracted_params


def get_output_percentiles_per_params(output_percentiles: dict, params: dict):
    """Return the ADC output percentiles for these params' ADC config, or None."""
    extracted_params = extract_adc_config_params(params)
    if not extracted_params:
        return None
    key = params_to_str(**extracted_params)
    if key in output_percentiles:
        return output_percentiles[key]
    return None


def format_number(x: int | float) -> str:
    """Format with just enough decimals: more near 0, and near 100 (percentiles)."""
    if x is None:
        return "None"
    if x == int(x):
        return str(int(x))
    if x < 50:
        magnitude = -int(np.log10(x)) if x != 0 else 0
        precision = magnitude + 2 if x < 1 else 2
        return f"{x:.{precision}f}"
    else:
        diff = 100 - x
        magnitude = -int(np.log10(diff)) if diff != 0 else 0
        precision = magnitude + 3 if diff < 1 else 2
        return f"{x:.{precision}f}"


def encode_slice(slice_list, bits):
    """Slices as "a,b,c"; None -> `bits` (one slice)."""
    if isinstance(slice_list, (list, tuple)):
        return ",".join(str(x) for x in slice_list)
    elif slice_list is None:
        return str(bits)
    return str(slice_list)


def format_percentile(p):
    if isinstance(p, tuple):
        return f"{format_number(p[0])}/{format_number(p[1])}"
    return format_number(p)


def params_to_str(
    weight_bits: int = None,
    input_bits: int = None,
    weight_slice_lsb_to_msb: int | list[int] = None,
    differential_weight: bool = True,
    input_slice_lsb_to_msb: int | list[int] = None,
    differential_input: bool = True,
    num_rows: int = None,
    mode: str = "adc",
    input_clip_percentile: float | int | tuple[float | int] = 100.0,
    weight_clip_percentile: float | int | tuple[float | int] = 100.0,
    adc_clip_percentile: float | int | tuple[float | int] = 100.0,
    adc_reference_mode: str = "layer",
    adc_bits: int = None,
    **kwargs,
):
    """Return the readable params key for `mode`; None if its required bits are unset.

    `mode`: 'adc', 'adc_quantize', 'weight', 'input', 'weight_quantize', 'input_quantize',
    or 'caching' (full cache-directory key).
    """
    if mode == "weight":
        if weight_bits is None:
            return None
        weight_slice_str = encode_slice(weight_slice_lsb_to_msb, weight_bits)
        return f"Wbits-{weight_bits}_Wslice-{weight_slice_str}_Wdiff-{differential_weight}_WeightClip-{format_percentile(weight_clip_percentile)}"

    elif mode == "input":
        if input_bits is None:
            return None
        input_slice_str = encode_slice(input_slice_lsb_to_msb, input_bits)
        return f"Ibits-{input_bits}_Islice-{input_slice_str}_Idiff-{differential_input}_InputClip-{format_percentile(input_clip_percentile)}"

    elif mode == "weight_quantize":
        return f"WeightClip-{format_percentile(weight_clip_percentile)}"

    elif mode == "input_quantize":
        return f"InputClip-{format_percentile(input_clip_percentile)}"

    elif mode == "adc" or mode == "adc_quantize":
        if input_bits is None or weight_bits is None:
            return None

        weight_slice_str = encode_slice(weight_slice_lsb_to_msb, weight_bits)
        input_slice_str = encode_slice(input_slice_lsb_to_msb, input_bits)
        key = f"Wbits-{weight_bits}_Wslice-{weight_slice_str}_WeightClip-{format_percentile(weight_clip_percentile)}_Ibits-{input_bits}_Islice-{input_slice_str}_InputClip-{format_percentile(input_clip_percentile)}_Nrows-{num_rows}_Idiff-{differential_input}_Wdiff-{differential_weight}"

        if mode == "adc_quantize":
            key += f"_AdcClip-{format_percentile(adc_clip_percentile)}_AdcReferenceMode-{adc_reference_mode}"

        return key

    elif mode == "caching":
        weight_slice_str = encode_slice(weight_slice_lsb_to_msb, weight_bits)
        input_slice_str = encode_slice(input_slice_lsb_to_msb, input_bits)
        key = f"Wbits-{weight_bits}_Wslice-{weight_slice_str}_WClip-{format_percentile(weight_clip_percentile)}_Ibits-{input_bits}_Islice-{input_slice_str}_IClip-{format_percentile(input_clip_percentile)}_Nrows-{num_rows}_Idiff-{differential_input}_Wdiff-{differential_weight}"
        key += f"_Abits-{adc_bits}_AClip-{format_percentile(adc_clip_percentile)}_AMode-{adc_reference_mode}"
        # Suffixes only when set, so existing cache directory names are unchanged
        weight_drift = kwargs.get("weight_drift")
        if weight_drift is not None:
            key += f"_Wdrift-{weight_drift}"
        if kwargs.get("quantization") == "noise_source":
            key += "_Qnoise"
        if kwargs.get("baseline") == "quantized":
            key += "_Qbase"
        return key

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'adc', 'weight', 'input', 'adc_quantize', 'weight_quantize', or 'input_quantize'."
        )


def get_cache_dir(params: dict, parent_dir: str = None, build_model: callable = None):
    """Return `<parent_dir>/<readable key>_<hash>` for `params` merged with defaults."""
    if parent_dir is None:
        if build_model is None:
            parent_dir = "cache"
        else:
            model_name = build_model(return_model_name=True)
            parent_dir = os.path.join(project_root(), "cache", model_name)

    # Params + defaults, minus run-only keys
    keys = list(params.keys()) + list(DEFAULT_PARAMS.keys())
    keys.sort()
    exhaustive_dict = {}
    exhaustive_hash_dict = {}
    for k in keys:
        if get(params, k) is not None and k not in {
            "sample_fraction",
            "output_stat",
            "output_percentiles",
        }:
            v = get(params, k)
            exhaustive_dict[k] = v
            # Normalize numbers so equal values hash equally
            if isinstance(v, (float, int)):
                v = format_number(v)
            exhaustive_hash_dict[k] = v

    items_tuple = tuple(sorted(exhaustive_hash_dict.items()))
    # sha256: stable across processes, unlike hash()
    hash_key = hashlib.sha256(str(items_tuple).encode()).hexdigest()[:16]
    readable_key = params_to_str(mode="caching", **exhaustive_dict)
    cache_dir = os.path.join(parent_dir, f"{readable_key}_{hash_key}")

    return cache_dir


def get_ideal_cache_dir(model_cache_dir: str, probe_layers: list[str]) -> str:
    """Return (creating it) `<model_cache_dir>/probe-<l1+l2+...>`, the ideal-output cache dir."""
    probe_key = f"probe-{'+' .join(probe_layers)}"
    path = os.path.join(model_cache_dir, probe_key)
    os.makedirs(path, exist_ok=True)
    return path
