"""Noise sources: separate the output error of each impairment.

`extract_noise_source_params` changes `params` so that only one source is active.
For example, `adc_physical` keeps the ADC noise and disables input/weight quantization and
clipping. Source names match by substring: `adc_physical_Wbits-8_..._GaussianNoise` still
selects the ADC physical keys.
"""

import hashlib

from accuracim.noise import Noise, noise_label
from accuracim.params import params_to_str

#: Params keys holding stationary impairments: applied once at programming, never
#: resampled (see `noise.ConductanceDrift`).
#:
#: Routed into the ideal baseline, not treated as additive sources: the baseline and
#: every isolated source run carry them, so cross-terms (decay rescales weight-/input-
#: referred noise by (1-alpha); ADC-referred noise stays absolute) land exactly in the
#: measured error tensors, with no additivity assumption.
#:
#: To make drift an additive source instead:
#:   1. drop the key here, and have `_quant_params_from_params` null it (it currently
#:      nulls only `*_physical_noise` keys);
#:   2. add its source name to `_SOURCE_ORDER`, so it is calibrated and interpolated.
#: `quantize._apply_weight_drift` is unaffected either way.
STATIONARY_NOISE_KEYS = ("weight_drift",)


def extract_noise_source_params(params: dict, noise_source: str) -> dict:
    """Return `params` reduced to one noise source: its keys only, other mechanisms disabled."""
    slicing_keys = [
        "weight_bits",
        "weight_clip_percentile",
        "weight_slice_lsb_to_msb",
        "differential_weight",
        "input_bits",
        "input_clip_percentile",
        "input_slice_lsb_to_msb",
        "differential_input",
    ]

    # Relevant keys per source (matched by substring)
    noise_source_keys = {
        "input_quantize": ["input_bits", "input_clip_percentile", "differential_input"],
        "input_physical": [
            "input_bits",
            "input_clip_percentile",
            "differential_input",
            "input_physical_noise",
            "input_slice_lsb_to_msb",
        ],
        "input_clipping": ["input_clip_percentile", "differential_input"],
        "weight_quantize": ["weight_bits", "weight_clip_percentile"],
        "weight_physical": [
            "weight_bits",
            "weight_clip_percentile",
            "weight_physical_noise",
            "weight_slice_lsb_to_msb",
            "differential_weight",
        ],
        "weight_clipping": ["weight_clip_percentile"],
        "adc_physical": [
            "adc_physical_noise",
            "adc_bits",
            "num_rows",
            "adc_clip_percentile",
            "adc_reference_mode",
        ]
        + slicing_keys,
        "adc_quantize": ["adc_bits", "num_rows", "adc_clip_percentile", "adc_reference_mode"]
        + slicing_keys,
        "adc_clipping": [
            "num_rows",
            "adc_clip_percentile",
            "adc_reference_mode",
            "weight_bits",
            "weight_clip_percentile",
            "weight_slice_lsb_to_msb",
            "differential_weight",
            "input_bits",
            "input_clip_percentile",
            "input_slice_lsb_to_msb",
            "differential_input",
        ],
        # Input/ADC quantization only, to calibrate how it suppresses noise
        "selective_quantize": [
            "num_rows",
            "adc_clip_percentile",
            "adc_reference_mode",
            "weight_bits",
            "weight_clip_percentile",
            "weight_slice_lsb_to_msb",
            "differential_weight",
            "input_bits",
            "input_clip_percentile",
            "input_slice_lsb_to_msb",
            "differential_input",
        ],
    }

    keys = []
    for k, v in noise_source_keys.items():
        if k in noise_source:
            keys += v
    keys = list(set(keys))

    extracted_params = {k: params[k] for k in keys if k in params}

    # Stationary impairments are in the baseline, so every isolated run carries
    # them too (else the decay would count as this source's noise)
    for key in STATIONARY_NOISE_KEYS:
        if key in params:
            extracted_params[key] = params[key]

    # 'full': quantization and clipping together
    full_quantization = "full" in noise_source
    is_quantize = ("quantize" in noise_source) or full_quantization
    is_clipping = ("clipping" in noise_source) or full_quantization

    if "input" in noise_source:
        extracted_params["enable_input_quantization"] = is_quantize
        extracted_params["enable_input_clipping"] = is_clipping
        extracted_params["enable_weight_quantization"] = False
        extracted_params["enable_weight_clipping"] = False

    if "weight" in noise_source:
        extracted_params["enable_weight_quantization"] = is_quantize
        extracted_params["enable_weight_clipping"] = is_clipping
        extracted_params["enable_input_quantization"] = False
        extracted_params["enable_input_clipping"] = False

    # ADC sources: input/weight quantization and clipping off
    elif "adc" in noise_source:
        extracted_params["enable_weight_quantization"] = False
        extracted_params["enable_input_quantization"] = False
        extracted_params["enable_input_clipping"] = False
        extracted_params["enable_weight_clipping"] = False

        extracted_params["enable_adc_quantization"] = is_quantize
        extracted_params["enable_adc_clipping"] = is_clipping

    if "selective_quantize" in noise_source:
        extracted_params["enable_input_quantization"] = True
        extracted_params["enable_adc_quantization"] = True
        extracted_params["enable_weight_quantization"] = False
        extracted_params["enable_input_clipping"] = False
        extracted_params["enable_weight_clipping"] = False
        extracted_params["enable_adc_clipping"] = False

    # Several noises: pick the one whose class name is in the source
    if "physical" in noise_source:
        noise = extracted_params.get(f"{noise_source}_noise", None)
        if noise is not None and isinstance(noise, tuple):
            for noise_obj in noise:
                if noise_obj.__class__.__name__ in noise_source:
                    extracted_params[f"{noise_source}_noise"] = noise_obj
                    break

    return extracted_params


def _quant_params_from_params(params: dict) -> dict:
    """Return noiseless-baseline params: `*_physical_noise` off, stationary impairments kept (see `STATIONARY_NOISE_KEYS`)."""
    quant_params = params.copy()
    for key in list(quant_params.keys()):
        if key.endswith("_physical_noise") and key not in STATIONARY_NOISE_KEYS:
            quant_params[key] = None
    return quant_params


def _names_profile(noise_source: str, label: str) -> bool:
    """True if `noise_source` names the profile `label` as a whole `_`-separated part.

    (`GaussianNoise` must not match `GaussianNoise(pattern=...)`; a suffix such as
    `_selective_quantize` may follow.)
    """
    start = noise_source.find("_" + label)
    while start != -1:
        end = start + 1 + len(label)
        if end == len(noise_source) or noise_source[end] == "_":
            return True
        start = noise_source.find("_" + label, start + 1)
    return False


def _select_physical_noise(params: dict, stage: str, noise_source: str):
    """Return this stage's noise object named in `noise_source` (default: the first)."""
    noise = params.get(f"{stage}_physical_noise", None)
    if noise is None:
        return None
    noise_list = [noise] if isinstance(noise, Noise) else noise
    for noise_obj in noise_list:
        if _names_profile(noise_source, noise_label(noise_obj)):
            return noise_obj
    return noise_list[0] if noise_list else None


def _physical_source_params(params: dict, noise_source: str) -> dict:
    """Isolate one physical source at the full hardware operating point.

    Bit widths, slicing, and clipping stay; other physical noises are off, so the run
    differs from its baseline by exactly this source.
    """
    isolated = _quant_params_from_params(params)
    for stage in ("input", "weight", "adc"):
        isolated[f"{stage}_physical_noise"] = None
    for stage in ("input", "weight", "adc"):
        if f"{stage}_physical" in noise_source:
            isolated[f"{stage}_physical_noise"] = _select_physical_noise(
                params, stage, noise_source
            )
            break
    return isolated


def source_params(params: dict, noise_source: str, quantization: str = "precise") -> dict:
    """Return params isolating one noise source under the given accounting mode.

    `noise_source`: full-precision baseline, so everything else (quantization included)
    is off (`extract_noise_source_params`).
    `precise`: quantized baseline, so physical sources are isolated at that operating
    point. Measuring on full precision and replaying onto quantized drops their
    interaction (the estimator under-predicts).
    Quantize/clipping sources use `extract_noise_source_params` in both modes.
    """
    if quantization == "precise" and "physical" in noise_source:
        return _physical_source_params(params, noise_source)
    return extract_noise_source_params(params, noise_source)


def _noise_seed_for_source(noise_source: str) -> int:
    """Deterministic noise seed from the source name."""
    digest = hashlib.sha256(noise_source.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# =====================================================================
#            How quantization error is accounted for
# =====================================================================
#
# The modes differ in what noise sources are measured against (exactness vs. speed):
#
#   precise       Quantized-but-noiseless baseline: quantization error is absorbed
#                 into the reference. Exact, but each quantization setting needs
#                 its own ideal run and anchors.
#
#   noise_source  Full-precision baseline: quantization is an additive source with
#                 an analytic strength. One baseline for the whole sweep, assuming
#                 quantization error adds independently like physical noise.
#
# Below is mostly `noise_source` support; `precise` (default) paths are unchanged.

QUANTIZATION_MODES = ("precise", "noise_source")

#: Sources per mode: `precise` has physical only; `noise_source` adds a quantize source per stage.
_SOURCE_ORDER = {
    "precise": ("weight_physical", "input_physical", "adc_physical"),
    "noise_source": (
        "weight_quantize",
        "weight_physical",
        "input_quantize",
        "input_physical",
        "adc_quantize",
        "adc_physical",
    ),
}


def quantize_source_strength(bits: int) -> float:
    """Analytic strength of a `bits`-bit uniform quantizer (cf. `Noise.get_scaling_factor`).

    Variance falls as 1/4^bits. Only ratios within a source matter, so 2**16 is an arbitrary shared scale.
    """
    return float(2**16) / (1 << (bits * 2))


def baseline_params(params: dict, quantization: str) -> dict | None:
    """Return the ideal-baseline params, or None for full precision."""
    if quantization == "noise_source":
        return None
    return _quant_params_from_params(params)


def tag_baseline(params: dict, quantization: str) -> dict:
    """Add the accounting mode to `params` so it reaches the cache key (`params.get_cache_dir`).

    Errors measured against full precision must not replay against a quantized baseline.
    `precise` returns `params` unchanged, keeping existing cache names.
    """
    if quantization == "precise":
        return params
    return {**params, "quantization": quantization}


def enumerate_noise_sources(
    params_list: list[dict], precise_mode_noise_sources: list | set, quantization: str = "precise"
):
    """Return the sweep's noise sources and each one's analytic strength.

    Returns:
    tuple: (noise_sources, strengths, precise_mode_noise_sources). `strengths[i][source]`
    is the x-axis value for `params_list[i]`; the returned precise set is what to measure exactly.

    `noise_source` mode, per quantize source:
    - in `precise_mode_noise_sources`: promoted to `<ns>_full`, measured exactly;
    - else: strength from `quantize_source_strength`, and `<stage>_clipping` is measured
      exactly (clipping is a tail effect the 1/4^bits model misses).
    """
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(
            f"Unknown quantization mode: {quantization}. Expected one of {QUANTIZATION_MODES}."
        )

    precise = set(precise_mode_noise_sources)
    if quantization == "precise":
        # Quantization is in the baseline: quantize entries have nothing to calibrate
        precise = {ns for ns in precise if "physical" in ns}
    precise_out = precise.copy()

    noise_sources = set()
    strengths = []

    for params in params_list:
        stat = {}

        for ns in _SOURCE_ORDER[quantization]:
            mode = ns.split("_")[0]  # 'adc', 'weight', 'input'

            if "quantize" in ns:
                if ns in precise:
                    ns_full = f"{ns}_full"
                    noise_sources.add(ns_full)
                    precise_out.add(ns_full)
                    precise_out.discard(ns)  # avoid calibrating it twice
                    continue

                noise_sources.add(f"{mode}_clipping")
                precise_out.add(f"{mode}_clipping")

                extracted = extract_noise_source_params(params, ns)
                bits = extracted.get(f"{mode}_bits", 0)
                if bits:
                    key = params_to_str(mode=f"{mode}_quantize", **extracted)
                    ns_q = f"{ns}_{key}" if key is not None else ns
                    noise_sources.add(ns_q)
                    stat[ns_q] = quantize_source_strength(bits)
                continue

            noise = params.get(f"{mode}_physical_noise", None)
            key = params_to_str(mode=mode, **params)
            if noise is not None and key is not None:
                base = f"{ns}_{key}"
                noise_list = [noise] if isinstance(noise, Noise) else noise
                labels = [noise_label(n) for n in noise_list]
                if len(set(labels)) != len(labels):
                    raise ValueError(
                        f"{mode}_physical_noise lists the same profile twice ({labels}); "
                        "combine them into one, or give them different shapes"
                    )
                for n, label in zip(noise_list, labels):
                    # One source per class and shape: interpolated along its scaling factor
                    ns_noise = f"{base}_{label}"
                    noise_sources.add(ns_noise)
                    stat[ns_noise] = n.get_scaling_factor()

        strengths.append(stat)

    return list(noise_sources), strengths, precise_out


def suppression_factor(
    suppressed_noise: float, measured_noise: float, noise_source: str, quantization: str
) -> float:
    """Return how much the quantizer clamps noise layered on it, in (0, 1].

    `precise`: quantization is in the baseline, so always 1.0.
    `noise_source`: `suppressed_noise` (from `full - quant`) is this source's noise as
    clamped by the quantizer; `measured_noise` is the standalone noise the cached error
    holds. sqrt of their ratio converts one to the other; capped at 1 (clamping never
    amplifies). Also folded into `estimate.calculate_piecewise_weights`.
    """
    if quantization != "noise_source":
        return 1.0
    if suppressed_noise <= 0.0 or measured_noise <= 0.0:
        return 1.0
    return min((suppressed_noise / measured_noise) ** 0.5, 1.0)
