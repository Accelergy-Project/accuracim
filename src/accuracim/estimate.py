"""Estimate: calibrate a few anchor points, and interpolate the rest.

Goal: do not run the full simulation for each hardware configuration.
1 | For each noise source:
    1-1 | Get the analytical noise magnitude (scaling_factor).
    1-2 | Run small calibration simulations to find the anchor points (0.5% - 5% accuracy loss).
    1-3 | Run full simulations at the anchors, and collect the error tensors (y_noisy - y_ideal).
    1-4 | Fit the analytical (scaling_factor) vs. measured noise magnitude in log-log.
2 | Estimate y_noisy = y_ideal + sum over noise sources of (error_tensor * noise_scaling).
+ Sources in `precise_mode_noise_sources` measure their error tensors directly (no interpolation).
"""

import datetime
import logging
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from accuracim import model_cache
from accuracim.anchors import search_anchors
from accuracim.config import project_root, results_path
from accuracim.device import get_device
from accuracim.evaluation import (
    average_performance_metrics,
    get_batch_size,
    predicted_class,
    predictions_per_sample,
    resolve_eval_mode,
    update_flip_metrics,
    update_performance_metrics,
)
from accuracim.log import (
    fmt_duration,
    fmt_metric,
    fmt_noise,
    indented,
    log_results,
    section,
    short_path,
    source_label,
)
from accuracim.params import get, get_cache_dir, get_ideal_cache_dir, is_percentile_below_100
from accuracim.simulate import _ideal_quant_cache_dir, run_noise_simulation
from accuracim.sources import (
    _noise_seed_for_source,
    _quant_params_from_params,
    baseline_params,
    enumerate_noise_sources,
    source_params,
    suppression_factor,
    tag_baseline,
)

logger = logging.getLogger(__name__)


def calculate_piecewise_weights(
    calibrated_noise, selected_noises_for_ns, quantize_suppress_factors_ns
):
    """Inverse-log-distance weights over the two anchors bracketing `calibrated_noise`.

    Parameters:
    calibrated_noise (float): Target noise.
    selected_noises_for_ns (np.ndarray): Anchor noises (any order).
    quantize_suppress_factors_ns (array-like): Per-anchor suppression factors, multiplied in.

    Returns:
    np.ndarray: Weights in input order. Only the bracketing anchors are non-zero
    (one if out of range); they sum to 1 before suppression.
    """
    # searchsorted needs sorted input
    sort_indices = np.argsort(selected_noises_for_ns)
    sorted_noises = selected_noises_for_ns[sort_indices]
    sorted_suppress_factors = np.asarray(quantize_suppress_factors_ns)[sort_indices]

    inv_distances = np.zeros_like(sorted_noises)

    # Upper bracket index
    idx = np.searchsorted(sorted_noises, calibrated_noise)

    # Below range: clamp to first anchor
    if idx == 0:
        inv_distances[0] = 1.0

    # At/above range: clamp to last anchor
    elif idx == len(sorted_noises):
        inv_distances[-1] = 1.0

    # In range: interpolate between the two brackets
    else:
        bracket_indices = [idx - 1, idx]
        bracket_noises = sorted_noises[bracket_indices]

        distances = abs(np.log(calibrated_noise / (bracket_noises + 1e-12)))
        weights = 1.0 / distances

        weights /= np.sum(weights)

        inv_distances[bracket_indices] = weights

    # Apply suppression, restore input order
    inv_distances = inv_distances * sorted_suppress_factors
    return inv_distances[np.argsort(sort_indices)]


def precise_mode_calibrate_noise(
    dataloader: torch.utils.data.DataLoader,
    build_model: callable,
    stat: dict,
    output_percentiles: dict,
    params_list: list[dict],
    noise_source: str,
    measured_noises: dict,
    params_dict: dict,
    N=500,
    quantization: str = "precise",
    cache_dir: str = "cache",
    eval_mode: str | None = None,
):
    """Measure exact noise and error tensors for every params entry of `noise_source`.

    Appends to `measured_noises[noise_source]` and `params_dict[noise_source]` in place.
    """
    result_cash = {}
    noise_layers = model_cache.leaf_module_names(build_model)

    for idx, params in enumerate(params_list):
        extracted_params = tag_baseline(
            source_params(params, noise_source, quantization), quantization
        )
        cached_params = extracted_params.copy()
        params_dict[noise_source].append(extracted_params)

        noise_type = noise_source.split("_")[0]  # 'input', 'weight', 'adc'

        if "clipping" in noise_source:
            # Clipping does not depend on bit width -- keep it out of the cache key.
            cached_params.pop(f"{noise_type}_bits", None)

        # No noise: 100% clipping and no quantization
        if "clipping" in noise_source or "quantize" in noise_source:
            clips = is_percentile_below_100(get(extracted_params, f"{noise_type}_clip_percentile"))
            quantizes = (
                "quantize" in noise_source
                and get(extracted_params, f"{noise_type}_bits") is not None
            )
            if not (clips or quantizes):
                measured_noises[noise_source].append(0.0)
                continue

        # Reuse the result for identical params
        params_key = tuple(sorted(cached_params.items()))
        if params_key in result_cash:
            cashed_idx = result_cash[params_key]
            measured_noises[noise_source].append(measured_noises[noise_source][cashed_idx])
            continue

        ideal_params = baseline_params(extracted_params, quantization)
        logger.info(f"exact run · idx {idx} · N {N}")
        with indented():
            outputs_ideal_paths, noises, measured_performance = run_noise_simulation(
                dataloader,
                build_model,
                stat,
                output_percentiles,
                noise_layers,
                ["__output__"],
                [extracted_params],
                N=N,
                save_result=False,
                save_error=True,
                noise_seed=_noise_seed_for_source(noise_source),
                ideal_params=ideal_params,
                cache_dir=cache_dir,
                eval_mode=eval_mode,
            )

        measured_noise = noises["__output__"][0]
        measured_noises[noise_source].append(measured_noise)

        result_cash[params_key] = idx

    logger.debug(f"Precise calibration done for {noise_source}")


def _build_loglog_interpolator(x_points, y_points, max_noise=None):
    """Return f(t): piecewise-linear log-log interpolation with linear extrapolation.

    One anchor: f(t) = (y0 / x0) * t. f(t) = 0 for t <= 0, inf above `max_noise`.
    """
    x = np.asarray(x_points, dtype=np.float64)
    y = np.asarray(y_points, dtype=np.float64)
    eps = 1e-14

    # log(0) would dominate the fit; an anchor that measured nothing is no anchor.
    keep = y > 0
    if keep.any():
        x, y = x[keep], y[keep]

    def wrap_with_max_noise(f):
        def wrapped_function(x):
            if max_noise is None:
                return f(x)
            x = np.asarray(x, dtype=np.float64)
            result = f(x)
            result[x > max_noise] = np.inf
            return result

        return wrapped_function

    if x.size == 1:
        scale = y[0] / (x[0] + eps)

        def f_single(t):
            t_arr = np.asarray(t, dtype=np.float64)
            out = np.zeros_like(t_arr, dtype=np.float64)
            pos = t_arr > 0
            out[pos] = scale * t_arr[pos]
            return out

        return wrap_with_max_noise(f_single)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    logx = np.log(x + eps)
    logy = np.log(y + eps)

    f_ll = interp1d(logx, logy, kind="linear", fill_value="extrapolate", assume_sorted=True)

    def f_scipy(t):
        t_arr = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t_arr, dtype=np.float64)
        pos = t_arr > 0
        if np.any(pos):
            logt = np.log(t_arr[pos] + eps)
            out[pos] = np.exp(f_ll(logt))
        return out

    return wrap_with_max_noise(f_scipy)


def get_valid_indices(curr_noise_perf, curr_noise, best_perf, curr_idx, valid_indices):
    """Update the min/max anchors in `valid_indices` with the current point if it qualifies.

    min anchor: perf >= 0.995 * best; max anchor: perf >= 0.95 * best (all non-flip metrics).
    """
    valid_min_keys = ("valid_min_idx", "valid_min_perf", "valid_min_noise")
    valid_max_keys = ("valid_max_idx", "valid_max_perf", "valid_max_noise")
    valid_min_idx, valid_min_perf, valid_min_noise = [valid_indices[k] for k in valid_min_keys]
    valid_max_idx, valid_max_perf, valid_max_noise = [valid_indices[k] for k in valid_max_keys]
    min_threshold = 0.995
    max_threshold = 0.95

    is_min_valid, is_max_valid = True, True
    for category in best_perf.keys():
        # Flip metrics not used for anchor selection (for now)
        if "flip" in category:
            continue

        best_acc = best_perf[category][0]
        max_noise_acc = valid_max_perf[category][0]
        min_noise_acc = valid_min_perf[category][0]
        curr_noise_acc = curr_noise_perf[category][0]

        logger.info(
            f"scan · idx {curr_idx} · {category} {fmt_metric(curr_noise_acc)} "
            f"(lowest noise {fmt_metric(best_acc)}) · loss {(1-curr_noise_acc / best_acc)*100:.2f}%"
        )
        logger.debug(
            f"anchor {category}: min {fmt_metric(min_noise_acc)} · max {fmt_metric(max_noise_acc)}"
        )
        if curr_noise_acc < min_threshold * best_acc:
            is_min_valid = False

        if curr_noise_acc < max_threshold * best_acc:
            is_max_valid = False

        if not (is_min_valid or is_max_valid):
            break

    if is_min_valid:
        for k, v in zip(valid_min_keys, (curr_idx, curr_noise_perf, curr_noise)):
            valid_indices[k] = v
    if is_max_valid:
        for k, v in zip(valid_max_keys, (curr_idx, curr_noise_perf, curr_noise)):
            valid_indices[k] = v

    return valid_indices


RUN_SIM_CACHE_ = {}


def calibrate_noise(
    dataloader: torch.utils.data.DataLoader,
    build_model: callable,
    stat: dict,
    output_percentiles: dict,
    params_list: list[dict],
    noise_source: str,
    estimated_noises,
    measured_noises: dict,
    params_dict: dict,
    num_points=2,
    N=500,
    N_calibration=128,
    quantization: str = "precise",
    cache_dir: str = "cache",
    eval_mode: str | None = None,
):
    """Fit measured vs. estimated noise for one source from a few anchor simulations.

    1. Calibration sims (`N_calibration`) to find the 0.5% / 5% loss anchors: adaptive search
       (`anchors.search_anchors`, at most `num_points` sims), or a scan for quantize sources.
    2. Full sims (`N`) on the anchors to cache their error tensors.
    3. Log-log interpolation over the anchors.

    Returns:
    tuple: (f, final_indices, outputs_ideal_paths, quantize_suppress_factors),
    or None if every estimated noise is 0.
    """

    positive_mask = estimated_noises > 0
    if not positive_mask.any():
        return None

    positive_indices = np.where(positive_mask)[0]
    min_idx = positive_indices[np.argmin(estimated_noises[positive_indices])]

    noise_layers = model_cache.leaf_module_names(build_model)

    def run_sim_for_idx(
        idx,
        noise_source=noise_source,
        N=N,
        append_to_params_dict=True,
        save_error=True,
        return_performance=False,
        quantized_baseline=False,
    ):
        """Simulate `params_list[idx]`; return (ideal_paths, noise, perf). Cached in `RUN_SIM_CACHE_`."""
        params = params_list[idx]
        extracted_params = tag_baseline(
            source_params(params, noise_source, quantization), quantization
        )
        if quantized_baseline:
            # Scored against the quantization-only model, not full precision --
            # tag it so it cannot reuse the FP-baseline cache for these params.
            extracted_params = {**extracted_params, "baseline": "quantized"}
        # Record params so their cached error tensors can be found later
        if append_to_params_dict:
            params_dict[noise_source].append(extracted_params)

        key_items = tuple(sorted((k, v) for k, v in extracted_params.items()))
        cache_key = hash((key_items, int(N), bool(save_error), bool(return_performance)))

        if cache_key in RUN_SIM_CACHE_:
            logger.debug(f"Reusing run for idx {idx} (N {N})")
            return RUN_SIM_CACHE_[cache_key]

        kind = "calibration run" if return_performance else "full run"
        if quantized_baseline:
            kind += " vs. quantized baseline"
        logger.info(f"{kind} · idx {idx} · N {N}")

        ideal_params = (
            _quant_params_from_params(extracted_params)
            if quantized_baseline
            else baseline_params(extracted_params, quantization)
        )
        with indented():
            outputs_ideal_paths, noises, noise_perf = run_noise_simulation(
                dataloader,
                build_model,
                stat,
                output_percentiles,
                noise_layers,
                ["__output__"],
                [extracted_params],
                N=N,
                save_result=False,
                save_error=save_error,
                return_performance=return_performance,
                noise_seed=_noise_seed_for_source(noise_source),
                ideal_params=ideal_params,
                cache_dir=cache_dir,
                eval_mode=eval_mode,
            )
        noise = noises["__output__"][0]
        output = (outputs_ideal_paths, noise, noise_perf)

        RUN_SIM_CACHE_[cache_key] = output
        return output

    def run_sim_for_idx_calibrated(idx, N_calibration=N_calibration):
        """Calibration run: `N_calibration` samples, performance only, not recorded in `params_dict`."""
        calib = dict(
            N=N_calibration, append_to_params_dict=False, save_error=False, return_performance=True
        )
        if quantization != "noise_source" or "physical" not in noise_source:
            return run_sim_for_idx(idx, **calib)

        # The noise's effect at the quantized operating point, measured directly
        # against a quantization-only baseline.
        out = run_sim_for_idx(
            idx, noise_source=f"{noise_source}_selective_quantize", quantized_baseline=True, **calib
        )
        logger.debug(f"idx {idx}: noise under quantization {fmt_noise(out[1])}")
        return out

    # Anchors: x = estimated noise, y = measured noise.
    # quantize_suppress_factors: unsuppressed -> quantization-suppressed noise level.
    calibration_indices, x_points, y_points = [], [], []
    final_indices, quantize_suppress_factors = [], []

    # Reference performance at the lowest noise
    outputs_ideal_paths, measured_noise_min_calib, min_noise_perf = run_sim_for_idx_calibrated(
        min_idx
    )

    valid_indices = np.where(estimated_noises > 0)[0]

    # Pick up to `num_points` points, evenly spaced over unique estimated noises (min_idx excluded)
    sorted_cand = valid_indices[np.argsort(estimated_noises[valid_indices])]

    unique_sorted = []
    seen_values = set([estimated_noises[min_idx]])
    for idx in sorted_cand:
        val = estimated_noises[idx]
        if val not in seen_values:
            unique_sorted.append(idx)
            seen_values.add(val)

    select_cnt = min(num_points, len(unique_sorted))
    if select_cnt == len(unique_sorted):
        selected_indices = unique_sorted
    else:
        pick_pos = np.linspace(0, len(unique_sorted) - 1, num=select_cnt, dtype=int)
        selected_indices = [unique_sorted[p] for p in pick_pos]

    valid_indices = {
        "valid_min_idx": min_idx,
        "valid_min_perf": min_noise_perf,
        "valid_min_noise": 0,
        "valid_max_idx": min_idx,
        "valid_max_perf": min_noise_perf,
        "valid_max_noise": 0,
    }

    # Non-quantize sources: adaptive search for the anchors, no full scan
    if "quantize" not in noise_source:

        def evaluate(idx):
            _, curr_noise, curr_noise_perf = run_sim_for_idx_calibrated(idx)
            return curr_noise, curr_noise_perf

        valid_indices = search_anchors(
            unique_sorted, estimated_noises, evaluate, min_idx, min_noise_perf, budget=num_points
        )
        selected_indices = []

    # Quantize sources: every selected point is also an interpolation point
    if selected_indices:
        logger.info(f"calibration points · idx {', '.join(str(i) for i in selected_indices)}")
    for idx in selected_indices:
        if idx < 0 or idx >= len(params_list):
            continue
        _, curr_noise, curr_noise_perf = run_sim_for_idx_calibrated(idx)

        if "quantize" in noise_source:
            calibration_indices.append(idx)
            x_points.append(estimated_noises[idx])
            y_points.append(curr_noise)

        # Track the 0.5% / 5% loss anchors
        valid_indices = get_valid_indices(
            curr_noise_perf, curr_noise, min_noise_perf, idx, valid_indices
        )

    # Non-quantize sources: the 0.5% / 5% loss points are the anchors
    valid_min_idx, valid_min_perf, valid_min_noise = [
        valid_indices[k] for k in ("valid_min_idx", "valid_min_perf", "valid_min_noise")
    ]
    valid_max_idx, valid_max_perf, valid_max_noise = [
        valid_indices[k] for k in ("valid_max_idx", "valid_max_perf", "valid_max_noise")
    ]

    # Quantize source past 0.5% loss at all scanned points: the lowest one is the low anchor
    if "quantize" in noise_source and valid_min_idx == min_idx and calibration_indices:
        valid_min_idx = min(calibration_indices, key=lambda i: estimated_noises[i])
        if valid_max_idx == min_idx:
            valid_max_idx = valid_min_idx

    mid_idx = None
    if "quantize" not in noise_source:
        for idx, noise in zip([valid_min_idx, valid_max_idx], [valid_min_noise, valid_max_noise]):
            if idx not in calibration_indices:
                calibration_indices.append(idx)
                x_points.append(estimated_noises[idx])
                y_points.append(noise)

        low, high = estimated_noises[valid_min_idx], estimated_noises[valid_max_idx]
        between = [i for i in unique_sorted if low < estimated_noises[i] < high]
        if len(unique_sorted) + 1 >= MID_ANCHOR_MIN_POINTS and between:
            target = 0.5 * np.log(low * high)
            mid_idx = min(between, key=lambda i: abs(np.log(estimated_noises[i]) - target))
            _, mid_noise, _ = run_sim_for_idx_calibrated(mid_idx)
            calibration_indices.insert(1, mid_idx)
            x_points.insert(1, estimated_noises[mid_idx])
            y_points.insert(1, mid_noise)

    def run_full_sim_for_idx(idx):
        """Run a full-N sim on `idx` (caches its error tensor) and record its suppression factor.

        `precise`: same quantity as calibration with more samples, so it replaces the anchor y
        (the cached tensor replays at factor 1).
        `noise_source`: different quantities (clamped vs. standalone); their ratio is the
        conversion, so the calibration y stays. See `sources.suppression_factor`.
        """
        outputs_ideal_paths, measured_noise, _ = run_sim_for_idx(idx)
        for k, (calibration_idx, suppressed_noise) in enumerate(zip(calibration_indices, y_points)):
            if estimated_noises[calibration_idx] == estimated_noises[idx]:
                quantize_suppress_factors.append(
                    suppression_factor(suppressed_noise, measured_noise, noise_source, quantization)
                )
                if quantization != "noise_source":
                    y_points[k] = measured_noise
                break
        final_indices.append(idx)
        return outputs_ideal_paths, measured_noise

    # Full sims on the min/max anchors
    outputs_ideal_paths, _ = run_full_sim_for_idx(valid_min_idx)
    if (
        estimated_noises[valid_max_idx] != estimated_noises[valid_min_idx]
        and len(unique_sorted) > 2
        and num_points >= 2
    ):
        if mid_idx is not None:
            run_full_sim_for_idx(mid_idx)
        run_full_sim_for_idx(valid_max_idx)

    max_noise = None
    f = _build_loglog_interpolator(x_points, y_points, max_noise)

    # Report only full-N anchors (updates `measured_noises` in place)
    final_y_points = []
    final_noises_set = set([estimated_noises[idx] for idx in final_indices])
    for y, idx in zip(y_points, calibration_indices):
        val = estimated_noises[idx]
        if val in final_noises_set:
            final_y_points.append(y)
    assert len(final_y_points) == len(final_indices)
    measured_noises[noise_source].extend(final_y_points)

    return f, final_indices, outputs_ideal_paths, quantize_suppress_factors


#: Calibration runs: enough scored predictions to place the anchors, at least
#: `CALIBRATION_MIN_SAMPLES` samples (diversity), at most N // 5 of the full run
CALIBRATION_PREDICTIONS = 1500
CALIBRATION_MIN_SAMPLES = 128

#: A physical source with at least this many noise points gets a middle anchor:
#: the point closest to sqrt(low x high) (one more calibration run and full run)
MID_ANCHOR_MIN_POINTS = 8


def calibration_size(dataloader: torch.utils.data.DataLoader, eval_mode: str, N: int) -> int:
    """Return the calibration sample count, sized by predictions per sample (see `predictions_per_sample`).

    ImageNet (1 per sample) -> 1500 samples; WikiText-2 (127 tokens per sample) -> 128 samples.
    """
    per_sample = predictions_per_sample(next(iter(dataloader)), eval_mode)
    wanted = math.ceil(CALIBRATION_PREDICTIONS / per_sample)
    cap = max(CALIBRATION_MIN_SAMPLES, N // 5)
    N_calibration = int(min(N, max(CALIBRATION_MIN_SAMPLES, min(wanted, cap))))
    logger.info(
        f"calibration runs · N {N_calibration} "
        f"({per_sample:.4g} predictions/sample, target {CALIBRATION_PREDICTIONS})"
    )
    return N_calibration


#: Estimated loss above which a result is flagged as beyond the small-noise range
#: (`beyond_small_noise` column): the Scaling and Additivity laws may not hold there.
SMALL_NOISE_MAX_LOSS = 0.05

_LOWER_IS_BETTER = ("cross_entropy", "perplexity")


def _relative_losses(estimated: dict, ideal: dict) -> dict[str, list[float]]:
    """Return {metric: per-params relative loss vs. the ideal baseline} (NaN without a baseline).

    Accuracy-like metrics: 1 - estimated/ideal. Cross-entropy / perplexity: estimated/ideal - 1.
    """
    losses = {}
    for metric, ideal_values in ideal.items():
        if "flip" in metric or metric not in estimated:
            continue
        values = []
        for est, base in zip(estimated[metric], ideal_values):
            if not base:
                values.append(float("nan"))
            elif metric in _LOWER_IS_BETTER:
                values.append(est / base - 1)
            else:
                values.append(1 - est / base)
        losses[metric] = values
    return losses


def run_noise_estimation(
    dataloader: torch.utils.data.DataLoader,
    build_model: callable,
    stat: dict,
    output_percentiles: dict,
    params_list: list[dict],
    precise_mode_noise_sources: list | set = {},
    N=500,
    max_calibration_noise=None,
    num_interpolation_points=2,
    save_result=True,
    eval_mode: str | None = None,
    metric_fn: callable = None,
    run_name: str | None = None,
    quantization: str = "precise",
    cache_dir: str | None = None,
    N_calibration: int | None = None,
):
    """Estimate performance for every params entry without simulating each one.

    1. Calibrate each noise source (or measure it, if in `precise_mode_noise_sources`).
    2. Per params: add scaled cached error tensors to cached ideal outputs, then score.

    `cache_dir` (default `<project root>/cache`) must be the one the anchor runs write to.
    `N_calibration` (default: `calibration_size`) is the sample count of anchor-search runs.

    Returns:
    None. Results are printed, and saved to `results/<model>/estimated/` if `save_result`.
    """
    if cache_dir is None:
        cache_dir = os.path.join(project_root(), "cache")

    # Which sources exist, and how strong each is analytically -- the one place
    # the two quantization modes differ structurally. See accuracim.sources.
    noise_sources, interpolated_noises, precise_mode_noise_sources = enumerate_noise_sources(
        params_list, precise_mode_noise_sources, quantization
    )

    if metric_fn is None:
        metric_fn = update_performance_metrics
    if eval_mode is None:
        eval_mode = build_model(return_mode=True)
    eval_mode = resolve_eval_mode(eval_mode)

    precise_noise_time_s = 0.0
    accelerated_noise_time_s = 0.0

    calibrated_noises = {ns: [] for ns in noise_sources}

    outputs_ideal_paths = None
    precise_modes = {ns: False for ns in noise_sources}
    selected_noises, measured_noises, quantize_suppress_factors, params_dict = [
        {ns: [] for ns in noise_sources} for i in range(4)
    ]

    for source_idx, noise_source in enumerate(noise_sources):
        section(
            f"Estimate · noise source {source_idx + 1}/{len(noise_sources)}: "
            f"{source_label(noise_source)}"
        )
        logger.debug(f"Noise source: {noise_source}")
        source_start = time.monotonic()
        # Precise mode: measure directly
        for precise_noise_source in precise_mode_noise_sources:
            if precise_noise_source in noise_source:
                precise_start = datetime.datetime.now()
                precise_mode_calibrate_noise(
                    dataloader,
                    build_model,
                    stat,
                    output_percentiles,
                    params_list,
                    noise_source,
                    measured_noises,
                    params_dict,
                    N=N,
                    quantization=quantization,
                    cache_dir=cache_dir,
                    eval_mode=eval_mode,
                )
                precise_noise_time_s += (datetime.datetime.now() - precise_start).total_seconds()
                calibrated_noises[noise_source] = measured_noises[noise_source]
                precise_modes[noise_source] = True

        if precise_modes[noise_source]:
            logger.info(
                f"done (measured exactly) · {fmt_duration(time.monotonic() - source_start)}"
            )
            continue

        accel_start = datetime.datetime.now()
        estimated_noises = np.array(
            [noise_stat.get(noise_source, 0.0) for noise_stat in interpolated_noises]
        )

        if quantization == "noise_source" and "quantize" not in noise_source:
            num_interpolation_points_ = max(len(estimated_noises) / 2, 6)
        else:
            # Scaling works poorly for quantization -- calibrate on every point.
            num_interpolation_points_ = len(estimated_noises)

        if N_calibration is None:
            N_calibration = calibration_size(dataloader, eval_mode, N)
        calibration_result = calibrate_noise(
            dataloader,
            build_model,
            stat,
            output_percentiles,
            params_list,
            noise_source,
            estimated_noises,
            measured_noises,
            params_dict,
            num_points=num_interpolation_points_,
            N=N,
            N_calibration=N_calibration,
            quantization=quantization,
            cache_dir=cache_dir,
            eval_mode=eval_mode,
        )
        # All noises are 0: nothing to calibrate
        if calibration_result is None:
            logger.info("no noise in this sweep: nothing to calibrate")
            calibrated_noises[noise_source] = estimated_noises
            accelerated_noise_time_s += (datetime.datetime.now() - accel_start).total_seconds()
            continue

        func, selected_indices, outputs_ideal_paths, quantize_suppress_factors_ns = (
            calibration_result
        )
        quantize_suppress_factors[noise_source] = quantize_suppress_factors_ns

        # Estimated -> calibrated noise
        calibrated_noises[noise_source] = func(estimated_noises)
        selected_noises[noise_source] = np.array(
            [calibrated_noises[noise_source][idx] for idx in selected_indices]
        )

        logger.info(f"done · {fmt_duration(time.monotonic() - source_start)}")
        accelerated_noise_time_s += (datetime.datetime.now() - accel_start).total_seconds()

    model_name = build_model(return_model_name=True)
    model_cache_dir = os.path.join(cache_dir, model_name)
    probe_layers = ["__output__"]
    noise_layers = model_cache.leaf_module_names(build_model)
    ideal_paths_by_quant = {}

    def _get_outputs_ideal_paths(params: dict):
        """Return ideal-output batch paths for this params' baseline (computed once per baseline)."""
        quant_params = baseline_params(params, quantization)
        ideal_base_dir = (
            _ideal_quant_cache_dir(model_cache_dir, probe_layers, quant_params)
            if quant_params is not None
            else get_ideal_cache_dir(model_cache_dir, probe_layers)
        )
        if ideal_base_dir not in ideal_paths_by_quant:
            outputs_ideal_paths, _, _ = run_noise_simulation(
                dataloader,
                build_model,
                stat,
                output_percentiles,
                noise_layers=noise_layers,
                probe_layers=probe_layers,
                params_list=[],
                N=N,
                save_result=False,
                return_performance=False,
                ideal_params=quant_params,
                cache_dir=cache_dir,
                eval_mode=eval_mode,
            )
            ideal_paths_by_quant[ideal_base_dir] = outputs_ideal_paths
        return ideal_paths_by_quant[ideal_base_dir]

    estimated_performance = {}
    ideal_performance = {}  # the baseline the errors are added to, same samples
    device = get_device(build_model=build_model)

    def get_error_from_cache(noise_source, noise_idx, batch_size, total_samples):
        """Load one batch's cached `__output__` error, or None."""
        params_cache_dir = get_cache_dir(
            params_dict[noise_source][noise_idx], parent_dir=model_cache_dir
        )
        error_path = os.path.join(
            params_cache_dir, f"{total_samples}_{total_samples+batch_size}.pt"
        )
        if os.path.exists(error_path):
            error_dict = torch.load(error_path, weights_only=False, map_location=device)
            error = error_dict["__output__"]
            return error
        return None

    # Estimate: ideal outputs + scaled cached errors
    section(f"Estimate · scoring {len(params_list)} params sets")
    estimation_loop_start = datetime.datetime.now()
    for idx in range(len(params_list)):
        match_index_dict = {ns: None for ns in noise_sources}
        error_scaling_factor_dict = {ns: None for ns in noise_sources}
        inv_distances_dict = {ns: None for ns in noise_sources}
        valid_param = True

        for ns in noise_sources:
            if not precise_modes[ns]:
                calibrated_noise = calibrated_noises[ns][idx]
                if calibrated_noise == 0:
                    continue

                if calibrated_noise == np.inf:
                    valid_param = False
                    break

                # Exact anchor match, else interpolate between bracketing anchors
                match_indices = np.isclose(
                    calibrated_noise, selected_noises[ns], atol=calibrated_noise * 1e-3
                )
                if np.any(match_indices):
                    match_index_dict[ns] = np.where(match_indices)[0][0]

                else:
                    inv_distances = calculate_piecewise_weights(
                        calibrated_noise, selected_noises[ns], quantize_suppress_factors[ns]
                    )
                    inv_distances_dict[ns] = inv_distances
                    error_scaling_factor_dict[ns] = np.sqrt(
                        calibrated_noise / (np.array(measured_noises[ns]) + 1e-12)
                    )

        if not valid_param:
            continue

        outputs_ideal_paths = _get_outputs_ideal_paths(params_list[idx])
        total_samples = 0
        for batch_idx, outputs_ideal_path in enumerate(outputs_ideal_paths):
            ideal_out_path = outputs_ideal_paths[batch_idx]
            ideal_out_obj = torch.load(ideal_out_path, weights_only=False, map_location=device)
            noisy_outputs, targets = (
                ideal_out_obj["outputs"]["__output__"],
                ideal_out_obj["targets"],
            )
            top_indices = ideal_out_obj["top_indices"]["__output__"]
            batch_size = get_batch_size((noisy_outputs, targets), N, total_samples)
            # Capture the prediction and score before the error terms mutate the tensor.
            ideal_top1 = predicted_class(noisy_outputs, top_indices)
            metric_fn(
                noisy_outputs,
                targets,
                ideal_performance,
                mode=eval_mode,
                param_idx=idx,
                return_counts=True,
                top_indices=top_indices,
            )

            for ns in noise_sources:
                if calibrated_noises[ns][idx] == 0:
                    continue

                # Precise mode: measured error
                if precise_modes[ns]:
                    error = get_error_from_cache(ns, idx, batch_size, total_samples)
                    if error is not None:
                        noisy_outputs += error

                # Anchor match: reuse its error
                elif match_index_dict[ns] is not None:
                    match_idx = match_index_dict[ns]
                    error = get_error_from_cache(ns, match_idx, batch_size, total_samples)
                    suppress_factor = quantize_suppress_factors[ns][match_idx]
                    if suppress_factor != 1.0:
                        noisy_outputs += error * suppress_factor
                    else:
                        noisy_outputs += error

                # Interpolate: weighted, scaled errors of the bracketing anchors
                else:
                    error_scaling_factor = error_scaling_factor_dict[ns]
                    inv_distances = inv_distances_dict[ns]
                    for noise_idx, inv_dist in enumerate(inv_distances):
                        if inv_dist == 0:
                            continue
                        error = get_error_from_cache(ns, noise_idx, batch_size, total_samples)
                        factor = error_scaling_factor[noise_idx] * inv_dist
                        noisy_outputs += error * factor

            metric_fn(
                noisy_outputs,
                targets,
                estimated_performance,
                mode=eval_mode,
                param_idx=idx,
                return_counts=True,
                top_indices=top_indices,
            )
            # Estimation only models error on the sampled indices, so the noisy
            # argmax is confined to them -- unlike the measured path, which sees
            # the whole output. The two agree while noise stays modest.
            update_flip_metrics(
                ideal_top1,
                predicted_class(noisy_outputs, top_indices),
                targets,
                estimated_performance,
                mode=eval_mode,
                param_idx=idx,
            )
            total_samples += batch_size
            if total_samples >= N:
                break
        if (idx + 1) % (len(params_list) // 10 + 1) == 0 or idx + 1 == len(params_list):
            elapsed_s = (datetime.datetime.now() - estimation_loop_start).total_seconds()
            logger.info(f"scored {idx + 1}/{len(params_list)} · {fmt_duration(elapsed_s)}")
    estimation_loop_time_s = (datetime.datetime.now() - estimation_loop_start).total_seconds()

    estimated_performance = average_performance_metrics(estimated_performance)
    ideal_performance = average_performance_metrics(ideal_performance)
    estimated_losses = _relative_losses(estimated_performance, ideal_performance)
    beyond_small_noise = [
        any(i < len(losses) and losses[i] > SMALL_NOISE_MAX_LOSS for losses in estimated_losses.values())
        for i in range(len(params_list))
    ]
    if any(beyond_small_noise):
        logger.warning(
            f"{sum(beyond_small_noise)} of {len(params_list)} params sets have an estimated loss "
            f"above {SMALL_NOISE_MAX_LOSS:.0%}, beyond the small-noise range: verify them with "
            f"--mode simulate (flagged in column beyond_small_noise)"
        )

    # Log first, in case saving fails
    total_noise = [
        sum(calibrated_noises[ns][i] for ns in noise_sources) for i in range(len(params_list))
    ]
    log_results("estimated", params_list, estimated_performance, {"__output__": total_noise})

    if save_result:
        model_name = build_model(return_model_name=True)
        # Every key of every params set (several grids can have different keys)
        keys = list(dict.fromkeys(k for params in params_list for k in params))
        data = []
        for i in range(len(params_list)):
            row = [params_list[i].get(k, "") for k in keys]
            for ns in noise_sources:
                row.append(calibrated_noises[ns][i])
            # Total across sources -> `estimated_noise`
            row.append(sum(calibrated_noises[ns][i] for ns in noise_sources))
            data.append(row)
        columns = keys + noise_sources + ["estimated_noise"]
        df = pd.DataFrame(data, columns=columns)

        estimated_cols = {
            f"estimated_{metric}": values for metric, values in estimated_performance.items()
        }
        df = df.assign(**estimated_cols)
        df = df.assign(
            **{f"ideal_{metric}": values for metric, values in ideal_performance.items()
               if "flip" not in metric},
            **{f"estimated_{metric}_loss": values for metric, values in estimated_losses.items()},
            beyond_small_noise=beyond_small_noise,
        )  # fmt: skip

        total_time_s = precise_noise_time_s + accelerated_noise_time_s + estimation_loop_time_s
        metadata = [
            {"key": "N", "value": N},
            {"key": "quantization", "value": quantization},
            {"key": "precise_mode_noise_sources", "value": sorted(precise_mode_noise_sources)},
            {"key": "max_calibration_noise", "value": max_calibration_noise},
            {"key": "num_interpolation_points", "value": num_interpolation_points},
            {
                "key": "loss_baseline",
                "value": "quantized, noiseless" if quantization == "precise" else "full precision",
            },
            {"key": "small_noise_max_loss", "value": SMALL_NOISE_MAX_LOSS},
            {"key": "save_result", "value": save_result},
            {"key": "precise_noise_time_s", "value": precise_noise_time_s},
            {"key": "accelerated_noise_time_s", "value": accelerated_noise_time_s},
            {"key": "estimation_loop_time_s", "value": estimation_loop_time_s},
            {"key": "total_time_s", "value": total_time_s},
        ]
        metadata_df = pd.DataFrame(metadata)

        xlsx_path = results_path(model_name, "estimated", run_name, N)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
            metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        logger.info(f"saved {short_path(xlsx_path)}")

    return calibrated_noises, estimated_performance
