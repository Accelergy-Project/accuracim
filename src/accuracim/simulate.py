"""Ground truth: measure the output noise with the noisy model.

Two passes go over the same batches:
1. The ideal model caches its top-k outputs (full precision, or quantization only
   with `ideal_params`).
2. Each noisy params set is scored against this cache (output MSE and task metrics).
All results are cached in `cache/` by (params, batch range). Sweeps can resume, and repeated
runs take no time.
"""

import datetime
import logging
import os
import time
from contextlib import nullcontext

import pandas as pd
import torch

from accuracim import model_cache
from accuracim.config import results_path
from accuracim.device import free_memory
from accuracim.evaluation import (
    average_performance_metrics,
    get_batch_size,
    predicted_class,
    resolve_eval_mode,
    run_batch,
    update_flip_metrics,
    update_performance_metrics,
)
from accuracim.graph import register_fused_conv_bn_hook
from accuracim.log import (
    describe_params,
    fmt_duration,
    fmt_metrics,
    fmt_noise,
    indented,
    log_results,
    short_path,
    swept_keys,
)
from accuracim.noise import restore_noise_seed, set_noise_seed
from accuracim.params import (
    get_cache_dir,
    get_ideal_cache_dir,
    get_output_percentiles_per_params,
)
from accuracim.sampling import gather_top_percentile_values, get_top_percentile_indices

logger = logging.getLogger(__name__)


def _ideal_quant_cache_dir(
    model_cache_dir: str, probe_layers: list[str], quant_params: dict
) -> str:
    """Return the ideal-output cache dir for a quantized baseline."""
    quant_parent = os.path.join(model_cache_dir, "ideal_quant")
    quant_dir = get_cache_dir(quant_params, parent_dir=quant_parent)
    return get_ideal_cache_dir(quant_dir, probe_layers)


# One results workbook per model per process
_MEASURED_XLSX_PATHS: dict[str, str] = {}

# Ideal-run batch boundaries per (dataloader, batch size, N): the same loader yields the
# same batches, so a later call can find cached batches without loading data
_LOADER_SEGMENTS: dict[tuple, list[tuple[int, int]]] = {}


class _Batches:
    """The dataloader's batches by index, loaded only when a batch has to run.

    Cached batches are skipped without loading; reaching a later batch loads the ones
    before it, exactly as iterating the loader would. `timer["data"]` accrues load time.
    """

    def __init__(self, dataloader, timer: dict):
        self.dataloader, self.timer = dataloader, timer
        self.iterator, self.position, self.batch = None, 0, None

    def get(self, index: int):
        """Return batch `index` (indices must not decrease), or None past the end."""
        start = time.monotonic()
        try:
            if self.iterator is None:
                self.iterator = iter(self.dataloader)
            while self.position <= index:
                self.batch = next(self.iterator)
                self.position += 1
        except StopIteration:
            self.batch = None
        finally:
            self.timer["data"] += time.monotonic() - start
        return self.batch

    def close(self):
        self.iterator = self.batch = None


def _run_batch(model, batch_data, num_processed: int, N: int) -> dict:
    """`run_batch`, explaining a GPU out-of-memory caused by GPU-resident weight slices."""
    try:
        return run_batch(model, batch_data, num_processed=num_processed, N=N)
    except RuntimeError as e:
        if "out of memory" in str(e) and getattr(model, "_weight_slices_on_gpu", False):
            raise RuntimeError(
                "GPU out of memory with weight slices kept on the GPU; "
                "set `run.weight_slices: cpu` in the config"
            ) from e
        raise


def _log_timing(what: str, start: float, timer: dict) -> None:
    total = time.monotonic() - start
    compute = max(total - timer["model"] - timer["data"], 0.0)
    logger.debug(
        f"{what} timing · model {fmt_duration(timer['model'])} · "
        f"data {fmt_duration(timer['data'])} · compute {fmt_duration(compute)}"
    )


def _batch_cache_paths(cache_dir: str, start_idx: int, end_idx: int) -> tuple[str, str, str]:
    """Return the (error, stats, noise) file paths for one batch."""
    stem = os.path.join(cache_dir, f"{start_idx}_{end_idx}")
    return f"{stem}.pt", f"{stem}_stats.pt", f"{stem}_noise.pt"


def _is_batch_cached(
    cache_dir: str,
    start_idx: int,
    end_idx: int,
    save_result: bool,
    return_performance: bool,
    save_error: bool,
) -> bool:
    """Whether the cache files this run would write all exist (stats only if results/performance are requested)."""
    error_path, stat_path, noise_path = _batch_cache_paths(cache_dir, start_idx, end_idx)
    return (
        ((not save_result and not return_performance) or os.path.exists(stat_path))
        and ((not save_error) or os.path.exists(error_path))
        and os.path.exists(noise_path)
    )


def _is_lookahead_cached(
    cache_dir: str,
    start_idx: int,
    end_idx: int,
    save_result: bool,
    return_performance: bool,
    save_error: bool,
) -> bool:
    """Whether the next two batches are cached too, so this one can skip loading.

    The last file written before an interrupt may be truncated, so one batch ahead is not enough.
    """
    batch_size = end_idx - start_idx
    bounds = [start_idx + batch_size * i for i in range(1, 4)]
    return all(
        _is_batch_cached(cache_dir, s_i, e_i, save_result, return_performance, save_error)
        for s_i, e_i in zip(bounds[:-1], bounds[1:])
    )


def run_noise_simulation(
    dataloader: torch.utils.data.DataLoader,
    build_model: callable,
    stat: dict,
    output_percentiles: dict,
    noise_layers: list[str],
    probe_layers: list[str],
    params_list: list[dict] = [],
    N: int = 500,
    save_result: bool = True,
    save_error: bool = False,
    return_performance: bool = True,
    eval_mode: str | None = None,
    metric_fn: callable = None,
    noise_seed: int | None = None,
    ideal_params: dict | None = None,
    xlsx_path: str = None,
    run_name: str | None = None,
    cache_dir: str = "cache",
) -> tuple:
    """Simulate each params set and measure its output noise against the cached ideal run.

    Returns:
    tuple: (outputs_ideal_paths, noises, avg_performance)
    - outputs_ideal_paths (list[str]): per-batch ideal-output files.
    - noises (dict[str, list[float]]): per probe layer, mean squared error per params set.
    - avg_performance (dict): averaged metrics of the noisy model(s).
    """
    model_name = build_model(return_model_name=True)
    device = build_model(return_device=True)
    if metric_fn is None:
        metric_fn = update_performance_metrics
    if eval_mode is None:
        eval_mode = build_model(return_mode=True)
    eval_mode = resolve_eval_mode(eval_mode)

    run_datetime = datetime.datetime.now()
    max_batch_size_seen = 0

    prev_noise_state = None
    if noise_seed is not None:
        prev_noise_state = set_noise_seed(noise_seed)
    model_cache_dir = os.path.join(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)

    # Ideal cache: full precision or quantized baseline
    if ideal_params is None:
        ideal_base_dir = get_ideal_cache_dir(model_cache_dir, probe_layers)
    else:
        ideal_base_dir = _ideal_quant_cache_dir(model_cache_dir, probe_layers, ideal_params)
    ideal_outputs_dir = os.path.join(ideal_base_dir, "outputs")
    ideal_stats_dir = os.path.join(ideal_base_dir, "stats")
    os.makedirs(ideal_outputs_dir, exist_ok=True)
    os.makedirs(ideal_stats_dir, exist_ok=True)

    existing_outputs = set(os.listdir(ideal_outputs_dir))
    existing_stats = set(os.listdir(ideal_stats_dir))

    # Paths only, to keep memory low
    outputs_ideal_paths = []
    segments: list[tuple[int, int]] = []

    ideal_performance_counts = {}
    total_numels = {pl: 0 for pl in probe_layers}

    # --- Ideal model: cache top-k outputs per batch ---
    def acquire_ideal_model():
        if ideal_params is None:
            return model_cache.ideal_model(build_model)
        output_percentiles_params = get_output_percentiles_per_params(
            output_percentiles, ideal_params
        )
        return model_cache.noisy_model(
            build_model, noise_layers, ideal_params, stat, output_percentiles_params
        )

    # The whole-model output is always hookable, so a model is only needed for a batch
    # to run. Other probe layers are checked against the model up front, as before.
    lazy = set(probe_layers) <= {"__output__"}
    model = None
    store_ideal = {pl: {} for pl in probe_layers}
    handles_ideal = []
    timer = {"model": 0.0, "data": 0.0}

    def prepare_ideal_model():
        nonlocal model, probe_layers
        model_start = time.monotonic()
        model = acquire_ideal_model()
        model.eval()
        active_probe_layers = []
        for pl in probe_layers:
            h = register_fused_conv_bn_hook(model, pl, {}, store_ideal[pl], detach_cpu=False)
            if h is not None:
                handles_ideal.append(h)
                active_probe_layers.append(pl)
            else:
                store_ideal.pop(pl)
        probe_layers = active_probe_layers
        timer["model"] += time.monotonic() - model_start

    total_samples = 0
    ideal_batches_run = 0
    ideal_start = time.monotonic()
    # Batch boundaries of this loader and N, known from an earlier call: cached batches
    # then need no data loaded
    loader_key = (id(dataloader), getattr(dataloader, "batch_size", None), N)
    known_segments = _LOADER_SEGMENTS.get(loader_key)
    batches = _Batches(dataloader, timer)
    try:
        if not lazy:
            prepare_ideal_model()
        with torch.no_grad():
            seg_idx = 0
            while total_samples < N:
                batch_data = None
                if known_segments is not None:
                    if seg_idx >= len(known_segments):
                        break
                    batch_size = known_segments[seg_idx][1] - known_segments[seg_idx][0]
                else:
                    batch_data = batches.get(seg_idx)
                    if batch_data is None:  # loader exhausted
                        break
                    batch_size = get_batch_size(batch_data, N, total_samples)
                max_batch_size_seen = max(max_batch_size_seen, batch_size)
                end_idx = total_samples + batch_size
                seg_name = f"{total_samples}_{end_idx}"
                out_file = f"{seg_name}.pt"
                stat_file = f"{seg_name}_stats.pt"
                out_path = os.path.join(ideal_outputs_dir, out_file)
                stat_path = os.path.join(ideal_stats_dir, stat_file)
                outputs_ideal_paths.append(out_path)

                have_outputs = out_file in existing_outputs
                have_stats = stat_file in existing_stats

                if have_outputs and have_stats:
                    # Cached: load stats only
                    stat_obj = torch.load(stat_path, weights_only=False, map_location="cpu")
                    ideal_performance_counts = stat_obj.get("performance_counts", {})
                    total_numels = stat_obj.get("total_numels", total_numels)
                    segments.append((total_samples, end_idx))

                else:
                    ideal_batches_run += 1
                    if batch_data is None:
                        batch_data = batches.get(seg_idx)
                    if model is None:
                        prepare_ideal_model()
                    batch_result = _run_batch(model, batch_data, total_samples, N)
                    logits_ideal, targets = batch_result["logits"], batch_result["targets"]

                    metric_fn(
                        logits_ideal,
                        targets,
                        ideal_performance_counts,
                        mode=eval_mode,
                        return_counts=True,
                    )

                    # Keep only the top 2.5% per position
                    save_outputs = {}
                    top_indices = {}
                    for pl in probe_layers:
                        out_tensor = store_ideal[pl]["output"]
                        top_indices[pl] = get_top_percentile_indices(
                            tensor=out_tensor,
                            p=2.5,
                        )
                        save_outputs[pl] = gather_top_percentile_values(
                            out_tensor, top_indices[pl]
                        )
                        total_numels[pl] += save_outputs[pl].numel()
                        store_ideal[pl].clear()

                    torch.save(
                        {"outputs": save_outputs, "targets": targets, "top_indices": top_indices},
                        out_path,
                    )
                    torch.save(
                        {
                            "performance_counts": ideal_performance_counts,
                            "total_numels": total_numels,
                        },
                        stat_path,
                    )

                    segments.append((total_samples, end_idx))

                total_samples += batch_size
                seg_idx += 1
    except BaseException:
        if model is not None:
            model_cache.discard(model)
        raise
    finally:
        for h in handles_ideal:
            h.remove()
        batches.close()
    _LOADER_SEGMENTS[loader_key] = list(segments)
    free_after = model is not None and not model_cache.is_cached(model)
    model = None
    if free_after:
        free_memory()
    _log_timing("ideal", ideal_start, timer)

    ideal_performance = average_performance_metrics(ideal_performance_counts)
    if ideal_batches_run:
        logger.info(
            f"ideal run · {ideal_batches_run} of {len(segments)} batches computed · "
            f"{fmt_metrics(ideal_performance)} · {fmt_duration(time.monotonic() - ideal_start)}"
        )
    logger.debug(f"Ideal outputs: {ideal_base_dir}")

    # --- Noisy runs ---
    performance_metrics = {}
    noises = {pl: [0.0 for _ in params_list] for pl in probe_layers}
    param_run_stats = []

    def _performance_metrics_snapshot(performance_metrics: dict, param_idx: int) -> dict:
        """Return this params set's metric totals, for caching."""
        snapshot = {}
        for metric_name, metric_value in performance_metrics.items():
            if isinstance(metric_value, list):
                snapshot[metric_name] = (
                    metric_value[param_idx] if param_idx < len(metric_value) else 0
                )
            else:
                snapshot[metric_name] = metric_value
        return snapshot

    def _restore_performance_metrics_snapshot(
        performance_metrics: dict, snapshot: dict, param_idx: int
    ) -> None:
        """Write a cached snapshot back into `performance_metrics[*][param_idx]`."""
        for metric_name, metric_value in snapshot.items():
            if metric_name not in performance_metrics:
                performance_metrics[metric_name] = []

            current_len = len(performance_metrics[metric_name])
            if current_len <= param_idx:
                performance_metrics[metric_name].extend([0] * (param_idx - current_len + 1))

            performance_metrics[metric_name][param_idx] = metric_value

        # A snapshot cached before a metric existed simply lacks it. Keep every
        # column the same length anyway, so one stale file cannot desync them.
        for totals in performance_metrics.values():
            if len(totals) <= param_idx:
                totals.extend([0] * (param_idx - len(totals) + 1))

    swept = swept_keys(params_list)
    for i, params in enumerate(params_list):
        cache_dir_params = get_cache_dir(params, parent_dir=model_cache_dir)
        os.makedirs(cache_dir_params, exist_ok=True)
        if len(params_list) > 1:
            logger.info(f"[{i + 1}/{len(params_list)}] {describe_params(params, swept)}".rstrip())
        logger.debug(f"Params: {params}")
        logger.debug(f"Cache: {cache_dir_params}")

        output_percentiles_params = get_output_percentiles_per_params(output_percentiles, params)

        # Built (or reused) on the first batch that actually runs
        model_noisy = None
        store_noisy = {pl: {} for pl in probe_layers}
        handles_noisy = []
        timer = {"model": 0.0, "data": 0.0}

        def prepare_noisy_model():
            nonlocal model_noisy
            model_start = time.monotonic()
            model_noisy = model_cache.noisy_model(
                build_model, noise_layers, params, stat, output_percentiles_params
            )
            model_noisy.eval()
            for pl in probe_layers:
                handles_noisy.append(
                    register_fused_conv_bn_hook(
                        model_noisy, pl, {}, store_noisy[pl], detach_cpu=False
                    )
                )
            timer["model"] += time.monotonic() - model_start

        param_actual_samples = 0
        total_samples = 0

        cache_flags = (save_result, return_performance, save_error)
        start_time = datetime.datetime.now()
        param_start = time.monotonic()
        batches = _Batches(dataloader, timer)
        try:
            if not lazy:
                prepare_noisy_model()
            with torch.no_grad():
                for seg_idx, (start_idx, end_idx) in enumerate(segments):
                    if total_samples >= N:
                        break
                    # Batches match the ideal run's segments (the last may be truncated by N)
                    batch_size = end_idx - start_idx
                    max_batch_size_seen = max(max_batch_size_seen, batch_size)
                    error_path, stat_path, noise_path = _batch_cache_paths(
                        cache_dir_params, start_idx, end_idx
                    )
                    skip_run = _is_batch_cached(cache_dir_params, start_idx, end_idx, *cache_flags)

                    if skip_run:
                        skip_load = _is_lookahead_cached(
                            cache_dir_params, start_idx, end_idx, *cache_flags
                        )
                        if not skip_load:
                            try:
                                cached_noise = torch.load(
                                    noise_path, weights_only=False, map_location="cpu"
                                )
                                for pl in probe_layers:
                                    noises[pl][i] = cached_noise[pl]
                                # Noise file loaded cleanly: restore metrics too
                                if save_result or return_performance:
                                    new_performance_metrics = torch.load(
                                        stat_path, weights_only=False, map_location="cpu"
                                    )
                                    _restore_performance_metrics_snapshot(
                                        performance_metrics, new_performance_metrics, i
                                    )
                            except Exception as e:
                                skip_run = False
                                logger.warning(
                                    f"Cache load failed for params set {i + 1}, segment {seg_idx} "
                                    f"({e}); re-running it"
                                )
                                logger.debug(f"Cache paths: {stat_path}, {noise_path}, {error_path}")
                                # Drop possibly corrupt files and re-run
                                for p in (stat_path, noise_path, error_path):
                                    if os.path.exists(p):
                                        os.remove(p)

                    # Not cached, or load failed: run
                    if not skip_run:
                        batch_data = batches.get(seg_idx)
                        if batch_data is None:
                            raise RuntimeError(f"Dataloader ended before cached segment {seg_idx}")
                        if model_noisy is None:
                            prepare_noisy_model()
                        result_dict = _run_batch(model_noisy, batch_data, total_samples, N)
                        param_actual_samples += batch_size
                        logits_noisy, targets = result_dict["logits"], result_dict["targets"]

                        error_dict = {}
                        noise_dict = {}
                        # Ideal outputs for this batch
                        ideal_out_path = outputs_ideal_paths[seg_idx]
                        ideal_out_obj = torch.load(
                            ideal_out_path, weights_only=False, map_location=device
                        )
                        ideal_outputs_batch = ideal_out_obj["outputs"]

                        # Only the model's own output has predictions to flip.
                        ideal_top1 = noisy_top1 = None

                        for pl in probe_layers:
                            top_indices_batch = ideal_out_obj["top_indices"][pl]
                            out_ideal = ideal_outputs_batch[pl]
                            out_noisy = gather_top_percentile_values(
                                store_noisy[pl]["output"], top_indices_batch
                            )
                            error = out_noisy - out_ideal
                            error -= error.mean().item()  # zero-mean

                            # Sum of squared error (normalized after the loop)
                            noises[pl][i] += error.pow(2).sum().item()
                            noise_dict[pl] = noises[pl][i]

                            if save_error:
                                error_dict[pl] = error
                            if pl == "__output__":
                                # Noisy argmax over the full output, so a class that
                                # climbs from outside the ideal top-k still counts.
                                ideal_top1 = predicted_class(out_ideal, top_indices_batch)
                                noisy_top1 = predicted_class(store_noisy[pl]["output"])

                            store_noisy[pl].clear()
                            del error

                        if save_result or return_performance:
                            metric_fn(
                                logits_noisy,
                                targets,
                                performance_metrics,
                                mode=eval_mode,
                                param_idx=i,
                                return_counts=True,
                            )
                            if ideal_top1 is not None:
                                update_flip_metrics(
                                    ideal_top1,
                                    noisy_top1,
                                    targets,
                                    performance_metrics,
                                    mode=eval_mode,
                                    param_idx=i,
                                )

                        if not os.path.exists(noise_path):
                            torch.save(noise_dict, noise_path)
                        if save_result:
                            torch.save(_performance_metrics_snapshot(performance_metrics, i), stat_path)
                        if save_error:
                            torch.save(error_dict, error_path)

                    total_samples += batch_size
                    if total_samples >= N:
                        break
        except BaseException:
            if model_noisy is not None:
                model_cache.discard(model_noisy)
            raise
        finally:
            for h in handles_noisy:
                if h is not None:
                    h.remove()
            batches.close()
        param_elapsed_s = (datetime.datetime.now() - start_time).total_seconds()
        param_run_stats.append(
            {
                "actual_samples_run": param_actual_samples,
                "param_time_s": param_elapsed_s,
            }
        )
        free_after = model_noisy is not None and not model_cache.is_cached(model_noisy)
        model_noisy = None
        if free_after:
            free_memory()
        if param_actual_samples:
            _log_timing(f"params set {i + 1}", param_start, timer)
        # Sum -> mean squared error
        for pl in probe_layers:
            denom = total_numels[pl]
            noises[pl][i] = noises[pl][i] / denom if denom > 0 else 0.0

        # This params set's result, as soon as it is done
        result = [
            f"noise{'' if pl == '__output__' else f'[{pl}]'} {fmt_noise(noises[pl][i])}"
            for pl in probe_layers
        ]
        if (save_result or return_performance) and "count" in performance_metrics:
            snapshot = _performance_metrics_snapshot(performance_metrics, i)
            result.append(fmt_metrics(average_performance_metrics(snapshot)))
        result.append(fmt_duration(param_elapsed_s) if param_actual_samples else "cached")
        with indented() if len(params_list) > 1 else nullcontext():
            logger.info(" · ".join(r for r in result if r))

    if save_result or return_performance:
        avg_performance = average_performance_metrics(performance_metrics)
    else:
        avg_performance = performance_metrics

    # Log first, in case saving fails
    if save_result or len(params_list) > 1:
        log_results(
            "measured",
            params_list,
            avg_performance if (save_result or return_performance) else {},
            noises,
        )

    # --- Save results ---
    if len(params_list) > 0 and save_result:
        model_name = build_model(return_model_name=True)
        # Every key of every params set (several grids can have different keys)
        keys = list(dict.fromkeys(k for params in params_list for k in params))
        data = []
        for i in range(len(params_list)):
            row = [params_list[i].get(k, "") for k in keys]
            for pl in probe_layers:
                row.append(noises[pl][i])
            data.append(row)
        columns = keys + probe_layers
        df = pd.DataFrame(data, columns=columns)

        num_rows = len(params_list)
        ideal_perf_cols = {
            f"ideal_{metric}": [val] * num_rows for metric, val in ideal_performance.items()
        }
        measured_perf_cols = {
            f"measured_{metric}": vals for metric, vals in avg_performance.items()
        }
        df = df.assign(**ideal_perf_cols, **measured_perf_cols)
        if len(param_run_stats) == num_rows:
            param_actual_samples = [stat["actual_samples_run"] for stat in param_run_stats]
            param_time_s = [stat["param_time_s"] for stat in param_run_stats]
            df = df.assign(actual_samples_run=param_actual_samples, param_time_s=param_time_s)

        if xlsx_path is None:
            # Reuse this process's workbook: repeated calls add sheets, not files
            if model_name not in _MEASURED_XLSX_PATHS:
                _MEASURED_XLSX_PATHS[model_name] = results_path(model_name, "measured", run_name, N)
            xlsx_path = _MEASURED_XLSX_PATHS[model_name]

        total_actual_samples_run = sum(stat["actual_samples_run"] for stat in param_run_stats)
        total_param_time_s = sum(stat["param_time_s"] for stat in param_run_stats)
        batch_size_meta = getattr(dataloader, "batch_size", None)
        if batch_size_meta is None:
            batch_size_meta = max_batch_size_seen

        metadata = [
            {"key": "run_datetime", "value": run_datetime.isoformat()},
            {"key": "N", "value": N},
            {"key": "save_result", "value": save_result},
            {"key": "save_error", "value": save_error},
            {"key": "noise_seed", "value": noise_seed},
            {"key": "xlsx_path", "value": xlsx_path},
            {"key": "total_actual_samples_run", "value": total_actual_samples_run},
            {"key": "total_param_time_s", "value": total_param_time_s},
            {"key": "batch_size", "value": batch_size_meta},
        ]
        metadata_df = pd.DataFrame(metadata)

        sheet_name = "simulation_results"
        mode = "a" if os.path.exists(xlsx_path) else "w"
        if_sheet_exists = "replace" if mode == "a" else None
        with pd.ExcelWriter(
            xlsx_path, engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        logger.info(f"saved {short_path(xlsx_path)} (sheet {sheet_name})")

    if prev_noise_state is not None:
        restore_noise_seed(prev_noise_state)

    return outputs_ideal_paths, noises, avg_performance
