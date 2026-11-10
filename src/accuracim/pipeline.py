"""The experiment: `run_experiment` runs a config from start to end.

It resolves the model, dataset and metric, and expands the params grid.
Then it calibrates the input ranges and the ADC ranges (these depend on the slicing).
Then it runs `estimate`, `simulate` or both.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import itertools
import logging
import math
import os
import time
from typing import Any

from accuracim import model_cache
from accuracim import noise as noise_profiles
from accuracim.check import run_model_check
from accuracim.config import REGISTRY, import_callable, project_root
from accuracim.device import get_best_gpu
from accuracim.estimate import run_noise_estimation
from accuracim.evaluation import resolve_eval_mode, update_performance_metrics
from accuracim.log import (
    default_log_path,
    fmt_duration,
    section,
    setup_console,
    setup_file,
    short_path,
)
from accuracim.noise import Noise, check_profile
from accuracim.params import KNOWN_PARAMS
from accuracim.quantize import set_weight_slice_placement
from accuracim.simulate import run_noise_simulation
from accuracim.sources import QUANTIZATION_MODES
from accuracim.statistics import dict_to_device, get_adc_statistics, get_input_statistics

logger = logging.getLogger(__name__)

#: Noise `type` names: every `Noise` subclass in `accuracim.noise`, found automatically.
#: Classes defined elsewhere are named by a "module:Class" path instead.
_NOISE_CLASS_MAP = {
    name: obj
    for name, obj in vars(noise_profiles).items()
    if isinstance(obj, type) and issubclass(obj, Noise) and obj is not Noise
}

#: Params-key suffixes whose grid values are noise specs.
_NOISE_KEY_SUFFIXES = ("_physical_noise", "_drift")


def _register_defaults() -> None:
    """Register the built-in "default" metric.

    Models and datasets have no default: configs name them, e.g.
    `workloads.models:build_model` and `workloads.dataset:get_imagenet_dataloader`.
    """
    if "default" not in REGISTRY.evaluations.items:
        REGISTRY.evaluations.register("default", update_performance_metrics)


def _expand_sweeps(values: list | dict) -> list:
    """Expand `{type, sweep: {start, stop, step | num, scale: log}}` entries into one spec per value.

    The value is the first arg; `stop` is included; `scale: log` spaces `num` values evenly in log.
    """
    if isinstance(values, dict):
        values = [values]
    expanded = []
    for spec in values:
        if not (isinstance(spec, dict) and "sweep" in spec):
            expanded.append(spec)
            continue
        sweep = spec["sweep"]
        start, stop = float(sweep["start"]), float(sweep["stop"])
        if "step" in sweep:
            num = round((stop - start) / float(sweep["step"])) + 1
        else:
            num = int(sweep["num"])
        fractions = [k / (num - 1) if num > 1 else 0.0 for k in range(num)]
        if sweep.get("scale") == "log":
            points = [start * (stop / start) ** f for f in fractions]
        else:
            points = [start + (stop - start) * f for f in fractions]
        base = {k: v for k, v in spec.items() if k != "sweep"}
        expanded += [{**base, "args": [float(f"{p:.12g}"), *base.get("args", [])]} for p in points]
    return expanded


def get_params_list(params_grid_list: list | list[dict]) -> list[dict]:
    """Expand one or more `{key: [values]}` grids into a flat list of param sets."""
    params_list = []
    if isinstance(params_grid_list, dict):
        params_grid_list = [params_grid_list]

    for params_grid in params_grid_list:
        keys = list(params_grid.keys())
        axes = [
            _expand_sweeps(params_grid[k]) if k.endswith(_NOISE_KEY_SUFFIXES) else params_grid[k]
            for k in keys
        ]
        combos = [dict(zip(keys, values)) for values in itertools.product(*axes)]
        params_list += combos

    return params_list


def chunk_params(params_list: list[dict], group_index: int, num_groups: int) -> list[dict]:
    """Return one contiguous chunk of `params_list` (for parallel runs)."""
    if num_groups <= 0:
        raise ValueError("--num-groups must be positive")
    if group_index < 0 or group_index >= num_groups:
        raise ValueError("--group-index must be in [0, --num-groups)")

    if not params_list:
        return []

    chunk_size = math.ceil(len(params_list) / num_groups)
    start = group_index * chunk_size
    end = min(start + chunk_size, len(params_list))
    return params_list[start:end]


def _instantiate_noise(spec: Any) -> Any:
    """Build a `Noise` from a {"type", "args", "kwargs"} spec; pass anything else through.

    `type`: a class in `accuracim.noise`, or a "module:Class" path. A list of specs (several
    profiles on one tensor) becomes a list of profiles; nested specs in args/kwargs are
    built too.
    """
    if isinstance(spec, Noise):
        return spec
    if isinstance(spec, list):
        return [_instantiate_noise(item) for item in spec]
    if isinstance(spec, dict) and "type" in spec:
        noise_type = spec["type"]
        cls = _NOISE_CLASS_MAP.get(noise_type)
        if cls is None and ":" in noise_type:
            cls = import_callable(noise_type)
        if cls is None:
            raise ValueError(
                f"Unknown noise type: {noise_type}. Use a class in accuracim/noise.py or a 'module:Class' path."
            )
        args = [_instantiate_noise(a) for a in spec.get("args", [])]
        kwargs = {k: _instantiate_noise(v) for k, v in spec.get("kwargs", {}).items()}
        return cls(*args, **kwargs)
    if isinstance(spec, str):
        try:
            return float(spec)
        except ValueError:
            return spec
    return spec


def _normalize_params_grid(params_grid: dict) -> dict:
    """Instantiate the noise specs in a params grid."""
    normalized = {}
    for key, value in params_grid.items():
        if key.endswith(_NOISE_KEY_SUFFIXES) and isinstance(value, (list, dict)):
            normalized[key] = [_instantiate_noise(v) for v in _expand_sweeps(value)]
        else:
            normalized[key] = value
    return normalized


def _check_profiles(params_list: list[dict], precise_mode_noise_sources: set) -> None:
    """Stop before an estimate sweep if an interpolated noise profile breaks the contract.

    See `noise.check_profile`. Stationary profiles (drift) are folded into the baseline and
    sources measured exactly (`precise_mode_noise_sources`) are not interpolated: skipped.
    """
    checked = set()
    for params in params_list:
        for stage in ("input", "weight", "adc"):
            if any(p in f"{stage}_physical" for p in precise_mode_noise_sources):
                continue
            noise = params.get(f"{stage}_physical_noise")
            for profile in noise if isinstance(noise, list) else [noise]:
                if not isinstance(profile, Noise) or profile.stationary:
                    continue
                if repr(profile) not in checked:
                    checked.add(repr(profile))
                    check_profile(profile)


def _resolve_base_dir(config: dict[str, Any]) -> str:
    base_dir = config.get("base_dir")
    if base_dir:
        return os.path.abspath(base_dir)
    return project_root()


def _attention_row(model_cfg: dict) -> str:
    """How attention runs, as configured: simulated (analog) or the model's own kernel."""
    kernel = model_cfg.get("attn_implementation")
    if model_cfg.get("analog_attention", True):
        return "analog (matmuls simulated)"
    return f"digital ({kernel or 'model default'} kernel)"


def _log_header(config, model_fn, model_name, dataloader, dataset_cfg, eval_mode, run, log_path):
    """Log what this run is: config, model, dataset, metric, run settings, device, log file."""
    loader_params, eval_params = dataset_cfg.get("params", {}), dataset_cfg.get("eval_params") or {}
    skip = ("batch_size", "num_workers", "deterministic_dataloader")
    details = [f"{k}={v}" for k, v in {**loader_params, **eval_params}.items() if k not in skip]
    batch_sizes = [b for b in (loader_params.get("batch_size"), eval_params.get("batch_size")) if b]
    if batch_sizes:
        details.append("batch " + "/".join(str(b) for b in dict.fromkeys(batch_sizes)))
    try:
        details.insert(0, f"{len(dataloader.dataset):,} samples")
    except (AttributeError, TypeError):
        pass
    dataset_spec = dataset_cfg.get("loader") or dataset_cfg.get("path") or dataset_cfg.get("name")

    model_mode = model_fn(return_mode=True)
    if eval_mode is None:
        eval_mode = f"{resolve_eval_mode(model_mode)} (model default)"
    try:
        version = importlib.metadata.version("accuracim")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    num_params = len(get_params_list(config.get("params_grid_list", [])))

    rows = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.get("config_path") or config.get("name"),
        "model": f"{model_name} ({model_mode})",
        **(
            {
                "attention": _attention_row(
                    config["model"] if isinstance(config.get("model"), dict) else {}
                )
            }
            if model_mode == "transformer"
            else {}
        ),
        "dataset": " · ".join([str(dataset_spec)] + details),
        "metric": eval_mode,
        "run": " · ".join(f"{k} {v}" for k, v in run.items()) + f" · {num_params} params sets",
        "device": model_fn(return_device=True),
        "log": short_path(log_path),
    }
    logger.info(f"AccuraCiM {version}".rstrip())
    for key, value in rows.items():
        logger.info(f"  {key:<8} {value}")


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run calibration, then estimation and/or simulation, from a config dict.

    Returns:
    dict: {"estimated", "simulated", "params_list"}; "estimated" is None
    (estimation results are printed and saved to xlsx).
    """
    _register_defaults()

    model_cfg = config.get("model", {})
    if isinstance(model_cfg, str):
        model_cfg = {"name": model_cfg}

    model_fn_spec = model_cfg.get("builder") or model_cfg.get("path")
    model_fn = REGISTRY.models.resolve(model_fn_spec)

    # A builder serving several models (workloads.models) selects one by `name`, and
    # (transformers) whether attention is simulated. Without them, the builder decides.
    selection = {
        builder_key: model_cfg[config_key]
        for config_key, builder_key in (
            ("name", "model_name"),
            ("analog_attention", "analog_attention"),
            ("attn_implementation", "attn_implementation"),
        )
        if model_cfg.get(config_key) is not None
    }
    if selection:
        model_fn(**selection, return_model_name=True)
    model_name = model_fn(return_model_name=True)

    # Console (unless the CLI set logging up already) and this run's log file
    if not logger.parent.handlers:
        setup_console()
    log_path = setup_file(default_log_path(project_root(), model_name, config.get("name")))
    run_start = time.monotonic()

    dataset_cfg = config.get("dataset", {})
    if isinstance(dataset_cfg, str):
        dataset_cfg = {"name": dataset_cfg}
    dataset_fn_spec = (
        dataset_cfg.get("loader") or dataset_cfg.get("path") or dataset_cfg.get("name")
    )
    dataloader_fn = REGISTRY.datasets.resolve(dataset_fn_spec)

    dataloader_params = dataset_cfg.get("params", {})
    dataloader = dataloader_fn(**dataloader_params)

    eval_cfg = config.get("evaluation", {})
    metric_fn_spec = eval_cfg.get("metric_fn") or eval_cfg.get("path") or eval_cfg.get("name")
    metric_fn = REGISTRY.evaluations.resolve(metric_fn_spec or "default")
    # Metric: the config's mode, else the one the dataloader declares (`loader.eval_mode`);
    # estimate/simulate fall back to the model's mode. Checked now, before calibration.
    eval_mode = eval_cfg.get("mode") or getattr(dataloader, "eval_mode", None)
    if eval_mode is not None:
        eval_mode = resolve_eval_mode(eval_mode)

    run_cfg = config.get("run", {})
    N = run_cfg.get("N", 128)
    save_result = run_cfg.get("save_result", True)
    run_name = config.get("name")
    mode = run_cfg.get("mode", "both")
    if mode not in {"both", "estimate", "simulate", "check"}:
        raise ValueError(f"Unknown run mode: {mode}. Expected both, estimate, simulate, or check.")

    # Speed only, never results: reuse built/programmed models across runs
    # (`model_cache`), and where weight slices live between passes
    model_cache.configure(reuse_models=run_cfg.get("reuse_models", True))
    set_weight_slice_placement(run_cfg.get("weight_slices", "auto"))

    # Quantization-error accounting; see `accuracim.sources`
    quantization = run_cfg.get("quantization", "precise")
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(
            f"Unknown run quantization: {quantization}. Expected one of {QUANTIZATION_MODES}."
        )

    grids = config.get("params_grid_list", [])
    unknown = {k for grid in ([grids] if isinstance(grids, dict) else grids) for k in grid}
    if unknown - KNOWN_PARAMS:
        raise ValueError(
            f"Unknown params key(s) {sorted(unknown - KNOWN_PARAMS)}; known: {sorted(KNOWN_PARAMS)}"
        )
    if mode in {"both", "estimate"}:
        # Interpolated profiles must be one-parameter families (`noise.check_profile`)
        _check_profiles(
            get_params_list(
                [_normalize_params_grid(grid) for grid in grids]
                if isinstance(grids, list)
                else [_normalize_params_grid(grids)]
            ),
            set(config.get("precise_mode_noise_sources", [])),
        )

    _log_header(
        config,
        model_fn,
        model_name,
        dataloader,
        dataset_cfg,
        eval_mode,
        {"mode": mode, "quantization": quantization, "N": N},
        log_path,
    )

    # Check only: is the model converted correctly? (see `accuracim.check`)
    if mode == "check":
        passed = run_model_check(dataloader, model_fn)
        section(f"Done in {fmt_duration(time.monotonic() - run_start)}")
        logger.info(f"log: {short_path(log_path)}")
        return {"check": passed}

    base_dir = _resolve_base_dir(config)
    paths_cfg = config.get("paths") or {}
    # Relative `paths` are relative to the working directory, or CIM_ROOT (absolute kept)
    calibrated_dir = os.path.join(
        base_dir, paths_cfg.get("calibrated_dir") or os.path.join("calibrated_config", model_name)
    )
    cache_dir = os.path.join(base_dir, paths_cfg.get("cache_dir") or "cache")

    section("Calibration statistics")
    input_cfg = config.get("input_stats", {})
    input_stat = get_input_statistics(
        dataloader,
        model_fn,
        N=input_cfg.get("N", 960),
        result_dir=calibrated_dir,
        override_stat=input_cfg.get("override", False),
    )

    device = get_best_gpu()
    dict_to_device(input_stat, device)

    params_grid_list = config.get("params_grid_list", [])
    normalized_grid = [_normalize_params_grid(grid) for grid in params_grid_list]
    params_list = get_params_list(normalized_grid)

    adc_cfg = config.get("adc_stats", {})
    output_percentiles = get_adc_statistics(
        dataloader,
        model_fn,
        input_stat,
        params_list,
        result_dir=calibrated_dir,
        override_stat=adc_cfg.get("override", False),
        N=adc_cfg.get("N", 512),
    )

    eval_loader_params = dataset_cfg.get("eval_params")
    if eval_loader_params:
        dataloader = dataloader_fn(**eval_loader_params)

    precise_mode_noise_sources = set(config.get("precise_mode_noise_sources", []))
    noise_layers = config.get("noise_layers")
    if noise_layers is None:
        noise_layers = model_cache.leaf_module_names(model_fn)
    probe_layers = config.get("probe_layers", ["__output__"])

    estimated = None
    simulated = None

    if mode in {"both", "estimate"}:
        estimated = run_noise_estimation(
            dataloader,
            model_fn,
            input_stat,
            output_percentiles,
            params_list,
            precise_mode_noise_sources=precise_mode_noise_sources,
            N=N,
            save_result=save_result,
            eval_mode=eval_mode,
            metric_fn=metric_fn,
            run_name=run_name,
            quantization=quantization,
            cache_dir=cache_dir,
            N_calibration=run_cfg.get(
                "N_calibration"
            ),  # optional override; default sized automatically
        )

    if mode in {"both", "simulate"}:
        section(f"Simulate · {len(params_list)} params sets")
        simulated = run_noise_simulation(
            dataloader,
            model_fn,
            input_stat,
            output_percentiles,
            noise_layers=noise_layers,
            probe_layers=probe_layers,
            params_list=params_list,
            N=N,
            save_result=save_result,
            cache_dir=cache_dir,
            eval_mode=eval_mode,
            metric_fn=metric_fn,
            run_name=run_name,
        )

    section(f"Done in {fmt_duration(time.monotonic() - run_start)}")
    logger.info(f"log: {short_path(log_path)}")

    return {
        "estimated": estimated,
        "simulated": simulated,
        "params_list": params_list,
    }
