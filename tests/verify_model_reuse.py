"""Developer check: reused models (`accuracim.model_cache`) must equal new models.

The check runs the important cases: baseline and noise runs in both quantization modes,
reprogramming, slice storage types, drift, and a return to an earlier programming.
For each case, it compares a new model with the cached model on all programmed tensors
and on a seeded forward pass.

Run the config once first, so that its calibrated_config exists:
    python tests/verify_model_reuse.py --config configs/2_scaling-test/r18_dep.yaml
    python tests/verify_model_reuse.py --config configs/2_scaling-test/gpt-s_dep.yaml --samples 2

The check exits with a non-zero code if a value is different.
"""

import argparse
import copy
import os
import sys
import time

import torch

from accuracim import model_cache
from accuracim.config import REGISTRY, load_config, project_root
from accuracim.device import free_memory
from accuracim.evaluation import run_batch
from accuracim.inject import replace_layers_with_noisy_version
from accuracim.layers import NoisyConv2d
from accuracim.noise import (
    ConductanceDrift,
    ProportionalNoise,
    StateIndependentNoise,
    restore_noise_seed,
    set_noise_seed,
)
from accuracim.params import _exact, get, get_output_percentiles_per_params
from accuracim.pipeline import _normalize_params_grid, _register_defaults, get_params_list
from accuracim.sources import (
    _quant_params_from_params,
    enumerate_noise_sources,
    source_params,
    tag_baseline,
)
from accuracim.statistics import dict_to_device, get_adc_statistics, get_input_statistics

SEED = 1234
_LAYER_ATTRS = (
    "name", "probe_outputs", "weight_slice_lsb_to_msb", "weight_slice_scales",
    "weight_scale", "weight_clip_range", "input_scale", "input_zp", "input_clip_range",
)  # fmt: skip


def same(a, b) -> bool:
    """Exact equality for tensors, numbers, and nested tuples/lists/dicts of them."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return same(float(a), float(b))
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return _exact(a) == _exact(b)


def compare_state(fresh, reused) -> list[str]:
    """Return what differs between two programmed models (empty if identical)."""
    problems = []
    fm, rm = dict(fresh.named_modules()), dict(reused.named_modules())
    if list(fm) != list(rm):
        return ["module names differ"]
    for name, f in fm.items():
        r = rm[name]
        if type(f) is not type(r):
            problems.append(f"{name}: type {type(f).__name__} vs {type(r).__name__}")
            continue
        if len(f._forward_hooks) != len(r._forward_hooks):
            problems.append(f"{name}: {len(f._forward_hooks)} vs {len(r._forward_hooks)} hooks")
        if not isinstance(f, NoisyConv2d):
            continue
        for attr in _LAYER_ATTRS:
            if not same(getattr(f, attr), getattr(r, attr)):
                problems.append(f"{name}.{attr} differs")
        fs, rs = f.sliced_weights, r.sliced_weights
        if (fs is None) != (rs is None) or (fs is not None and not same(list(fs), list(rs))):
            problems.append(f"{name}.sliced_weights differ")
        if f.params.keys() != r.params.keys() or not all(
            _exact(f.params[k]) == _exact(r.params[k]) for k in f.params
        ):
            problems.append(f"{name}.params differ")
        if not same(f.output_percentiles, r.output_percentiles):
            problems.append(f"{name}.output_percentiles differ")
    fsd, rsd = fresh.state_dict(), reused.state_dict()
    if fsd.keys() != rsd.keys():
        problems.append("state_dict keys differ")
    else:
        problems += [f"tensor {k} differs" for k in fsd if not same(fsd[k], rsd[k])]
    if hasattr(fresh, "_noisy_tensor_hooks") != hasattr(reused, "_noisy_tensor_hooks"):
        problems.append("NoisyTensor hooks present in only one model")
    return problems


def seeded_logits(model, batch, samples):
    state = set_noise_seed(SEED)
    try:
        with torch.no_grad():
            # A copy: noisy layers clamp their input in place
            batch = copy.deepcopy(batch)
            logits = run_batch(model.eval(), batch, num_processed=0, N=samples)["logits"]
    finally:
        restore_noise_seed(state)
    return logits["logits"] if isinstance(logits, dict) else logits


def cases(params_list):
    """(name, params) to check, in order: reuse, reprogramming, and coming back."""
    p = params_list[0]
    other = params_list[1] if len(params_list) > 1 else p
    base = _quant_params_from_params(p)
    physical = [
        ns for ns in enumerate_noise_sources([p], set(), "noise_source")[0] if "physical" in ns
    ]
    out = [
        ("precise baseline", base),
        ("precise, configured noise", p),
        ("precise, another noise (reuse)", other),
        ("precise, weight + input noise (reuse)", {
            **base,
            "weight_physical_noise": ProportionalNoise(0.02),
            "input_physical_noise": StateIndependentNoise(0.001),
        }),
    ]
    if physical:
        ns = physical[0]
        run = tag_baseline(source_params(p, ns, "noise_source"), "noise_source")
        selective = {
            **tag_baseline(source_params(p, f"{ns}_selective_quantize", "noise_source"),
                           "noise_source"),
            "baseline": "quantized",
        }
        out += [("noise_source, physical source", run), ("noise_source, selective", selective)]
    noisy = {**p, "weight_physical_noise": ProportionalNoise(0.02)}
    out += [
        ("weight clip 99.9 (reprogram)", {**noisy, "weight_clip_percentile": 99.9}),
        ("offset encoding (reprogram)", {**noisy, "differential_weight": False}),
    ]
    bits = get(p, "weight_bits")
    if bits is not None:
        out += [
            ("one full-width slice: int16", {**noisy, "weight_slice_lsb_to_msb": [bits]}),
            ("unsliced: float slice", {**noisy, "weight_slice_lsb_to_msb": None}),
            ("drift (never cached)", {**noisy, "weight_drift": ConductanceDrift(0.05)}),
        ]
    out.append(("back to configured noise (reprogram)", p))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=4, help="samples in the forward check")
    args = parser.parse_args()

    def say(msg):
        print(msg, flush=True)

    sys.path.insert(0, project_root())
    _register_defaults()
    config = load_config(args.config)
    model_cfg = config["model"]
    build_model = REGISTRY.models.resolve(model_cfg.get("builder") or model_cfg.get("path"))
    if model_cfg.get("name"):
        build_model(model_name=model_cfg["name"], return_model_name=True)
    model_name = build_model(return_model_name=True)

    dataset_cfg = config["dataset"]
    loader_fn = REGISTRY.datasets.resolve(dataset_cfg.get("loader") or dataset_cfg.get("name"))
    loader_params = dict(dataset_cfg.get("eval_params") or dataset_cfg.get("params", {}))
    loader_params["batch_size"] = args.samples
    batch = next(iter(loader_fn(**loader_params)))
    say(f"setup · {model_name} · one batch of {args.samples} loaded")

    calibrated_dir = os.path.join(project_root(), "calibrated_config", model_name)
    if not os.path.exists(os.path.join(calibrated_dir, "input_stat.txt")):
        sys.exit(f"Run {args.config} once first: {calibrated_dir}/input_stat.txt is missing")
    grid = [_normalize_params_grid(g) for g in config.get("params_grid_list", [])]
    params_list = get_params_list(grid)

    # Saved tables only: variations without one simply run without ADC effects
    stat = get_input_statistics(None, build_model, result_dir=calibrated_dir)
    dict_to_device(stat, build_model(return_device=True))
    output_percentiles = get_adc_statistics(
        None, build_model, stat, params_list[:1], result_dir=calibrated_dir
    )
    noise_layers = model_cache.leaf_module_names(build_model)
    say("setup · statistics loaded")

    failures = 0
    previous = None
    checks = cases(params_list)
    for k, (name, run) in enumerate(checks):
        start = time.monotonic()
        op = get_output_percentiles_per_params(output_percentiles, run)
        fresh = replace_layers_with_noisy_version(
            build_model(), noise_layers, copy.copy(run), stat, op
        )
        model = model_cache.noisy_model(build_model, noise_layers, copy.copy(run), stat, op)
        how = ("reused" if model is previous
               else "programmed" if model_cache.is_cached(model) else "uncached")
        problems = compare_state(fresh, model)
        a, b = seeded_logits(fresh, batch, args.samples), seeded_logits(model, batch, args.samples)
        if not torch.equal(a, b):
            problems.append(f"outputs differ (max |diff| {(a - b).abs().max().item():.3g})")
        say(f"[{k + 1}/{len(checks)}] {'ok  ' if not problems else 'FAIL'} {name} · {how} · "
            f"{time.monotonic() - start:.1f}s")
        for p in problems[:10]:
            say(f"      {p}")
        failures += bool(problems)
        previous = model if model_cache.is_cached(model) else None
        del fresh, model
        free_memory()

    fresh, cached_fp = build_model(), model_cache.ideal_model(build_model)
    ok = not compare_state(fresh, cached_fp) and torch.equal(
        seeded_logits(fresh, batch, args.samples), seeded_logits(cached_fp, batch, args.samples)
    )
    say(f"[fp] {'ok  ' if ok else 'FAIL'} full-precision model")
    failures += not ok
    del fresh, cached_fp

    model = previous
    try:
        with model_cache.using(model):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    ok = not model_cache.is_cached(model)
    say(f"[evict] {'ok  ' if ok else 'FAIL'} a failed pass drops the model from the cache")
    failures += not ok

    say(f"\n{len(checks) + 2} checks; {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
