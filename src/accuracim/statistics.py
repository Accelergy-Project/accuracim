"""Calibration: learn all clipping ranges from data.

A probe run over a few batches records the extreme tails of each layer input and of each
pre-ADC partial sum. These tails become percentile tables, from which `quantize` gets the
clipping ranges. The tables are saved as text in `calibrated_config/` and reused by sweeps.
"""

import ast
import logging
import os
import time
from pprint import pformat

import numpy as np
import torch

from accuracim import model_cache
from accuracim.device import free_memory
from accuracim.evaluation import run_batch
from accuracim.inject import replace_layers_with_noisy_version
from accuracim.log import fmt_duration, short_path
from accuracim.params import (
    close_eq,
    extract_adc_config_params,
    params_to_str,
)
from accuracim.sampling import (
    convert_nested_defaultdict_to_dict,
    nested_defaultdict,
    torch_quantile,
)

logger = logging.getLogger(__name__)

# CDF percentiles to record (dense in the tails)
DEFAULT_PERCENTILES = (
    0,
    0.0000005,
    0.00005,
    0.0005,
    0.005,
    0.05,
    0.5,
    99.5,
    99.95,
    99.995,
    99.9995,
    99.99995,
    99.9999995,
    100.0,
)


def np_to_torch(array, device=None, dtype=None):
    """Convert to a tensor (if NumPy) and move to `device`."""
    if isinstance(array, np.ndarray):
        if dtype:
            array = torch.tensor(array, dtype=dtype)
        else:
            array = torch.tensor(array)

    if device:
        array = array.to(device)

    return array


def dict_to_device(dict, device=None):
    """Move the tensor values of `dict` to `device`, in place."""
    for key, val in dict.items():
        if isinstance(val, torch.Tensor):
            dict[key] = np_to_torch(val, device)


def save_readable(data, path):
    """Write `data` as a pretty-printed Python literal (read back by `load_readable`)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(pformat(data, width=120, sort_dicts=False))
        f.write("\n")


def load_readable(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def quantile(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """Return the `q` quantile as a float (`torch_quantile` past the tiny-size cutoff)."""
    size = tensor.numel()
    torch_quantile_size_limit = 10
    if size > torch_quantile_size_limit:
        return torch_quantile(tensor, q, interpolation="nearest").item()
    else:
        return torch.quantile(tensor, q).item()


def sample_from_list(tensor_list, num_samples):
    """Randomly sample ~`num_samples` values across tensors, proportional to size, without concatenating."""
    sampled_vals = []
    remaining = num_samples
    total_samples = sum(t.numel() for t in tensor_list)
    for t in tensor_list:
        device = t.device
        n = t.numel()
        k = min(remaining, max(1, int(num_samples * (n / total_samples))))
        if k > 0:
            idx = torch.randint(low=0, high=n, size=(k,), device=device)
            sampled_vals.append(t.view(-1).index_select(0, idx))
            remaining -= k
        if remaining <= 0:
            break
    return torch.cat(sampled_vals, dim=0)


def get_percentile_value(
    sampled_tensors: dict,
    clip_percentiles: list = DEFAULT_PERCENTILES,
    samples_per_quantile: int | float = 100,
    sample_fraction=1.00,
):
    """Return {percentile: value} from sampled tails.

    `low`/`high` hold the lowest/highest `sample_fraction` of values, so each
    percentile is rescaled into its tail.
    """
    chunk_size = 1_000_000
    lows, highs = sampled_tensors["low"], sampled_tensors["high"]
    output_values = {}

    for p in clip_percentiles:
        if close_eq(p, 0):
            output_values[p] = min(t.min().item() for t in lows)
            continue
        elif close_eq(p, 100):
            output_values[p] = max(t.max().item() for t in highs)
            continue

        if p <= 50:
            p_eff = p / 100.0 / sample_fraction
            target = lows
        else:
            p_eff = 1.0 - ((100 - p) / 100.0 / sample_fraction)
            target = highs

        target_size = sum(t.numel() for t in target)
        required_samples = max(chunk_size, int(samples_per_quantile / p_eff / (1 - p_eff)))
        required_samples = min(required_samples, target_size)
        sampled_target = sample_from_list(target, required_samples)

        # Move only the sample to GPU, if there is one
        sampled_target = sampled_target.to(
            "cuda" if torch.cuda.is_available() else "cpu", non_blocking=True
        )

        output_values[p] = quantile(sampled_target, p_eff)

    return output_values


def get_adc_statistics(
    dataloader,
    build_model,
    input_percentiles,
    params_list: list[dict] = [],
    result_dir: str = ".",
    N=500,
    override_stat: bool = False,
    clip_percentiles: list = DEFAULT_PERCENTILES,
    sample_fraction=0.02,
):
    """Return {ADC-config key: pre-ADC percentile tables}, loading saved ones or probing.

    Tables are keyed [layer][input slice scale][weight slice scale]; None pools over that axis.
    """
    layer_names = model_cache.leaf_module_names(build_model)

    percentiles = {}
    num_loaded = num_probed = 0

    for params in params_list:
        extracted_params = extract_adc_config_params(params)

        if extracted_params is None:  # no input/weight bits: no ADC
            continue

        params_key = params_to_str(**extracted_params)

        output_percentiles = nested_defaultdict()
        store_dict = nested_defaultdict()

        percentiles_path = f"{result_dir}/{params_key}.txt"

        # Already done in this sweep
        if params_key in percentiles:
            continue

        # Saved earlier
        if os.path.exists(percentiles_path) and not override_stat:
            percentiles[params_key] = load_readable(percentiles_path)
            logger.debug(f"Loaded ADC ranges from {percentiles_path}")
            num_loaded += 1
            continue

        logger.info(f"ADC ranges · probing {N} samples for a new ADC configuration")
        logger.debug(f"ADC configuration: {params_key}")
        probe_start = time.monotonic()
        extracted_params |= {"sample_fraction": sample_fraction}
        # Probe pre-ADC outputs
        noisy_model = replace_layers_with_noisy_version(
            model_cache.fresh_model(build_model),
            noise_layers=layer_names,
            params=extracted_params,
            input_percentiles=input_percentiles,
            output_percentiles=None,
            probe_inputs=False,
            probe_outputs=True,
            store_dict=store_dict,
            sample_fraction=sample_fraction,
        )
        noisy_model.eval()
        total_samples = 0
        with torch.no_grad():
            for batch_data in dataloader:
                result_dict = run_batch(noisy_model, batch_data, num_processed=total_samples, N=N)
                total_samples += result_dict["batch_size"]
                if total_samples >= N:
                    break

        # Merge layer probes (module.outputs) with matmul probes (store_dict)
        for layer_name, module in noisy_model.named_modules():
            if not hasattr(module, "outputs"):
                continue
            outputs_store_layer = module.outputs
            store_dict[layer_name] = outputs_store_layer
        # Free this model before the next configuration builds its own (large models)
        noisy_model = result_dict = None
        free_memory()

        for layer_name, outputs_store_layer in store_dict.items():
            sliced_outputs = nested_defaultdict()

            iss_keys = list(outputs_store_layer.keys())
            wss_keys = list(outputs_store_layer[iss_keys[0]].keys())
            idx_keys = list(outputs_store_layer[iss_keys[0]][wss_keys[0]].keys())

            for i_s_s in iss_keys + [None]:
                for w_s_s in wss_keys + [None]:
                    sliced_outputs[i_s_s][w_s_s] = {"low": [], "high": []}

            for i_s_s in iss_keys:
                for w_s_s in wss_keys:
                    for idx in idx_keys:
                        outputs = outputs_store_layer[i_s_s][w_s_s][idx]
                        if not outputs:
                            continue

                        for k, v in outputs.items():
                            # Pool into per-axis and layer-wide (None) tables too
                            sliced_outputs[None][None][k].extend(v)
                            sliced_outputs[i_s_s][None][k].extend(v)
                            sliced_outputs[None][w_s_s][k].extend(v)
                            sliced_outputs[i_s_s][w_s_s][k].extend(v)

            for i_s_s in iss_keys + [None]:
                for w_s_s in wss_keys + [None]:
                    outputs = sliced_outputs[i_s_s][w_s_s]
                    if not outputs:
                        continue
                    output_percentiles[layer_name][i_s_s][w_s_s] = get_percentile_value(
                        sliced_outputs[i_s_s][w_s_s],
                        clip_percentiles,
                        sample_fraction=sample_fraction,
                    )

        output_percentiles = convert_nested_defaultdict_to_dict(output_percentiles)

        os.makedirs(result_dir, exist_ok=True)
        save_readable(output_percentiles, percentiles_path)
        logger.info(f"ADC ranges · saved after {fmt_duration(time.monotonic() - probe_start)}")
        logger.debug(f"Saved ADC ranges to {percentiles_path}")
        num_probed += 1
        percentiles[params_key] = output_percentiles

    if num_loaded:
        logger.info(f"ADC ranges · {num_loaded} loaded from {short_path(result_dir)}")
    elif not num_probed:
        logger.info("ADC ranges · none needed (no input/weight bits set)")
    return percentiles


def get_input_statistics(
    dataloader,
    build_model,
    N=500,
    stat_path=None,
    result_dir=".",
    override_stat: bool = False,
    clip_percentiles: list = DEFAULT_PERCENTILES,
    sample_fraction=0.02,
    verbose=True,
):
    """Return per-layer input percentile tables, loading `stat_path` if it exists."""
    os.makedirs(result_dir, exist_ok=True)

    if stat_path is None:
        stat_path = f"{result_dir}/input_stat.txt"

    if os.path.exists(stat_path) and not override_stat:
        input_stats = load_readable(stat_path)
        logger.info(f"input ranges · loaded {short_path(stat_path)}")
        return input_stats

    logger.info(f"input ranges · probing {N} samples")
    probe_start = time.monotonic()
    model = model_cache.fresh_model(build_model)
    model.eval()

    store_dict = nested_defaultdict()

    replace_layers_with_noisy_version(
        model,
        probe_inputs=True,
        store_dict=store_dict,
        sample_fraction=sample_fraction,
        verbose=True,
    )

    total_samples = 0
    with torch.no_grad():
        for batch_data in dataloader:
            result_dict = run_batch(model, batch_data, num_processed=total_samples, N=N)
            total_samples += result_dict["batch_size"]
            if total_samples >= N:
                break

    input_stat = {}
    if verbose:
        logger.debug("Probed inputs:\n" + pformat(list(store_dict.keys())))

    for name, raw_data in store_dict.items():
        if "low" in raw_data and "high" in raw_data:
            # Layer inputs
            percentile_dict = get_percentile_value(
                raw_data, clip_percentiles=clip_percentiles, sample_fraction=sample_fraction
            )
            input_stat[name] = percentile_dict

        else:
            input_stat[name] = {}
            # LLM matmuls: one table per operand ('input', 'weight')
            for k, v in raw_data.items():
                percentile_dict = get_percentile_value(
                    v, clip_percentiles=clip_percentiles, sample_fraction=sample_fraction
                )
                input_stat[name][k] = percentile_dict

    save_readable(input_stat, stat_path)
    logger.info(
        f"input ranges · saved {short_path(stat_path)} "
        f"after {fmt_duration(time.monotonic() - probe_start)}"
    )

    return input_stat
