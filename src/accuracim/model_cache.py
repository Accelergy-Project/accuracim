"""Reuse models across runs. Building and programming a model is slow for short runs.

For each model, the cache keeps:
- a pristine copy on CPU (`fresh_model` gives copies of it)
- the full-precision model, read-only (`ideal_model`)
- one programmed model (`noisy_model`), reused when only `params.FORWARD_PARAMS` change

The cache key is `params.programming_key`. An unknown param causes a rebuild, not a wrong reuse.
Probe models and models with `weight_drift` are not reused.
Set `run.reuse_models: false` to build a new model every time.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch.nn as nn

from accuracim.device import free_memory
from accuracim.graph import get_all_leaf_module_names
from accuracim.inject import replace_layers_with_noisy_version, set_forward_params
from accuracim.params import get, programming_key

logger = logging.getLogger(__name__)

_SETTINGS = {"reuse_models": True}


def configure(reuse_models: bool | None = None) -> None:
    """Turn model reuse on or off (`run.reuse_models`); off drops every cached model."""
    if reuse_models is not None:
        _SETTINGS["reuse_models"] = bool(reuse_models)
        if not reuse_models:
            clear()


@dataclass
class _Entry:
    pristine: nn.Module | None = None  # on CPU, never handed out
    device: object = None  # where `build_model` put the model
    leaf_names: list[str] | None = None
    full_precision: nn.Module | None = None
    programmed_key: tuple | None = None
    programmed: nn.Module | None = None


_ENTRIES: dict[tuple, _Entry] = {}


def _entry(build_model: callable) -> _Entry:
    # One builder can serve several models (`workloads.models`): key by the selected one too
    key = (id(build_model), build_model(return_model_name=True))
    return _ENTRIES.setdefault(key, _Entry())


def _pristine(build_model: callable) -> _Entry:
    entry = _entry(build_model)
    if entry.pristine is None:
        model = build_model()
        entry.device = next(model.parameters()).device
        entry.pristine = model.to("cpu")
    return entry


def leaf_module_names(build_model: callable) -> list[str]:
    """Return the model's leaf module names (the default noise layers)."""
    if not _SETTINGS["reuse_models"]:
        return get_all_leaf_module_names(build_model(state_dict=None, device=None))
    entry = _pristine(build_model)
    if entry.leaf_names is None:
        entry.leaf_names = get_all_leaf_module_names(entry.pristine)
    return list(entry.leaf_names)


def fresh_model(build_model: callable) -> nn.Module:
    """Return a new model, as `build_model()` would: a copy of the pristine one, on its device."""
    if not _SETTINGS["reuse_models"]:
        return build_model()
    entry = _pristine(build_model)
    return copy.deepcopy(entry.pristine).to(entry.device)


def ideal_model(build_model: callable) -> nn.Module:
    """Return the full-precision model. Read-only: callers may hook it, never change it."""
    if not _SETTINGS["reuse_models"]:
        return build_model()
    entry = _entry(build_model)
    if entry.full_precision is None:
        entry.full_precision = fresh_model(build_model)
    return entry.full_precision


def noisy_model(
    build_model: callable,
    noise_layers: list | set,
    params: dict,
    stat: dict,
    output_percentiles: dict,
) -> nn.Module:
    """Return a model programmed for `params`, reusing the last one if it is programmed the same.

    Same result as `replace_layers_with_noisy_version(build_model(), ...)`.
    """
    if not _SETTINGS["reuse_models"] or get(params, "weight_drift") is not None:
        model = fresh_model(build_model)
        return replace_layers_with_noisy_version(
            model, noise_layers, params, stat, output_percentiles
        )

    entry = _entry(build_model)
    key = (frozenset(noise_layers), id(stat), programming_key(params))
    if entry.programmed is not None and entry.programmed_key == key:
        return set_forward_params(entry.programmed, params, stat, output_percentiles)

    _drop_programmed(entry)
    model = replace_layers_with_noisy_version(
        fresh_model(build_model), noise_layers, params, stat, output_percentiles
    )
    entry.programmed, entry.programmed_key = model, key
    logger.debug("Programmed a new model (cached for runs that differ only in noise)")
    return model


def _drop_programmed(entry: _Entry) -> None:
    if entry.programmed is not None:
        entry.programmed = entry.programmed_key = None
        free_memory()


def is_cached(model: nn.Module) -> bool:
    """Whether the cache keeps `model` (so callers must not free it)."""
    return any(model is e.programmed or model is e.full_precision for e in _ENTRIES.values())


def discard(model: nn.Module) -> None:
    """Drop `model` from the cache (e.g. a forward pass failed and may have left it modified)."""
    for entry in _ENTRIES.values():
        if entry.programmed is model:
            _drop_programmed(entry)
        if entry.full_precision is model:
            entry.full_precision = None
            free_memory()


@contextmanager
def using(model: nn.Module):
    """Use `model`; if anything raises, drop it from the cache before re-raising."""
    try:
        yield model
    except BaseException:
        discard(model)
        raise


def clear() -> None:
    """Drop every cached model."""
    _ENTRIES.clear()
    free_memory()
