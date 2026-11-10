"""AccuraCiM: fast and precise estimation of Compute-in-Memory accuracy loss.

Run an experiment::

    python -m accuracim --config configs/2_scaling-test/r18_dep.yaml

This package is the engine. Models and datasets are in the `workloads/` folder of the
repository. Configs name them by "module:func" path.

The engine, from the bottom up::

    config      config loading and the model/dataset/metric registry
    params      the params dict: defaults, canonical string and cache keys
    sampling    sub-sampling of large activations, and the sample stores
    quantize    clipping ranges, scales, bit slicing, quantization/clipping error
    graph       find, rewrite and hook modules in a model
    device      select a GPU, release memory
    dataloader  deterministic batches, so that noisy and ideal runs align
    evaluation  run a batch; accuracy, cross-entropy and flip metrics
    noise       noise profiles (ProportionalNoise, StateIndependentNoise, ...)

    layers      NoisyConv2d / NoisyLinear / NoisyConv1D
    tensors     NoisyTensor: intercepts `@` for attention matmuls
    inject      turn a model into a noisy model

    statistics  calibration: input and pre-ADC percentiles
    sources     noise sources
    simulate    ground truth: measure noise with the noisy model
    estimate    fast path: calibrate a few anchors, interpolate the rest
    pipeline    runs all of the above from a config
"""

from accuracim.config import REGISTRY, load_config, project_root
from accuracim.noise import (
    ADCGain,
    ADCNonlinearity,
    ADCOffset,
    ConductanceDrift,
    GaussianNoise,
    Noise,
    PowerLawDrift,
    ProportionalNoise,
    ShotNoise,
    StateIndependentNoise,
    TelegraphNoise,
    check_profile,
)
from accuracim.pipeline import get_params_list, run_experiment

__all__ = [
    "project_root",
    "REGISTRY",
    "load_config",
    "run_experiment",
    "get_params_list",
    "Noise",
    "check_profile",
    "ProportionalNoise",
    "StateIndependentNoise",
    "GaussianNoise",
    "ShotNoise",
    "TelegraphNoise",
    "ConductanceDrift",
    "PowerLawDrift",
    "ADCGain",
    "ADCOffset",
    "ADCNonlinearity",
]
