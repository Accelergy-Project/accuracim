# AccuraCiM

**Fast and precise estimation of Compute-in-Memory (CiM) accuracy loss.**

[![Paper](https://img.shields.io/badge/ICCAD'26-paper-b31b1b.svg)](https://doi.org/10.1145/3831252.3834215)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Estimating the DNN accuracy loss of an analog CiM accelerator is slow: every
hardware configuration needs a full-dataset simulation (e.g., 50,000 ImageNet
samples). AccuraCiM uses small-signal linearity to run full-dataset simulations
on only a few configurations and predict the rest of the design space.

- **Scaling Law.** DNN output error scales linearly with noise magnitude, so we
  measure the error for a noise source once and scale it to other magnitudes.
- **Additivity Law.** Total output error is the sum of errors from individual
  noise sources, so we measure each source on its own and sum the errors for
  any combination.

In the reasonable-loss regime (< 3% loss), AccuraCiM is up to **188× faster**
than exhaustive simulation while modeling accuracy loss within **0.4%** of it.

## Features

| | Supported |
|---|---|
| Datasets | ImageNet, Imagenette, WikiText-2 |
| Image models | ResNet-18/34/50/101/152, VGG16-BN, MobileNetV2, EfficientNet-B0, ViT-B16 |
| Language models | GPT-2 Small/Medium/Large/XL, DistilGPT2, SmolLM2-135M/360M, Qwen2.5-0.5B, Pythia-160M |
| Noise | state-independent, state-proportional, Gaussian (per element), shot, random telegraph, ADC gain / offset / nonlinearity, conductance drift (constant, power-law) |
| Quantization | input, weight and ADC quantization and clipping |
| CiM architecture | rows per ADC (row tiling), input and weight bit slicing, differential or offset encoding, ADC ranges per layer or per slice, analog or digital attention |
| Runs | fast estimate, full simulation, model check. Sweeps are cached, resumable and split across GPUs. |

You can add your own dataset, model, metric or noise profile in `workloads/`.
You do not need to change the engine. You can apply several noise profiles to one tensor.

## Quick start

```sh
git clone https://github.com/Accelergy-Project/accuracim.git
cd accuracim
pip install -e .
```

Try it first with DistilGPT2 on WikiText-2. The dataset downloads itself:

```sh
python -m accuracim --config configs/quick_examples/distilgpt2_wikitext.yaml                   # estimate (fast)
python -m accuracim --config configs/quick_examples/distilgpt2_wikitext.yaml --mode simulate   # full simulation, to compare
```

For an image model, `configs/quick_examples/imagenette_r34.yaml` runs ResNet-34 on
Imagenette (10 ImageNet classes, 3,925 images). Imagenette also downloads itself.

ImageNet cannot be redistributed. To use it for the paper's experiments:
1. Download `ILSVRC2012_img_val.tar` from [image-net.org](https://image-net.org).
2. Extract it.
3. Point to the folder (the val labels are already in `workloads/data/`):

```sh
export IMAGENET_VAL_DIR=~/imagenet/val
```

Run an experiment from the repository root:

```sh
python -m accuracim --config configs/2_scaling-test/r18_dep.yaml                   # estimate (fast)
python -m accuracim --config configs/2_scaling-test/r18_dep.yaml --mode simulate   # full-dataset simulation, to compare
```

Results are written as `.xlsx` sweep summaries to
`results/<model>/estimated/` and `results/<model>/measured/`.

## Repository layout

```
accuracim/
├── configs/          experiment configs                  ← edit, or add your own
├── workloads/        models and datasets                 ← edit, or add your own
└── src/accuracim/    the engine: noise injection, simulation, estimation
```

You can change everything outside `src/`. Configs point to code in `workloads/` by
`module:func` path. To add a model or dataset, add it there. You do not need to change the engine.

## Example experiments

| Config | What it runs |
|---|---|
| [`configs/quick_examples/`](configs/quick_examples/) | small runs that download their own data: ResNet-34 on Imagenette, DistilGPT2 on WikiText-2 |
| [`configs/0_sram-cim-dse/`](configs/0_sram-cim-dse/) | design space exploration of an SRAM-CiM accelerator for ResNet-18 |
| [`configs/1_additivity-test/`](configs/1_additivity-test/) | memory-cell, DAC and ADC noise together, over a wide noise range |
| [`configs/2_scaling-test/`](configs/2_scaling-test/) | memory-cell noise (state-dependent / independent) at several magnitudes |
| [`configs/3_drift-test/`](configs/3_drift-test/) | conductance drift (experimental, simulate only) |
| [`configs/model_examples/`](configs/model_examples/) | ADC noise sweeps for the other models in `workloads/models.py` |

Paper models: `r18`, `r50` (ResNet-18/50) and `gpt-s`, `gpt-m`, `gpt-l` (GPT-2
Small/Medium/Large). See [`configs/README.md`](configs/README.md).

## Usage

### Modes

| `--mode` | What it does |
|---|---|
| `estimate` | full-dataset simulation at a few anchor points; Scaling and Additivity laws for the rest |
| `simulate` | full-dataset simulation at every point (ground truth) |
| `both` | `estimate`, then `simulate` |
| `check` | one batch: is the model converted correctly? Run it before trusting a new model ([details](workloads/README.md#check-a-new-model)) |

The default comes from `run.mode` in the config.

### Configs

Configs are YAML, JSON, or Python (a module defining `CONFIG`). They select the
model, dataset, metric and the grid of hardware parameters:

```yaml
model:
  builder: workloads.models:build_model
  name: ResNet18
dataset:
  loader: workloads.dataset:get_imagenet_dataloader
params_grid_list:
  - weight_bits: [8]
    weight_slice_lsb_to_msb: [2]
    weight_physical_noise:
      - {type: ProportionalNoise, args: [0.01]}
      - {type: ProportionalNoise, args: [0.12]}
run:
  N: 50000
  mode: estimate
```

You can also write a noise list as a sweep. It gives one entry per value, and includes `stop`:

```yaml
    weight_physical_noise:
      - {type: ProportionalNoise, sweep: {start: 0.01, stop: 0.12, step: 0.01}}
    adc_physical_noise:
      - {type: StateIndependentNoise, sweep: {start: 1e-8, stop: 1e-3, num: 16, scale: log}}
```

### Your own model, dataset, metric or noise

Write it in `workloads/` and point the config at it:

| To add | Write | Config key |
|---|---|---|
| a model | a builder, like `workloads/models.py:build_model` | `model.builder` |
| a dataset | a loader wrapped in `get_deterministic_dataloader`, like `workloads/dataset.py` | `dataset.loader` |
| a metric | a function with `update_performance_metrics`'s signature | `evaluation.metric_fn` |
| a noise profile | a `Noise` subclass with `__call__` and `get_scaling_factor` | `type:` in `*_physical_noise` |

[`workloads/README.md`](workloads/README.md) has a short template for each.

### Parallel sweeps

Split one sweep across processes or GPUs:

```sh
python -m accuracim --config configs/0_sram-cim-dse/dse_sweep.yaml --group-index 3 --num-groups 8 --log group-3.log
```

## Details

<details>
<summary><b>Where files are read and written</b></summary>

Run from the repository root. Configs, `workloads/` and all outputs are relative to the
**working directory**, not to the install location of the package.
To keep code and data apart, set the output location with `CIM_ROOT`:

```sh
CIM_ROOT=~/experiments/run-7 python -m accuracim --config configs/2_scaling-test/r18_dep.yaml
```

| Path | Contents |
|---|---|
| `configs/` | experiment configs |
| `workloads/` | models and datasets, imported by the configs |
| `calibrated_config/<model>/` | pretrained weights and calibrated percentile tables |
| `cache/` | per-batch ideal outputs, error tensors and noise, keyed by params |
| `logs/<model>/` | one log per run: the console output plus debug details |
| `results/<model>/<measured\|estimated>/` | `.xlsx` sweep summaries: one sheet of results, one of run metadata |

`measured` is the ground truth from `simulate`. `estimated` is the prediction from
`estimate`. A result filename starts with a timestamp, so a listing sorts by time.
Then it has the config `name` (default: the filename stem) and the sample count:

```
results/ResNet18/measured/2026-08-31_18-17-39_r18_N512.xlsx
```

`--group-index` appends `-g<i>`.

Each run writes a log to `logs/<model>/<timestamp>_<name>.log`. The log has the console
output and debug details (full params, cache folders).
- `--verbose` shows the debug details on the console too.
- `--log <file>` sends all output to that file, including the output of other libraries.

With `cache/`, an interrupted sweep can resume and a repeated params set takes no time.
You can delete the cache safely. The next run only takes longer.
Two checkouts, or two `CIM_ROOT`s, have separate caches and results.

</details>

<details>
<summary><b>Quantization accounting</b></summary>

Two ways to account for quantization error, set with `run.quantization` or
`--quantization`:

| | baseline | quantization is | cost |
|---|---|---|---|
| `precise` *(default)* | quantized, noiseless | absorbed into the baseline | one ideal run + anchors **per quantization setting** |
| `noise_source` | full precision | an additive source with strength `1/4^bits` | one baseline for the whole sweep |

`precise` is exact. `noise_source` is faster. It treats quantization error as one more
independent noise source, so one full-precision baseline is enough for the whole sweep.
To keep this valid:
- `*_clipping` sources are always measured exactly. The `1/4^bits` model cannot describe clipping.
- Physical noise is measured twice, and the quantization error is subtracted.
- Cached error tensors are scaled by a suppression factor (at most 1), because quantization
  clamps the noise on top of it.

To measure a stage exactly in `noise_source` mode, add it to `precise_mode_noise_sources`
(e.g. `adc_quantize`). The two modes use separate caches (`_Qnoise` suffix).

</details>

<details>
<summary><b>Metrics</b></summary>

`evaluation.mode` selects the metric. `classification` and `next_token` report
`top1` / `top5`. `perplexity` reports `cross_entropy` / `perplexity`. Without it, the
`eval_mode` of the dataloader is used (ImageNet: `classification`, WikiText-2: `next_token`).

Each run also reports **`top1_flip`**: the fraction of predictions that the noise changed
(noisy model vs. ideal model).

At low noise, `top1_flip` is much more sensitive than accuracy. A flip needs any two logits
to swap. An accuracy change needs the *correct* logit to be one of them. Thus `top1_flip`
shows a difference even on small calibration runs (128 samples), where accuracy stays flat.

`top1_flip` and `top1` have the same denominator, so you can compare them directly.
`estimated_top1_flip` uses only the top-k indices that the estimator keeps. At large noise
it is slightly low, as `estimated_top1` is.

</details>

<details>
<summary><b>Code layout</b></summary>

The engine in `src/accuracim/`, from the bottom up. Each module depends only on the
modules above it.

```
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
```

A run goes through these steps: `statistics` learns the clipping ranges → `inject` builds
the noisy model → `simulate` measures it against a cached ideal run → `estimate` uses those
anchors to predict the whole sweep.

</details>

## Citation

If you use AccuraCiM in your research, please cite:

```bibtex
@inproceedings{bae2026accuracim,
  title     = {Small-Noise is All You Need: Fast and Precise Estimation of Compute-in-Memory Accuracy Loss},
  author    = {Bae, Soojung and Andrulis, Tanner and Sze, Vivienne and Emer, Joel S.},
  booktitle = {Proceedings of the IEEE/ACM International Conference on Computer-Aided Design (ICCAD '26)},
  year      = {2026},
  doi       = {10.1145/3831252.3834215}
}
```

## License

The code is released under the [MIT License](LICENSE). The paper is published
under CC BY 4.0.
