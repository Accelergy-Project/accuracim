# Workloads

The models and datasets that AccuraCiM runs on. You can edit or copy everything here.
Configs point to this code by `module:func` path. The engine in `src/accuracim/` does not
import it. Thus, code that you add here runs without changes to the engine.

| File | Contents |
|---|---|
| [`models.py`](models.py) | torchvision CNNs, Hugging Face language models and ViT (list below) |
| [`dataset.py`](dataset.py) | ImageNet val, Imagenette (downloads itself) and WikiText-2 dataloaders |
| `data/` | ImageNet val labels |

```yaml
model:
  builder: workloads.models:build_model
  name: ResNet18
dataset:
  loader: workloads.dataset:get_imagenet_dataloader
  params: {batch_size: 24, num_workers: 2, deterministic_dataloader: true}
```

## Models

| Dataset | Models | Status |
|---|---|---|
| ImageNet | ResNet-18, ResNet-50 | used in the paper |
| ImageNet | ResNet-34/101/152, VGG16-BN, MobileNetV2, EfficientNet-B0, ViT-B16 (analog and digital attention) | `--mode check` passes |
| WikiText-2 | GPT-2 Small/Medium/Large | used in the paper |
| WikiText-2 | GPT2-XL, DistilGPT2, SmolLM2-135M/360M, Qwen2.5-0.5B, Pythia-160M | `--mode check` passes |

For language models other than GPT-2, set the tokenizer to the model's `hf_id`
(`dataset.params.tokenizer: HuggingFaceTB/SmolLM2-135M`). Models with a large
vocabulary (Qwen2.5: 151k) need a smaller batch size: logits are batch × tokens × vocab.

### Check a new model

```sh
python -m accuracim --config my_config.yaml --mode check
```

Run this before you trust the results of a new model. The check uses one batch, and
saves only the log. It verifies that the model is converted correctly:

| Line | Passes when |
|---|---|
| `transparent` | with every impairment off, outputs match the original model (≤ 1e-3) |
| `attention` | (transformers) every attention module's Q·Kᵀ and A·V run on the simulated arrays |
| `bypassed the simulation` | no matmul-like op (`bmm`, `einsum`, flash kernels, …) skips the arrays |
| `array path` | bits, slices and row tiling run (catches shape bugs, e.g. grouped convolutions) |
| `noise reaches the output` | small input and weight noise changes the output |

`digital weights` lists the matrix weights that stay digital (e.g. a packed
`MultiheadAttention` projection). This is not a failure, but these weights get no noise.

### Attention: analog or digital

Transformers simulate their attention matmuls by default. To simulate only the
Linear layers and keep the model's own (faster) attention kernel:

```yaml
model:
  builder: workloads.models:build_model
  name: SmolLM2-135M
  analog_attention: false
  attn_implementation: flash_attention_2   # optional, used only with digital attention
```

Simulated attention needs a kernel that AccuraCiM can intercept (`sdpa` or `eager`).
A requested flash kernel changes to `sdpa`. With `analog_attention: false`, the model name
gets a `-digital-attn` suffix. Its calibration, caches and results are thus kept apart.
The run header and `--mode check` both show which attention ran.

## Add a model

For a torchvision or Hugging Face model, add one entry to `MODELS` in
`models.py`:

```python
MODELS = {
    ...
    "WideResNet50": {"mode": "cnn", "torchvision": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V1)},
    "OPT-125M":     {"mode": "transformer", "hf_id": "facebook/opt-125m"},
}
```

Import `wide_resnet50_2` and `Wide_ResNet50_2_Weights` from `torchvision.models` at the top.
For Hugging Face image classifiers, add `"hf_task": "image_classification"`.
Then run `--mode check`.

`mode` is `"cnn"` or `"transformer"`. A transformer also simulates its attention matmuls,
unless the config sets `analog_attention: false`.
Weights download on first use to `calibrated_config/<name>/weights.pth`.

For other models, write your own builder. It returns a model. When called with a
`return_*` argument, it returns that value instead:

```python
def build_my_model(state_dict=None, device="cuda:0", **kwargs):
    if kwargs.get("return_model_name"):
        return "MyNet"                     # folder name under calibrated_config/, cache/, results/
    if kwargs.get("return_mode"):
        return "cnn"                       # or "transformer"
    if kwargs.get("return_device"):
        return device

    model = MyNet()
    model.load_state_dict(state_dict or torch.load("my_weights.pth"))
    model.mode = "cnn"                     # same as return_mode
    return model.to(device).eval()
```

If the config sets `model.name`, the builder gets it as `model_name=`.
Thus one builder can serve several models.

## Add a dataset

A loader is a callable that returns a DataLoader. `dataset.params` in the config are its
keyword arguments. Wrap the dataset with `get_deterministic_dataloader`. Simulation compares
a noisy run with a cached ideal run, batch by batch, so the two runs must see the same batches.
Then set the metric that fits the data:

```python
from accuracim.dataloader import get_deterministic_dataloader

def get_my_loader(batch_size=32, num_workers=2, deterministic_dataloader=True):
    dataset = MyDataset()                  # (input, label) items, or dicts with input_ids / labels
    loader = get_deterministic_dataloader(dataset, batch_size, seed=42, num_workers=num_workers)
    loader.eval_mode = "classification"    # or "next_token" / "perplexity"
    return loader
```

If you already ran a model on another dataset, give the new dataset its own
`paths.cache_dir` and `paths.calibrated_dir`. Caches are keyed by model and params,
not by dataset (see `configs/quick_examples/imagenette_r34.yaml`).

The batch type selects how the model is called. An `(inputs, targets)` tuple runs
`model(inputs)`. A dict runs `model(**batch)`, and its `labels` are the targets.

## Choose or add a metric

`evaluation.mode` selects the built-in metric. Without it, the `eval_mode` of the
dataloader is used, and then the mode of the model.

| `evaluation.mode` | Reports | Logits |
|---|---|---|
| `classification` | `top1`, `top5` | (batch, classes) |
| `next_token` | next-token `top1`, `top5` | (batch, tokens, vocab) |
| `perplexity` | `cross_entropy`, `perplexity` | (batch, tokens, vocab) |

A custom metric has the signature of `accuracim.evaluation:update_performance_metrics`.
It adds raw counts with `accumulate`, including `count` (the denominator).
The engine calculates the averages at the end:

```python
from accuracim.evaluation import accumulate, predicted_class

def top1_error(logits, targets, performance_dict, mode="classification", param_idx=None,
               return_counts=True, top_indices=None):
    # Classification example. Under `estimate`, logits hold only the top-k candidates.
    # predicted_class maps them back to class ids through top_indices.
    pred = predicted_class(logits, top_indices)
    wrong = (pred != targets).sum().item()
    accumulate(performance_dict, {"top1_error": wrong, "count": targets.numel()}, param_idx)
```

```yaml
evaluation:
  metric_fn: workloads.my_metrics:top1_error
```

## Noise profiles

Built-in profiles (`accuracim.noise`). x is the perturbed value. FS = high − low is its range.

| Profile | Error | Strength (shape) | Typical key |
|---|---|---|---|
| `ProportionalNoise` | N(0, (α·x)²) | α | `weight_physical_noise` |
| `StateIndependentNoise` | N(0, (α·FS)²) | α | any |
| `GaussianNoise` | N(0, std²), std scalar or per element | std (pattern) | any |
| `ShotNoise` | N(0, α²·\|x\|·FS) | α | any |
| `TelegraphNoise` | −α·x·b, b ~ Bernoulli(p) per read | α (p) | `weight_physical_noise` |
| `ADCGain` | γ·x | γ | `adc_physical_noise` |
| `ADCOffset` | o·FS | o | `adc_physical_noise` |
| `ADCNonlinearity` | cubic INL, peak a·FS | a | `adc_physical_noise` |
| `ConductanceDrift` | −α·x, once when programmed | α | `weight_drift` |
| `PowerLawDrift` | −x·(1 − (t/t₀)^−ν), once when programmed | 1 − (t/t₀)^−ν | `weight_drift` |

Citations are in each class's docstring.

### Several profiles on one tensor

Give a list. The errors add, and `estimate` treats each profile as a separate noise
source (Additivity Law):

```yaml
adc_physical_noise:
  - [{type: ADCGain, args: [0.001]}, {type: ADCOffset, args: [0.0005]}]
```

### One strength per profile

`estimate` interpolates each noise source along one axis: its scaling factor.
Thus a profile has:
- one **strength**: the value that `noise * k` scales. It alone sets `get_scaling_factor`.
- optional fixed **shape** parameters, from `shape()` (e.g. `p=0.1`). A different shape is
  a different noise source.

`accuracim.noise.check_profile` tests this. Each estimate run checks its profiles first.
For two independent settings (e.g. gain and offset), use two profiles in a list.

### Add a noise profile

Subclass `Noise`:
- `__call__` returns the error tensor.
- `get_scaling_factor` returns the strength (like a variance).
- `__mul__` scales the strength.

Draw random numbers with `normal_like` / `uniform_like`. Calibration runs and the sweep
then get the same random draw.

```python
from accuracim.noise import Noise, uniform_like

class UniformNoise(Noise):
    """Uniform error in [-alpha * FS, alpha * FS]."""
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = abs(alpha)

    def __call__(self, x, low=0, high=0, **kwargs):
        return self.alpha * abs(high - low) * (2 * uniform_like(x) - 1)

    def get_scaling_factor(self):
        return self.alpha ** 2

    def __mul__(self, scalar):
        return UniformNoise(self.alpha * scalar)
```

```yaml
params_grid_list:
  - weight_bits: [8]
    weight_physical_noise:
      - {type: "workloads.my_noise:UniformNoise", args: [0.01]}
```

For a fixed parameter, add `shape()`, as in `TelegraphNoise`: `return (f"p={self.p}",)`.
