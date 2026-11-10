"""Models: torchvision CNNs, and Hugging Face language models and ViT.

A config selects a model by name:

    model:
      builder: workloads.models:build_model
      name: ResNet18
      # analog_attention: false   # transformers only: keep attention digital (default: true)

Analog attention needs the "sdpa" kernel. A requested flash or flex kernel changes to "sdpa".
Digital attention keeps the kernel of the model and adds `-digital-attn` to the model name.

To add a model, add an entry to `MODELS`. For other architectures, write a builder
with the signature of `build_model`.

The pipeline uses only `build_model`. With `return_mode`, `return_device` or
`return_model_name`, it returns that value instead of a model.
"""

import os

import torch

from accuracim.config import project_root
from accuracim.device import get_best_gpu
from accuracim.log import get_logger

import torch.nn as nn

# torchvision CNNs
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg16_bn,
    mobilenet_v2,
    efficientnet_b0,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG16_BN_Weights,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)

# Hugging Face
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageClassification

logger = get_logger(__name__)

#: One entry per model.
#: `mode`: "cnn", or "transformer" (attention matmuls can be simulated too).
#: `torchvision`: (constructor, pretrained weights). `hf_id`: Hugging Face model id;
#: `hf_task`: "causal_lm" (default, WikiText-2) or "image_classification" (ImageNet).
# fmt: off
MODELS = {
    # ImageNet
    "ResNet18":     {"mode": "cnn", "torchvision": (resnet18, ResNet18_Weights.IMAGENET1K_V1)},
    "ResNet34":     {"mode": "cnn", "torchvision": (resnet34, ResNet34_Weights.IMAGENET1K_V1)},
    "ResNet50":     {"mode": "cnn", "torchvision": (resnet50, ResNet50_Weights.IMAGENET1K_V1)},
    "ResNet101":    {"mode": "cnn", "torchvision": (resnet101, ResNet101_Weights.IMAGENET1K_V1)},
    "ResNet152":    {"mode": "cnn", "torchvision": (resnet152, ResNet152_Weights.IMAGENET1K_V1)},
    "VGG16-BN":     {"mode": "cnn", "torchvision": (vgg16_bn, VGG16_BN_Weights.IMAGENET1K_V1)},
    "MobileNetV2":  {"mode": "cnn", "torchvision": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1)},
    "EfficientNet-B0": {"mode": "cnn", "torchvision": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1)},
    "ViT-B16":      {"mode": "transformer", "hf_id": "google/vit-base-patch16-224", "hf_task": "image_classification"},
    # WikiText-2 (set dataset.params.tokenizer to the same hf_id)
    "GPT2-Small":   {"mode": "transformer", "hf_id": "gpt2"},
    "GPT2-Medium":  {"mode": "transformer", "hf_id": "gpt2-medium"},
    "GPT2-Large":   {"mode": "transformer", "hf_id": "gpt2-large"},
    "GPT2-XL":      {"mode": "transformer", "hf_id": "gpt2-xl"},
    "DistilGPT2":   {"mode": "transformer", "hf_id": "distilbert/distilgpt2"},
    "SmolLM2-135M": {"mode": "transformer", "hf_id": "HuggingFaceTB/SmolLM2-135M"},
    "SmolLM2-360M": {"mode": "transformer", "hf_id": "HuggingFaceTB/SmolLM2-360M"},
    "Qwen2.5-0.5B": {"mode": "transformer", "hf_id": "Qwen/Qwen2.5-0.5B"},
    "Pythia-160M":  {"mode": "transformer", "hf_id": "EleutherAI/pythia-160m"},
}
# fmt: on

_HF_CLASSES = {
    "causal_lm": AutoModelForCausalLM,
    "image_classification": AutoModelForImageClassification,
}

#: Attention kernels whose matmuls `NoisyTensor` intercepts
_INTERCEPTABLE_ATTENTION = ("sdpa", "eager")

#: The selected model; `build_model(model_name=...)` switches it
model_name = "ResNet18"

#: Transformers: simulate attention matmuls (True), or keep the model's own kernel (False)
analog_attention = True
#: Attention kernel to request (None: the model's default)
attn_implementation = None

# Weights of the selected model and the device, both resolved on first use
_state_dict = None
_device = None


def _get_device() -> str:
    """Return the GPU with the most free memory (else CPU), chosen once on first use."""
    global _device
    if _device is None:
        _device = get_best_gpu()
    return _device


def _resolve_device(device):
    """Map "auto" to `_get_device()`; return anything else (a device, or None) as given."""
    if isinstance(device, str) and device == "auto":
        return _get_device()
    return device


def _weights_path(name: str) -> str:
    return os.path.join(project_root(), "calibrated_config", name, "weights.pth")


def get_or_create_state_dict(path: str, model_name: str) -> dict:
    """Load a state_dict from `path`, or download pretrained `model_name` weights and save them there."""
    if os.path.exists(path):
        logger.debug(f"Loading weights from {path}")
        state_dict = torch.load(path, map_location=_get_device(), weights_only=True)
    else:
        logger.info(f"Weights not found at {path}; downloading pretrained {model_name}")
        spec = MODELS[model_name]

        if "torchvision" in spec:
            constructor, weights = spec["torchvision"]
            state_dict = constructor(weights=weights).state_dict()
        else:
            hf_class = _HF_CLASSES[spec.get("hf_task", "causal_lm")]
            state_dict = hf_class.from_pretrained(spec["hf_id"]).state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(state_dict, path)
        logger.info(f"Saved pretrained weights to {path}")

    return state_dict


def _current_state_dict() -> dict:
    """Return the selected model's weights, loading (or downloading) them on first use."""
    global _state_dict
    if _state_dict is None:
        _state_dict = get_or_create_state_dict(_weights_path(model_name), model_name)
    return _state_dict


def set_model_name(name: str) -> None:
    """Select the model `build_model` builds; its weights load on first use."""
    global model_name, _state_dict
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODELS)}. "
            "Add it to MODELS in workloads/models.py, or set model.builder to your own builder."
        )
    model_name = name
    _state_dict = None


def _display_name() -> str:
    """The model's name for calibration, caches and results (attention setting included)."""
    digital = MODELS[model_name]["mode"] == "transformer" and not analog_attention
    return model_name + ("-digital-attn" if digital else "")


def _attention_kernel() -> str | None:
    """The attention kernel to load with: interceptable when attention is simulated."""
    if analog_attention and attn_implementation not in (None, *_INTERCEPTABLE_ATTENTION):
        logger.warning(
            f"attn_implementation {attn_implementation!r} cannot be simulated; using 'sdpa' "
            "(set model.analog_attention: false to keep it)"
        )
        return "sdpa"
    return attn_implementation


class ImageClassifier(nn.Module):
    """Hugging Face image classifier on ImageNet-normalized input; returns plain logits.

    The ImageNet loaders normalize with torchvision's mean/std; the model expects its own.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.mean, self.std = tuple(mean), tuple(std)

    def forward(self, x):
        def column(values):
            return torch.tensor(values, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)

        pixels = x * column(self._IMAGENET_STD) + column(self._IMAGENET_MEAN)
        pixels = (pixels - column(self.mean)) / column(self.std)
        logits = self.model(pixel_values=pixels).logits
        # Plain tensor out, as a CNN returns (no simulated-matmul wrapper past the model)
        return logits.as_subclass(torch.Tensor)


def _image_normalization(hf_id: str) -> tuple:
    """(mean, std) the Hugging Face image model expects (from its preprocessor config)."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(hf_id)
    return processor.image_mean, processor.image_std


def build_model(state_dict: dict | None = None, device="auto", *args, **kwargs):
    """Build the selected model on `device`, or return its metadata.

    Parameters:
    state_dict (dict | None): Weights to load; None uses the selected model's weights.
    device: "auto" (the GPU with the most free memory, else CPU), a device, or None
        (left where it is built, i.e. CPU).
    kwargs: `model_name` selects the model first. `return_mode` / `return_state_dict` /
        `return_device` / `return_model_name` return that value instead of a model.
        Remaining args go to the torchvision constructor.
    """
    global analog_attention, attn_implementation
    model_name_override = kwargs.pop("model_name", None)
    if model_name_override is not None and model_name_override != model_name:
        set_model_name(model_name_override)
    if "analog_attention" in kwargs:
        analog_attention = bool(kwargs.pop("analog_attention"))
    if "attn_implementation" in kwargs:
        attn_implementation = kwargs.pop("attn_implementation")

    spec = MODELS[model_name]

    if kwargs.pop("return_mode", False):
        return spec["mode"]
    if kwargs.pop("return_state_dict", False):
        return state_dict if state_dict is not None else _current_state_dict()
    if kwargs.pop("return_device", False):
        return _resolve_device(device)
    if kwargs.pop("return_model_name", False):
        return _display_name()

    if state_dict is None:
        state_dict = _current_state_dict()
    device = _resolve_device(device)

    if "torchvision" in spec:
        constructor, _ = spec["torchvision"]
        model = constructor(*args, **kwargs)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    elif spec.get("hf_task", "causal_lm") == "causal_lm":
        hf_id = spec["hf_id"]
        loss_type = "ForCausalLMLoss"
        # Only pass a kernel when one is needed, so the default load stays as before
        kernel = _attention_kernel()
        load_kwargs = {"attn_implementation": kernel} if kernel else {}
        if isinstance(state_dict, dict) and state_dict:
            config = AutoConfig.from_pretrained(hf_id)
            config.loss_type = loss_type
            model = AutoModelForCausalLM.from_config(config, **load_kwargs)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.warning(
                    f"{model_name}: weights loaded non-strictly; missing {missing}, unexpected {unexpected}"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
            model.config.loss_type = loss_type
        # Simulated in float32 (some checkpoints default to bfloat16; GPT-2 is float32 already)
        model = model.float().to(device)

    else:  # Hugging Face image classification (ImageNet labels)
        hf_id = spec["hf_id"]
        kernel = _attention_kernel()
        load_kwargs = {"attn_implementation": kernel} if kernel else {}
        config = AutoConfig.from_pretrained(hf_id)
        inner = AutoModelForImageClassification.from_config(config, **load_kwargs)
        if isinstance(state_dict, dict) and state_dict:
            inner.load_state_dict(state_dict)
        processor = _image_normalization(hf_id)
        model = ImageClassifier(inner.float(), *processor).to(device)
        model.eval()

    model.mode = spec["mode"]
    if spec["mode"] == "transformer":
        model.analog_attention = analog_attention
        hf_model = getattr(model, "model", model)  # ImageClassifier wraps the HF model
        used = getattr(getattr(hf_model, "config", None), "_attn_implementation", None)
        if analog_attention and used not in (None, *_INTERCEPTABLE_ATTENTION):
            raise ValueError(
                f"{model_name} uses attention kernel {used!r}, which cannot be simulated; "
                "use 'sdpa'/'eager', or set model.analog_attention: false"
            )

    return model
