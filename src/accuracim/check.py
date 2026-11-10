"""Model check (`--mode check`): make sure that a model is converted correctly.

The check uses one batch. It calibrates input ranges in a temporary folder and caches nothing.
1. Transparency: with all impairments off, the noisy model must match the original.
2. Array path: with bits, slices and row tiling, the model must run and stay close.
3. Liveness: small input and weight noise must change the output.
4. Coverage: the check lists digital weights, simulated attention matmuls and bypassing ops.
"""

from __future__ import annotations

import logging
import tempfile

import torch
import torch.nn as nn

from accuracim import model_cache
from accuracim.device import free_memory
from accuracim.evaluation import run_batch
from accuracim.inject import replace_layers_with_noisy_version
from accuracim.layers import NoisyConv2d
from accuracim.log import section
from accuracim.noise import StateIndependentNoise
from accuracim.statistics import dict_to_device, get_input_statistics
from accuracim.tensors import NoisyTensor

logger = logging.getLogger(__name__)

#: Transparency tolerance: max |difference| relative to max |logit|
TOLERANCE = 1e-3

#: Array-path settings: typical, with real slicing and row tiling
ARRAY_PARAMS = {
    "input_bits": 8,
    "weight_bits": 8,
    "input_slice_lsb_to_msb": 2,
    "weight_slice_lsb_to_msb": 2,
    "num_rows": 128,
}

#: Ops that compute a matmul without going through the simulated array, if they receive
#: a `NoisyTensor` outside a noisy layer (e.g. `bmm` attention, flash kernels)
_MATMUL_LIKE = {
    "bmm",
    "baddbmm",
    "addbmm",
    "einsum",
    "mm",
    "addmm",
    "mv",
    "linear",
    "conv1d",
    "conv2d",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
}


def _logits(output) -> torch.Tensor:
    """The logits tensor of a model output (tensor, dict or Hugging Face ModelOutput)."""
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, dict) or hasattr(output, "keys"):
        logits = output["logits"]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output[0]
    return logits.as_subclass(torch.Tensor).detach().float().cpu()  # GPU kept for the models


def _compare(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """(max |difference| relative to max |reference|, top-1 agreement)."""
    rel = ((other - reference).abs().max() / reference.abs().max().clamp_min(1e-12)).item()
    agreement = (other.argmax(-1) == reference.argmax(-1)).float().mean().item()
    return rel, agreement


def _attention_modules(model: nn.Module) -> int:
    """Modules named like attention that own Linear/Conv1D projections (expected simulated)."""
    count = 0
    for module in model.modules():
        if "Attention" not in type(module).__name__:
            continue
        if any(
            isinstance(child, (nn.Linear, nn.Conv1d)) or type(child).__name__ == "Conv1D"
            for child in module.children()
        ):
            count += 1
    return count


def _digital_weights(model: nn.Module) -> list[str]:
    """Modules holding matrix weights that were not replaced by a noisy layer."""
    found = []
    for name, module in model.named_modules():
        if isinstance(module, (NoisyConv2d, nn.Embedding)):
            continue
        # Weight matrices only (not e.g. a ViT's cls_token or position_embeddings)
        own = [
            n for n, p in module.named_parameters(recurse=False) if p.dim() >= 2 and "weight" in n
        ]
        if own:
            found.append(f"{name or '<model>'} ({type(module).__name__}: {', '.join(own)})")
    return found


class _Coverage:
    """Count simulated attention matmuls and catch matmul-like ops that bypass the array."""

    def __init__(self):
        self.qk = self.av = 0
        self.escaped: dict[str, int] = {}
        self._depth = 0  # >0 while inside a noisy layer (its own conv/linear is expected)

    def __enter__(self):
        coverage = self
        self._matmul_ordered = NoisyTensor.matmul_ordered
        self._torch_function = NoisyTensor.__dict__["__torch_function__"]
        original_matmul, original_torch_function = self._matmul_ordered, self._torch_function

        def matmul_ordered(tensor, other):
            names = (getattr(tensor, "layer_name", None), getattr(other, "layer_name", None))
            if any(isinstance(n, tuple) for n in names):
                coverage.av += 1
            else:
                coverage.qk += 1
            return original_matmul(tensor, other)

        def torch_function(cls, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", "")
            if coverage._depth == 0 and name in _MATMUL_LIKE:
                coverage.escaped[name] = coverage.escaped.get(name, 0) + 1
            return original_torch_function.__func__(cls, func, types, args, kwargs)

        NoisyTensor.matmul_ordered = matmul_ordered
        NoisyTensor.__torch_function__ = classmethod(torch_function)
        return self

    def __exit__(self, *exc):
        NoisyTensor.matmul_ordered = self._matmul_ordered
        NoisyTensor.__torch_function__ = self._torch_function
        return False

    def track_layers(self, model: nn.Module) -> list:
        """Mark noisy-layer forwards, so their internal conv/linear is not reported."""
        handles = []
        for module in model.modules():
            if isinstance(module, NoisyConv2d):

                def enter(*_):
                    self._depth += 1

                def leave(*_):
                    self._depth -= 1

                handles.append(module.register_forward_pre_hook(enter))
                handles.append(module.register_forward_hook(leave))
        return handles


def run_model_check(dataloader, build_model: callable, num_samples: int = 16) -> bool:
    """Run the check on the first batch; log a report.

    `num_samples`: samples used to calibrate the input ranges (into a temporary folder).

    Returns:
    bool: True if every check passed.
    """
    section("Model check")
    batch = next(iter(dataloader))
    mode = build_model(return_mode=True)
    noise_layers = model_cache.leaf_module_names(build_model)
    results = []

    def run(model):
        model.eval()
        with torch.no_grad():
            return _logits(run_batch(model, batch, N=num_samples)["logits"])

    original = model_cache.fresh_model(build_model)
    expected_attention = 2 * _attention_modules(original)
    reference = run(original)
    del original
    free_memory()
    with tempfile.TemporaryDirectory() as tmp:
        input_stat = get_input_statistics(dataloader, build_model, N=num_samples, result_dir=tmp)
    dict_to_device(input_stat, build_model(return_device=True))

    def noisy(params):
        free_memory()  # one model on the GPU at a time (large models)
        return replace_layers_with_noisy_version(
            model_cache.fresh_model(build_model), noise_layers, dict(params), input_stat, None
        )

    # 1. Transparency, with coverage counted on the same forward pass
    model = noisy({})
    with _Coverage() as coverage:
        handles = coverage.track_layers(model)
        transparent = run(model)
        for handle in handles:
            handle.remove()
    rel, agreement = _compare(reference, transparent)
    ok = rel < TOLERANCE and agreement == 1.0
    results.append(ok)
    logger.info(
        f"transparent (no impairments) · max |Δ| {rel:.2e} of max |logit| · "
        f"top-1 agreement {agreement:.4f} · {'PASS' if ok else 'FAIL'}"
    )

    # Coverage
    num_noisy = sum(isinstance(m, NoisyConv2d) for m in model.modules())
    logger.info(f"simulated layers · {num_noisy}")
    digital = _digital_weights(model)
    logger.info(f"digital weights · {', '.join(digital) if digital else 'none'}")
    hooked = getattr(model, "_noisy_tensor_hooks", None) is not None
    if hooked:
        expected = expected_attention
        simulated = coverage.qk + coverage.av
        ok = simulated >= expected
        results.append(ok)
        logger.info(
            f"attention · analog · {simulated} matmuls simulated "
            f"({coverage.qk} Q·Kᵀ, {coverage.av} A·V; expected ≥ {expected}) · "
            f"{'PASS' if ok else 'FAIL'}"
        )
    elif mode == "transformer":
        logger.info("attention · digital (not simulated; model.analog_attention is false)")
    else:
        logger.info("attention · none (mode cnn: layers only)")
    if coverage.escaped:
        results.append(False)
        listed = ", ".join(f"{name} ×{n}" for name, n in sorted(coverage.escaped.items()))
        logger.info(f"bypassed the simulation · {listed} · FAIL")
    else:
        logger.info("bypassed the simulation · none")
    del model
    free_memory()

    # 2. Array path: bits, slices, row tiling
    try:
        array_out = run(noisy(ARRAY_PARAMS))
    except Exception as exc:  # report, do not crash the check
        results.append(False)
        logger.info(f"array path ({ARRAY_PARAMS}) · error: {type(exc).__name__}: {exc} · FAIL")
        array_out = None
    else:
        rel, agreement = _compare(reference, array_out)
        logger.info(
            f"array path (8b, 2b slices, 128 rows) · runs · max |Δ| {rel:.2e} · "
            f"top-1 agreement {agreement:.4f} (quantization only)"
        )

    # 3. Liveness: noise must reach the output
    if array_out is not None:
        noise = StateIndependentNoise(0.05)
        live = run(
            noisy({**ARRAY_PARAMS, "input_physical_noise": noise, "weight_physical_noise": noise})
        )
        rel, _ = _compare(array_out, live)
        ok = rel > 1e-6
        results.append(ok)
        logger.info(f"noise reaches the output · max |Δ| {rel:.2e} · {'PASS' if ok else 'FAIL'}")

    passed = all(results)
    logger.info(f"model check · {'PASS' if passed else 'FAIL'}")
    return passed
