"""Turn a model into a noisy model with `replace_layers_with_noisy_version`.

- With `probe_inputs`: add hooks, so that `statistics` can learn the input range of each layer.
- Else: replace each Conv2d/Linear/Conv1D with its noisy version (and fold a BatchNorm after it).
Transformers also wrap projection outputs in `NoisyTensor`, so that attention matmuls get noise.
"""

import logging

import torch.nn as nn

# For Hugging Face's Conv1D
from transformers.pytorch_utils import Conv1D

from accuracim.graph import fuse_conv_bn, get_next_module, set_module_by_name
from accuracim.layers import NoisyConv1D, NoisyConv2d, NoisyLinear
from accuracim.quantize import place_weight_slices
from accuracim.sampling import return_input_hook_fn
from accuracim.tensors import (
    attention_mask_context_patch_model_forward,
    return_noisy_tensor_hook,
)

logger = logging.getLogger(__name__)


def add_noisy_tensor_hooks(
    model: nn.Module,
    params: dict = {},
    input_percentiles: dict = None,
    output_percentiles: dict = None,
    probe_inputs: bool = False,
    probe_outputs: bool = False,
    store_dict={},
    verbose: bool = False,
    *args,
    **kwargs,
):
    """Hook every Linear/Conv layer so its output becomes a `NoisyTensor` (noisy matmuls downstream).

    Returns:
    list: The hook handles (kept on the model, so `set_forward_params` can replace them).
    """
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, Conv1D)):
            handle = module.register_forward_hook(
                return_noisy_tensor_hook(
                    layer_name=name,
                    params=params,
                    input_percentiles=input_percentiles,
                    output_percentiles=output_percentiles,
                    probe_inputs=probe_inputs,
                    probe_outputs=probe_outputs,
                    store_dict=store_dict,
                )
            )
            handles.append(handle)
            if verbose:
                logger.debug(f"Noisy Tensor Hook added to {name}")
    return handles


def replace_layers_with_noisy_version(
    model: nn.Module,
    noise_layers: list | set = None,
    params: dict = None,
    input_percentiles: dict = None,
    output_percentiles: dict = None,
    probe_inputs: bool = False,
    probe_outputs: bool = False,
    store_dict={},
    sample_fraction=0.02,
    verbose: bool = False,
):
    """Probe layer inputs (`probe_inputs`), or swap `noise_layers` for noisy versions in place.

    Returns:
    nn.Module: The noisy model; None when probing.
    """
    # "transformer": attention matmuls get noise too (NoisyTensor); "cnn": layers only.
    # Older names "LLM" / "CNN" are accepted.
    mode = {"LLM": "transformer", "CNN": "cnn"}.get(model.mode, model.mode)
    assert mode in [
        "transformer",
        "cnn",
    ], f"Unknown model mode: {model.mode}, supports ['cnn', 'transformer']"

    # Attention matmuls are simulated unless the model opts out (`analog_attention: false`)
    simulate_attention = mode == "transformer" and getattr(model, "analog_attention", True)

    # Fresh defaults per call: `params` is mutated below
    params = {} if params is None else params
    noise_layers = set() if noise_layers is None else set(noise_layers)
    named_modules_list = list(model.named_modules())

    params["sample_fraction"] = sample_fraction

    # Probe mode: record layer inputs, layers unchanged
    if probe_inputs:
        for name, module in named_modules_list:
            if verbose:
                logger.debug(f"Layer {name}: {module}")
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, Conv1D)):
                store_dict[name] = {"low": [], "high": []}
                module.register_forward_hook(
                    return_input_hook_fn(name, sample_fraction, store_dict)
                )

        if simulate_attention:
            # Also probe matmul operands (stored as {'input' | 'weight': percentiles})
            model._noisy_tensor_hooks = add_noisy_tensor_hooks(
                model,
                params=params,
                probe_inputs=True,
                probe_outputs=False,
                store_dict=store_dict,
                verbose=verbose,
            )
            # Patch forward to provide the attention-mask context
            attention_mask_context_patch_model_forward(model)
        return

    for name, module in named_modules_list:
        if name not in noise_layers:
            continue

        if not isinstance(module, (nn.Linear, nn.Conv1d, Conv1D, nn.Conv2d)):
            continue

        # Shared kwargs for every noisy layer
        common = dict(
            name=name,
            probe_outputs=probe_outputs,
            input_percentiles=input_percentiles[name],
            output_percentiles=output_percentiles[name] if output_percentiles is not None else None,
            **params,
        )

        if isinstance(module, nn.Linear):
            new_module = NoisyLinear(
                module=module,
                in_features=module.in_features,
                out_features=module.out_features,
                **common,
            )

        elif isinstance(module, nn.Conv1d):
            new_module = NoisyConv1D(
                module=module,
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                **common,
            )

        elif isinstance(module, Conv1D):
            # Hugging Face Conv1D stores weights as (out_features, in_features)
            new_module = NoisyConv1D(
                module=module,
                in_channels=module.weight.shape[1],
                out_channels=module.weight.shape[0],
                **common,
            )

        elif isinstance(module, nn.Conv2d):
            # Fold a following BatchNorm into the conv, then replace it with Identity
            next_module, next_name = get_next_module(model, name)
            weight, bias = fuse_conv_bn(model, name, module.weight.data, return_bias=True)
            new_module = NoisyConv2d(
                weight_tensor=weight,
                bias_tensor=bias,
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(bias is not None),
                padding_mode=module.padding_mode,
                device=module.weight.device,
                dtype=module.weight.dtype,
                **common,
            )
            if isinstance(next_module, nn.BatchNorm2d):
                set_module_by_name(model, next_name, nn.Identity())

        set_module_by_name(model, name, new_module)
        if verbose:
            logger.debug(f"Initialized {type(new_module).__name__} '{name}'")

    # Transformer: noisy attention matmuls via NoisyTensor
    if simulate_attention:
        model._noisy_tensor_hooks = add_noisy_tensor_hooks(
            model,
            params=params,
            input_percentiles=input_percentiles,
            output_percentiles=output_percentiles,
            probe_inputs=probe_inputs,
            probe_outputs=probe_outputs,
            store_dict=store_dict,
            verbose=verbose,
        )

        # Patch forward to provide the attention-mask context
        attention_mask_context_patch_model_forward(model)

    place_weight_slices(model)
    return model


def set_forward_params(
    model: nn.Module,
    params: dict,
    input_percentiles: dict = None,
    output_percentiles: dict = None,
    sample_fraction=0.02,
):
    """Point a programmed model at new per-pass params, as a fresh build would set them.

    Valid only if `params` programs the model the same way (`params.programming_key`):
    the weights, slices and scales are kept; the params every noisy layer and `NoisyTensor`
    hook reads on each forward pass are replaced. Mirrors `replace_layers_with_noisy_version`.
    """
    params["sample_fraction"] = sample_fraction
    for name, module in model.named_modules():
        if isinstance(module, NoisyConv2d):
            assert module.name == name, f"Noisy layer {module.name!r} moved to {name!r}"
            assert not module.probe_outputs, "Probe models are never reused"
            module.params = dict(params)
            module.output_percentiles = (
                output_percentiles[name] if output_percentiles is not None else None
            )

    # Hooks capture params when registered: replace them, never stack them
    handles = getattr(model, "_noisy_tensor_hooks", None)
    if handles is not None:
        for handle in handles:
            handle.remove()
        model._noisy_tensor_hooks = add_noisy_tensor_hooks(
            model,
            params=params,
            input_percentiles=input_percentiles,
            output_percentiles=output_percentiles,
            probe_inputs=False,
            probe_outputs=False,
            store_dict={},
        )
    return model
