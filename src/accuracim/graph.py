"""Find, rewrite and hook the modules of a built `nn.Module` by dotted name.

`inject` uses it to add noisy layers. `simulate` uses it to tap probe layers.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_module_by_name(model: nn.Module, layer_name: str):
    """Return the module at a dotted name, or None."""
    names = layer_name.split(".")
    module = model
    for name in names:
        try:
            if name.isdigit():
                module = module[int(name)]
            else:
                module = getattr(module, name)
        except:
            logger.warning(f"Error accessing module '{name}' in layer '{layer_name}'")
            return None

        if module is None:
            logger.warning(f"Module '{layer_name}' not found in the model")
            return None
    return module


def get_parent_and_child_name(model, layer_name):
    names = layer_name.split(".")
    parent = model
    for name in names[:-1]:
        if name.isdigit():
            parent = parent[int(name)]
        else:
            parent = getattr(parent, name)
    return parent, names[-1]


def get_next_module(model, layer_name):
    """Return (next sibling module, its dotted name), or (None, None) if last."""
    parent, child_name = get_parent_and_child_name(model, layer_name)
    if isinstance(parent, nn.Sequential):
        keys = list(map(str, range(len(parent))))
    else:
        keys = list(dict(parent.named_children()).keys())
    idx = keys.index(child_name)
    if idx + 1 >= len(keys):
        return None, None
    next_name = keys[idx + 1]
    next_module = (
        getattr(parent, next_name)
        if not isinstance(parent, nn.Sequential)
        else parent[int(next_name)]
    )
    next_full_name = ".".join(layer_name.split(".")[:-1] + [next_name])
    return next_module, next_full_name


def set_module_by_name(model: nn.Module, module_path: str, new_module: nn.Module):
    if module_path is None or module_path == "":
        raise ValueError("module_path must be a non-empty string")
    parts = module_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def register_fused_conv_bn_hook(
    model,
    layer_name,
    input_store,
    output_store,
    hook_input=False,
    hook_output=True,
    detach_cpu=True,
):
    """Hook a layer's input and/or output into `input_store['input']` / `output_store['output']`.

    Supports Linear, Conv1d, Conv2d, and `__output__` (whole model). A Conv2d followed
    by BatchNorm2d has its output taken after the BN (fused). Stores detached CPU
    copies if `detach_cpu`.

    Returns:
    (input_handle, output_handle) if both hooked, else the single handle;
    None if the module is missing or unsupported (e.g. BatchNorm).
    """
    if layer_name == "__output__":
        module = model
    else:
        module = get_module_by_name(model, layer_name)

    if module is None:
        return None

    def input_hook(mod, inp, out):
        # First input only
        if detach_cpu:
            input_store["input"] = inp[0].detach().cpu()
        else:
            input_store["input"] = inp[0]

    def output_hook(mod, inp, out):
        # Tensor, ModelOutput.logits, or first tuple item
        output = None
        if isinstance(out, torch.Tensor):
            output = out
        elif hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            output = out.logits
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            output = out[0]

        if output is not None and detach_cpu:
            output = output.detach().cpu()
        output_store["output"] = output

    input_handle = None
    output_handle = None

    # Linear / Conv1d / whole model: hook directly
    if isinstance(module, (nn.Linear, nn.Conv1d)) or layer_name == "__output__":
        if hook_input:
            input_handle = module.register_forward_hook(input_hook)
        if hook_output:
            output_handle = module.register_forward_hook(output_hook)

    # Conv2d: output after the following BN, if any
    elif isinstance(module, nn.Conv2d):
        next_module, next_name = get_next_module(model, layer_name)
        if isinstance(next_module, nn.BatchNorm2d):
            if hook_input:
                input_handle = module.register_forward_hook(input_hook)
            if hook_output:
                output_handle = next_module.register_forward_hook(output_hook)
        else:
            if hook_input:
                input_handle = module.register_forward_hook(input_hook)
            if hook_output:
                output_handle = module.register_forward_hook(output_hook)
    else:
        # BatchNorm / unsupported
        return None

    if hook_input and hook_output:
        return (input_handle, output_handle)
    return input_handle or output_handle


def fuse_conv_bn(model, layer_name, weight, return_bias=False, epsilon=None):
    """Fold the BatchNorm2d following `layer_name` (if any) into the conv `weight`.

    `epsilon`: None uses the BatchNorm's own `eps`, so the folded conv matches it exactly.

    Returns:
    weight, or (weight, bias) if `return_bias`; bias is None without a BN.
    """
    next_module, next_name = get_next_module(model, layer_name)

    conv_layer = get_module_by_name(model, layer_name)
    if hasattr(conv_layer, "bias") and conv_layer.bias is not None:
        conv_bias = conv_layer.bias.data
    else:
        conv_bias = None
    bias = conv_bias  # kept as is without a BatchNorm

    if isinstance(next_module, nn.BatchNorm2d):
        bn_weight = next_module.weight.data
        bn_var = next_module.running_var.data
        bn_bias = next_module.bias.data
        bn_mean = next_module.running_mean.data

        # w' = w * gamma / sqrt(var + eps);  b' = (b - mean) * gamma / sqrt(var + eps) + beta
        eps = next_module.eps if epsilon is None else epsilon
        std_inv = bn_weight / torch.sqrt(bn_var + eps)

        weight = weight * std_inv.view(-1, 1, 1, 1)
        if conv_bias is not None:
            bias = (conv_bias - bn_mean) * std_inv + bn_bias
        else:
            bias = -bn_mean * std_inv + bn_bias

    if return_bias:
        return weight, bias
    else:
        return weight


def get_all_leaf_module_names(model):
    """Return names of all leaf modules except BatchNorm2d."""
    names = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and not isinstance(module, nn.BatchNorm2d):
            names.append(name)
    return names
