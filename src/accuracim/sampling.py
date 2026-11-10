"""Sub-sampling of large activations, and the stores for the samples.

The code never keeps full activations. It keeps the extreme tails to fit clipping ranges
(`sample_fraction_tensor`), and the top-k entries to score accuracy
(`get_top_percentile_indices`). The top-k functions are compiled with TorchScript.
"""

from collections import defaultdict
from math import ceil, floor

import torch

from accuracim.params import close_eq


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def convert_nested_defaultdict_to_dict(d):
    """Recursively convert nested defaultdicts to dicts."""
    if isinstance(d, defaultdict):
        return {k: convert_nested_defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d


def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`torch.quantile` for one scalar `q`, via `torch.kthvalue`.

    No 2^24 size limit (pytorch/pytorch#64947) and faster on large tensors. `out` unsupported.
    """
    q_float = float(q)
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # dim=None: flatten (squeeze/kthvalue reject None)
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Indices (and weight) to interpolate between
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


def sample_fraction_tensor(tensor: torch.Tensor, sample_fraction: float) -> tuple[torch.Tensor]:
    """Return (low, high): the lowest / highest `sample_fraction` of elements, as 1D tensors.

    An all-zero `low` (e.g. ReLU) collapses to [0.].
    """
    assert 0 < sample_fraction <= 0.5, "sample_fraction must be between 0 and 0.5."

    num_elements = tensor.numel()
    num_samples = int(num_elements * sample_fraction)

    low_indices = torch.topk(tensor.reshape(-1), num_samples, largest=False, sorted=False).indices
    low = tensor.reshape(-1)[low_indices]
    if close_eq(low.min(), low.max()) and close_eq(low[0], 0):
        low = torch.Tensor([0.0])

    high_indices = torch.topk(tensor.reshape(-1), num_samples, largest=True, sorted=False).indices
    high = tensor.reshape(-1)[high_indices]
    return (low, high)


def get_top_percentile_indices_no_jit_(
    tensor: torch.Tensor, p: float = 5.0, k_min: int = 75, k_max: int = 150
):
    """Return top-k indices along the last dim: (B, ..., L) -> (B, ..., k), k = p% of L clamped to [k_min, k_max]."""
    last_len = tensor.size(-1)
    k = int(
        torch.clamp(
            torch.tensor(last_len * p / 100.0, device=tensor.device),
            min=min(k_min, last_len),
            max=min(k_max, last_len),
        )
    )
    _, top_indices = torch.topk(tensor, k, dim=-1, largest=True, sorted=False)
    return top_indices


def gather_top_percentile_values_no_jit_(tensor: torch.Tensor, top_indices: torch.Tensor):
    """Gather values at `top_indices` along the last dim: (B, ..., L) -> (B, ..., k)."""
    return torch.gather(tensor, dim=-1, index=top_indices)


def add_sampled_values_per_index_no_jit_(
    result: torch.Tensor,
    sampled_error: torch.Tensor,
    top_indices: torch.Tensor,
):
    """Scatter-add `sampled_error` (B, ..., k) into `result` (B, ..., L) at `top_indices`, in place."""
    result.scatter_add_(-1, top_indices, sampled_error)
    return result


# TorchScript-compiled versions
get_top_percentile_indices: callable = torch.jit.script(get_top_percentile_indices_no_jit_)
gather_top_percentile_values: callable = torch.jit.script(gather_top_percentile_values_no_jit_)
add_sampled_values_per_index: callable = torch.jit.script(add_sampled_values_per_index_no_jit_)


def return_input_hook_fn(name, sample_fraction, store_dict):
    """Return a forward hook appending input tails to `store_dict[name]['low'/'high']`."""

    def hook_fn(module, input, output):
        low, high = sample_fraction_tensor(input[0], sample_fraction)
        store_dict[name]["low"].append(low.detach().cpu())
        store_dict[name]["high"].append(high.detach().cpu())

    return hook_fn
