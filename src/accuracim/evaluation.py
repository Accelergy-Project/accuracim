"""Run a batch and score it.

To add a metric, write a function with the signature of `update_performance_metrics`.
Set `evaluation.metric_fn` to it.

Contracts:
- `run_batch` returns `{inputs, targets, logits, batch_size}`.
  A `(inputs, targets)` batch runs `model(inputs)`. A dict batch runs `model(**batch)`,
  and its `labels` are the targets.
- `update_performance_metrics` adds raw counts (with `count`), per params index if given.
  Per-batch results can then be cached and added. `average_performance_metrics` divides
  at the end. Custom metrics must use `accumulate` to keep this format.

Evaluation modes (`evaluation.mode` in the config, else the `eval_mode` of the dataloader):
- "classification": top1/top5 over classes, logits (B, C)
- "next_token": next-token top1/top5, logits (B, T, V)
- "perplexity": summed cross-entropy, averaged to `cross_entropy` and `perplexity`

Metric types:
- Task metrics: model vs. labels. You can replace them with `metric_fn`.
- Agreement metrics: noisy vs. ideal model (`update_flip_metrics`). You cannot replace them.
"""

import torch
import torch.nn as nn
import numpy as np

from accuracim.device import get_device

#: Evaluation modes: which metric `update_performance_metrics` computes
EVAL_MODES = ("classification", "next_token", "perplexity")

#: Older names, still accepted. A model's mode ("cnn" / "transformer", formerly
#: "CNN" / "LLM") is the fallback when neither the config nor the dataloader names one.
_EVAL_MODE_ALIASES = {
    "CNN": "classification",
    "LLM": "next_token",
    "LLM_CE": "perplexity",
    "cnn": "classification",
    "transformer": "next_token",
}


def resolve_eval_mode(mode: str) -> str:
    """Return the evaluation mode `mode` names (older names accepted); raise if unknown."""
    resolved = _EVAL_MODE_ALIASES.get(mode, mode)
    if resolved not in EVAL_MODES:
        raise ValueError(f"Unknown evaluation mode: {mode!r}. Expected one of {EVAL_MODES}.")
    return resolved


def get_batch_size(batch_data, N, num_processed):
    """Return the batch size, capped at the `N - num_processed` samples left."""
    batch_size = 0

    if isinstance(batch_data, dict):
        batch_size = batch_data["input_ids"].size(0)
    elif isinstance(batch_data, (tuple, list)):
        batch_size = batch_data[1].size(0)
    return min(batch_size, N - num_processed)


def predictions_per_sample(batch_data, eval_mode: str) -> float:
    """Return how many predictions a sample contributes to the metric's `count`.

    Counted from the batch's targets, so any dataset works: one label per sample gives 1,
    per-token labels give their valid count (shifted by one for next-token metrics).
    """
    if isinstance(batch_data, dict):
        targets = batch_data.get("labels")
    elif isinstance(batch_data, (tuple, list)) and len(batch_data) > 1:
        targets = batch_data[1]
    else:
        targets = None
    if not isinstance(targets, torch.Tensor) or targets.dim() == 0:
        return 1.0
    if targets.dim() >= 2 and resolve_eval_mode(eval_mode) != "classification":
        targets = targets[:, 1:]  # position t predicts token t+1
    valid = (targets != -100) & (targets >= 0)
    return max(valid.sum().item() / targets.size(0), 1.0)


def run_batch(model: torch.nn.Module, batch_data, num_processed=0, N=np.inf):
    """Run one batch (truncated to `N` total) without grad; return `{inputs, targets, logits, batch_size}`."""
    device = get_device(model)
    batch_size = get_batch_size(batch_data, N, num_processed)
    # The last batch is cut to the `N - num_processed` samples left
    truncate = batch_size < get_batch_size(batch_data, np.inf, 0)

    with torch.no_grad():
        if isinstance(batch_data, dict):
            # Dict batch
            if truncate:
                limit = N - num_processed
                batch_data = {
                    k: (v[:limit].to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch_data.items()
                }
            else:
                batch_data = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch_data.items()
                }
            # Pop labels so the model skips its internal loss
            targets = batch_data.pop("labels", None)
            logits = model(**batch_data)
            # Unwrap noisy tensors so no further noise is added
            logits = {
                k: (v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v)
                for k, v in logits.items()
            }
            input_ids = batch_data.get("input_ids")
            inputs = input_ids.clone() if isinstance(input_ids, torch.Tensor) else input_ids

        elif isinstance(batch_data, (tuple, list)):
            # (inputs, targets) batch
            inputs, targets = batch_data
            if truncate:
                limit = N - num_processed
                inputs, targets = inputs[:limit], targets[:limit]

            inputs, targets = inputs.to(device), targets.to(device)
            batch_data = (inputs, targets)
            logits = model(inputs)

        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch_data)}; expected a dict or an (inputs, targets) tuple"
            )

        return {"inputs": inputs, "targets": targets, "logits": logits, "batch_size": batch_size}


def accuracy_before(outputs, targets, topk=(1, 5), return_counts=True):
    """Top-k accuracy over all classes (counts or fractions)."""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        if return_counts:
            res.append(correct_k.item())
        else:
            res.append((correct_k / batch_size).item())

    return tuple(res)


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    top_indices: torch.Tensor,
    topk=(1, 5),
    return_counts=True,
    ignore_index: int = -100,
):
    """Top-k accuracy where `outputs` holds only the candidates in `top_indices`.

    Parameters:
    outputs (Tensor): (..., K) logits gathered at `top_indices`.
    targets (Tensor): (...) class ids; `ignore_index` entries are skipped.
    top_indices (LongTensor): (..., K) candidate class ids; None -> plain top-k (`accuracy_before`).

    Returns:
    tuple: Per-k correct counts, or fractions if not `return_counts`.
    """
    if top_indices is None:
        return accuracy_before(outputs, targets, topk=topk, return_counts=return_counts)

    assert (
        outputs.shape[:-1] == targets.shape
    ), f"outputs.shape[:-1]={outputs.shape[:-1]} must match targets.shape={targets.shape}"
    assert (
        top_indices.shape[:-1] == targets.shape
    ), f"top_indices.shape[:-1]={top_indices.shape[:-1]} must match targets.shape={targets.shape}"
    assert top_indices.dtype == torch.long, "top_indices must be LongTensor"

    K = top_indices.size(-1)
    maxk = max(topk)

    # Top-k positions within the K candidates
    _, idx_in_selected = outputs.topk(min(maxk, K), dim=-1, largest=True, sorted=True)

    # k > K: pad with the last pick (adds no matches)
    if maxk > K:
        pad = idx_in_selected[..., -1:].expand(*idx_in_selected.shape[:-1], maxk - K)
        idx_in_selected = torch.cat([idx_in_selected, pad], dim=-1)

    # Positions -> class ids
    pred_classes = torch.gather(top_indices, dim=-1, index=idx_in_selected)

    N = targets.numel()
    preds_flat = pred_classes.reshape(N, maxk)
    targets_flat = targets.reshape(N)

    valid = (targets_flat != ignore_index) & (targets_flat >= 0)
    if valid.any():
        preds_flat = preds_flat[valid]
        targets_flat = targets_flat[valid]
        # (n_valid, maxk) == (n_valid, 1)
        eq = preds_flat == targets_flat.unsqueeze(-1)
        res = []
        n_valid = preds_flat.size(0)
        for k in topk:
            correct_k = eq[:, :k].any(dim=1).sum().item()
            if return_counts:
                res.append(float(correct_k))
            else:
                res.append(float(correct_k) / float(n_valid) if n_valid > 0 else 0.0)
        return tuple(res)
    else:
        if return_counts:
            return tuple(0.0 for _ in topk)
        else:
            return tuple(0.0 for _ in topk)


loss_fn_sum_reduction = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")


def accumulate(performance_dict: dict, metrics: dict, param_idx: int | None) -> None:
    """Add one batch's raw counts to the running totals (normalized by `average_performance_metrics`).

    With `param_idx`, each metric is a list indexed by params, grown as needed.
    """
    for name, value in metrics.items():
        if param_idx is None:
            performance_dict[name] = performance_dict.get(name, 0) + value
            continue

        totals = performance_dict.setdefault(name, [])
        if len(totals) <= param_idx:
            totals.extend([0] * (param_idx - len(totals) + 1))
        totals[param_idx] += value


# Old private name, kept so existing custom metrics keep working
_accumulate = accumulate


def update_performance_metrics(
    logits,
    targets,
    performance_dict,
    mode="classification",
    param_idx=None,
    return_counts=False,
    top_indices=None,
):
    """Accumulate one batch's metrics into `performance_dict`.

    Parameters:
    mode (str): "classification" (top1/top5), "next_token" (next-token top1/top5), or
        "perplexity" (summed cross_entropy); older names are accepted (`resolve_eval_mode`).
    param_idx (int | None): Accumulate into per-params lists; None for scalars.
    return_counts (bool): Raw counts instead of fractions.
    top_indices (Tensor | None): Candidate class ids `logits` was gathered at (see `accuracy`).
    """
    mode = resolve_eval_mode(mode)

    # classification: top-k accuracy
    if mode == "classification":
        metrics_to_calc = {"top1": 1, "top5": 5}
        topk_values = tuple(metrics_to_calc[k] for k in metrics_to_calc.keys())

        metric_results = accuracy(
            logits, targets, topk=topk_values, return_counts=return_counts, top_indices=top_indices
        )
        if not isinstance(metric_results, (list, tuple)):
            metric_results = [metric_results]
        else:
            metric_results = list(metric_results)

        count = targets.size(0)

    # perplexity: cross-entropy (-> perplexity)
    elif mode == "perplexity":
        metrics_to_calc = {"cross_entropy": None}

        # Unwrap ModelOutput / dict / tuple / tensor to raw logits
        hf_loss = None
        if hasattr(logits, "loss"):
            hf_loss = logits.loss
        if hasattr(logits, "logits"):
            raw_logits = logits.logits

        elif isinstance(logits, dict):
            if "loss" in logits:
                hf_loss = logits["loss"]
            if "logits" in logits:
                raw_logits = logits["logits"]

        elif isinstance(logits, (tuple, list)) and len(logits) > 0:
            raw_logits = logits[0]
        elif isinstance(logits, torch.Tensor):
            raw_logits = logits
        else:
            raise TypeError(f"Unrecognized logits type: {type(logits)}")

        # Language modeling: logits [B, T, V], targets [B, T]
        if raw_logits.dim() == 3 and targets is not None and targets.dim() >= 2:
            shifted_targets = targets[:, 1:].contiguous()
            count = int((shifted_targets != -100).sum().item())

            if hf_loss is not None:
                # HF loss is a mean over valid tokens -> sum
                loss_sum = float(hf_loss) * count
            else:
                # (B, V, T-1) input avoids flattening a huge tensor
                loss_sum = loss_fn_sum_reduction(
                    raw_logits[:, :-1, :].transpose(1, 2), shifted_targets
                ).item()

            metric_results = [loss_sum]

        # Classification fallback: logits [B, C], targets [B]
        else:
            batch_count = targets.size(0)
            if hf_loss is not None:
                loss_sum = float(hf_loss) * batch_count
            else:
                loss_sum = nn.functional.cross_entropy(raw_logits, targets, reduction="sum").item()
            count = batch_count
            metric_results = [loss_sum]

    # next_token: next-token top-k accuracy
    elif mode == "next_token":
        metrics_to_calc = {"top1": 1, "top5": 5}
        topk_values = tuple(metrics_to_calc[k] for k in metrics_to_calc.keys())

        # Unwrap ModelOutput / dict / tuple / tensor to raw logits
        if hasattr(logits, "logits"):
            raw_logits = logits.logits
        elif isinstance(logits, dict) and "logits" in logits:
            raw_logits = logits["logits"]
        elif isinstance(logits, (tuple, list)) and len(logits) > 0:
            raw_logits = logits[0]
        elif isinstance(logits, torch.Tensor):
            raw_logits = logits
        else:
            raise TypeError(f"Unrecognized logits type for next_token: {type(logits)}")

        # Language modeling: logits [B, T, V], targets [B, T]
        if raw_logits.dim() == 3 and targets is not None and targets.dim() >= 2:
            # Next-token: logits[:, :-1] predict targets[:, 1:]
            shifted_targets = targets[:, 1:].contiguous()
            logits_aligned = raw_logits[:, :-1, :]  # (B, T-1, V)

            B, Tp1, V = logits_aligned.shape
            # Flatten to (N, V), (N,); -100 is masked in `accuracy`
            logits_flat = logits_aligned.reshape(-1, V)
            targets_flat = shifted_targets.reshape(-1)
            if top_indices is not None:
                top_indices = top_indices[:, :-1, :].contiguous()
                top_indices = top_indices.reshape(logits_flat.shape[0], -1)

            count = int(targets_flat.numel())
            if count == 0:
                metric_results = [0.0 for _ in topk_values]
            else:
                metric_results = list(
                    accuracy(
                        logits_flat,
                        targets_flat,
                        topk=topk_values,
                        return_counts=return_counts,
                        top_indices=top_indices,
                    )
                )

        # Classification fallback: logits [B, C], targets [B]
        elif raw_logits.dim() == 2 and targets is not None and targets.dim() == 1:
            count = targets.size(0)
            metric_results = list(
                accuracy(
                    raw_logits,
                    targets,
                    topk=topk_values,
                    return_counts=return_counts,
                    top_indices=top_indices,
                )
            )
        else:
            raise ValueError(
                f"next_token expects logits of shape (B,T,V) with targets (B,T) or (B,C) with targets (B,), "
                f"got logits {tuple(raw_logits.shape)} and targets {tuple(targets.shape) if isinstance(targets, torch.Tensor) else type(targets)}"
            )

    else:
        raise ValueError(f"Unknown evaluation mode: {mode}, supports {EVAL_MODES}")

    # `count`: denominator for `average_performance_metrics`
    metric_results.append(count)
    accumulate(
        performance_dict, dict(zip(list(metrics_to_calc) + ["count"], metric_results)), param_idx
    )


def predicted_class(outputs: torch.Tensor, top_indices: torch.Tensor = None) -> torch.Tensor:
    """Return predicted class ids (argmax), mapped through `top_indices` if given.

    Ideal outputs are cached gathered at their own top-k, so the ideal argmax is
    always inside it and is recovered exactly.
    """
    best = outputs.argmax(dim=-1, keepdim=True)
    if top_indices is None:
        return best.squeeze(-1)
    return top_indices.gather(-1, best).squeeze(-1)


def update_flip_metrics(
    ideal_top1,
    noisy_top1,
    targets,
    performance_dict,
    mode="classification",
    param_idx=None,
    ignore_index=-100,
):
    """Accumulate top-1 flips: positions where noise changed the prediction.

    A flip needs any two logits to swap; an accuracy change needs the correct one.
    So flips show up at lower noise and with fewer samples: a usable calibration
    signal before accuracy visibly moves.

    Model vs. itself (not labels), so not part of the swappable `metric_fn`.
    Accumulates no `count`: shares `update_performance_metrics`'s denominator, so
    `top1_flip` is comparable to `top1`. Call alongside it, never instead.
    """
    if resolve_eval_mode(mode) != "classification":
        # Next-token alignment, as in `update_performance_metrics`
        ideal_top1, noisy_top1 = ideal_top1[..., :-1], noisy_top1[..., :-1]
        targets = targets[..., 1:] if targets is not None else None

    flipped = ideal_top1 != noisy_top1
    if targets is not None and targets.shape == flipped.shape:
        flipped = flipped & (targets != ignore_index)

    accumulate(performance_dict, {"top1_flip": float(flipped.sum().item())}, param_idx)


def average_performance_metrics(performance_dict):
    """Return totals divided by `count`, plus `perplexity` if `cross_entropy` is present."""
    averaged_metrics = {}
    for metric_name, total in performance_dict.items():
        if metric_name != "count":
            if isinstance(total, list):
                averaged_metrics[metric_name] = [
                    float(x) / count if count != 0 else 0
                    for x, count in zip(total, performance_dict["count"])
                ]
            else:
                count = performance_dict["count"]
                count = 1 if count == 0 else count
                averaged_metrics[metric_name] = float(total) / count

            if metric_name == "cross_entropy":
                averaged_metrics["perplexity"] = np.exp(
                    np.array(averaged_metrics[metric_name])
                ).tolist()
    return averaged_metrics
