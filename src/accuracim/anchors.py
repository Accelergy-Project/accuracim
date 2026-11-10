"""Adaptive anchor search: find the low and high accuracy-loss anchors in few calibration runs.

Calibration runs share a noise seed. Thus accuracy loss increases with noise, close to a
power law. The search brackets each target in log(sigma). When two losses are measurable,
a log-log fit predicts the target. Otherwise, the search bisects the bracket.
An isotonic fit of all probes makes the decisions, so one noisy small-N run cannot change them.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

LOW_LOSS = 0.005  # low anchor: largest noise with loss <= 0.5%
HIGH_LOSS = 0.05  # high anchor: largest noise with loss <= 5%, ...
HIGH_BAND = (0.04, 0.06)  # ... or a probe in this band, closest to HIGH_LOSS
NOISE_TOL = 1.25  # sigma ratio within which noises count as equivalent
FIT_RANGE = (
    0.001,
    0.3,
)  # losses usable for the log-log fit (above the 1-sample floor, below saturation)

_LOWER_IS_BETTER = ("cross_entropy", "perplexity")


def relative_loss(curr_perf: dict, best_perf: dict) -> float:
    """Return the worst relative degradation over non-flip metrics (direction-aware)."""
    losses = []
    for category, best in best_perf.items():
        if "flip" in category or category not in curr_perf:
            continue
        best_value, curr_value = best[0], curr_perf[category][0]
        if best_value == 0:
            continue
        if category in _LOWER_IS_BETTER:
            losses.append(curr_value / best_value - 1)
        else:
            losses.append(1 - curr_value / best_value)
    return max(losses) if losses else 0.0


def _isotonic(values) -> np.ndarray:
    """Return the least-squares non-decreasing fit of `values` (pool adjacent violators)."""
    blocks = []  # [sum, count]
    for v in values:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    return np.array([s / c for s, c in blocks for _ in range(c)])


def search_anchors(
    candidates: list[int],
    estimated_noises,
    evaluate: callable,
    min_idx: int,
    min_noise_perf: dict,
    budget: int | None = None,
) -> dict:
    """Find the low (<= LOW_LOSS) and high (~HIGH_LOSS) anchors with few `evaluate` calls.

    Parameters:
    candidates (list[int]): Params indices with unique positive estimated noise, ascending; `min_idx` excluded.
    estimated_noises (array): Estimated noise (variance) per params index.
    evaluate (callable): idx -> (measured_noise, perf), one calibration run.
    min_idx (int), min_noise_perf (dict): Reference point; loss is relative to its perf.
    budget (int | None): Max `evaluate` calls (default: all candidates).

    Returns:
    dict: `valid_min_*` / `valid_max_*` (idx, perf, noise), as `estimate.get_valid_indices` builds.
    """
    n = len(candidates)
    budget = n if budget is None else min(int(budget), n)
    u = 0.5 * np.log(np.asarray(estimated_noises, dtype=np.float64)[list(candidates)])  # log sigma
    tol = np.log(NOISE_TOL)
    probes = {}  # position -> (measured_noise, perf, loss)

    def probe(pos):
        noise, perf = evaluate(candidates[pos])
        probes[pos] = (noise, perf, relative_loss(perf, min_noise_perf))
        logger.info(
            f"anchor search · probe {len(probes)}/{budget} · idx {candidates[pos]} · "
            f"strength {estimated_noises[candidates[pos]]:.4g} · loss {probes[pos][2] * 100:.2f}%"
        )

    def smoothed():
        order = sorted(probes)
        return dict(zip(order, _isotonic([probes[p][2] for p in order])))

    def bracket(target, s):
        """(ok, bad): last probe with smoothed loss <= target (-1: none), first above (n: none)."""
        ok, bad = -1, n
        for p in sorted(s):
            if s[p] <= target:
                ok = p
            elif bad == n:
                bad = p
        return ok, bad

    def fit():
        """Return (a, b) of log(loss) = a + b * log(sigma), or None if not measurable yet."""
        pts = [
            (u[p], np.log(l))
            for p, (_, _, l) in probes.items()
            if FIT_RANGE[0] <= l <= FIT_RANGE[1]
        ]
        if len({x for x, _ in pts}) < 2:
            return None
        b, a = np.polyfit(*zip(*pts), 1)
        return (a, b) if b > 0 else None

    def width(ok, bad):
        return (u[bad] if bad < n else u[-1]) - (u[ok] if ok >= 0 else u[0])

    def search(target, band=None):
        """Probe until `target` is bracketed to NOISE_TOL; return the anchor position (-1: reference)."""
        bisect_next = False
        while True:
            s = smoothed()
            if band is not None:
                in_band = [p for p in s if band[0] <= s[p] <= band[1]]
                if in_band:
                    return min(in_band, key=lambda p: abs(s[p] - target))
            ok, bad = bracket(target, s)
            inside = [p for p in range(ok + 1, bad) if p not in probes]
            if not inside or len(probes) >= budget:
                return ok
            if ok >= 0 and bad < n and u[bad] - u[ok] <= tol:
                return ok
            if ok >= 0 and bad == n and u[-1] - u[ok] <= tol:
                return ok

            left = u[ok] if ok >= 0 else u[0]
            right = u[bad] if bad < n else u[-1]
            goal, guided = 0.5 * (left + right), False
            ab = None if bisect_next else fit()
            if ab is not None:
                predicted = (np.log(target) - ab[0]) / ab[1]
                if left < predicted < right:
                    goal, guided = predicted, True
            old_width = width(ok, bad)
            probe(min(inside, key=lambda p: abs(u[p] - goal)))
            # A fit-guided probe that did not halve the bracket: bisect next
            bisect_next = guided and width(*bracket(target, smoothed())) > 0.5 * old_width

    def anchor(pos):
        if pos < 0:
            return min_idx, min_noise_perf, 0
        noise, perf, _ = probes[pos]
        return candidates[pos], perf, noise

    if n == 0 or budget == 0:
        hi = lo = -1
    else:
        probe(int(np.argmin(np.abs(u - 0.5 * (u[0] + u[-1])))))  # center of the range in log(sigma)
        hi = search(HIGH_LOSS, band=HIGH_BAND)
        lo = search(LOW_LOSS)
        lo = min(lo, hi)

    valid_indices = {}
    for prefix, pos in (("valid_min", lo), ("valid_max", hi)):
        idx, perf, noise = anchor(pos)
        valid_indices.update(
            {f"{prefix}_idx": idx, f"{prefix}_perf": perf, f"{prefix}_noise": noise}
        )

    s = smoothed()
    ab = fit()
    summary = [f"{len(probes)}/{n} candidates probed"]
    if lo >= 0 and hi >= 0:
        summary.append(f"sigma ratio high/low {np.exp(u[hi] - u[lo]):.2f}")
    if ab is not None:
        summary.append(f"log-log slope {ab[1]:.2f}")
    logger.info("anchor search · " + " · ".join(summary))
    anchors = []
    for name, pos in (("low", lo), ("high", hi)):
        loss = s[pos] * 100 if pos >= 0 else 0.0
        anchors.append(f"{name} idx {anchor(pos)[0]} (loss {loss:.2f}%)")
    logger.info("anchors · " + " · ".join(anchors))
    return valid_indices
