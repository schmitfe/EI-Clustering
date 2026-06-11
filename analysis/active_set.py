from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class ActiveSetEMResult:
    masks: np.ndarray
    K: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    var0: np.ndarray
    var1: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    margin: np.ndarray


@dataclass
class PopulationStateDetection:
    result: ActiveSetEMResult
    segments: list[tuple[int, int]]
    X: np.ndarray
    L: np.ndarray
    preprocessing: Dict[str, Any]
    labels: np.ndarray
    time_masks: np.ndarray
    episodes: list[Dict[str, Any]]
    cluster_labels: np.ndarray
    cluster_occupancy: np.ndarray
    status: str = "ok"
    atomic_segments: Optional[list[tuple[int, int]]] = None
    atomic_masks: Optional[np.ndarray] = None
    merge_metadata: Optional[Dict[str, Any]] = None


def robust_run_scale(Y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(Y, dtype=float)
    offset = float(np.nanmedian(arr))
    scale = float(np.nanquantile(arr, 0.95) - np.nanquantile(arr, 0.05))
    if not np.isfinite(scale) or scale < float(eps):
        scale = 1.0
    return (arr - offset) / scale, offset, scale


def fixed_segments(T: int, width: int) -> list[tuple[int, int]]:
    segment_width = int(width)
    if segment_width <= 0:
        raise ValueError("fixed_width must be positive.")
    return [(idx, min(idx + segment_width, int(T))) for idx in range(0, int(T), segment_width)]


def get_segments_pelt(Y_scaled: np.ndarray, penalty: float = 10.0, min_size: int = 5) -> list[tuple[int, int]]:
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime path
        raise ModuleNotFoundError("ruptures is required for active_set_em PELT segmentation.") from exc
    arr = np.asarray(Y_scaled, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Y_scaled must be a 2D array.")
    if arr.shape[0] == 0:
        return []
    bkps = rpt.Pelt(model="l2", min_size=int(min_size)).fit(arr).predict(pen=float(penalty))
    starts = [0] + [int(value) for value in bkps[:-1]]
    stops = [int(value) for value in bkps]
    return list(zip(starts, stops))


def estimate_cluster_weights(Y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-cluster reliability for changepoint features without z-scoring."""

    arr = np.asarray(Y, dtype=float)
    q05 = np.nanquantile(arr, 0.05, axis=0)
    q95 = np.nanquantile(arr, 0.95, axis=0)
    robust_range = q95 - q05
    if arr.shape[0] > 1:
        diff = np.diff(arr, axis=0)
        center = np.nanmedian(diff, axis=0)
        noise = 1.4826 * np.nanmedian(np.abs(diff - center[None, :]), axis=0) / np.sqrt(2.0)
    else:
        noise = np.zeros(arr.shape[1], dtype=float)
    signal = np.maximum(robust_range - 2.0 * noise, 0.0)
    weight = signal**2 / (signal**2 + noise**2 + float(eps))
    weight = np.clip(np.where(np.isfinite(weight), weight, 0.0), 0.0, 1.0)
    return weight, robust_range, noise


def make_pelt_features(Y: np.ndarray, smooth_width: int = 3) -> tuple[np.ndarray, Dict[str, np.ndarray | float]]:
    """Build denoised, globally scaled PELT features with flat clusters downweighted."""

    arr = np.asarray(Y, dtype=float)
    width = int(smooth_width)
    if width > 1:
        kernel = np.ones(width, dtype=float) / float(width)
        smoothed = np.vstack([np.convolve(arr[:, col], kernel, mode="same") for col in range(arr.shape[1])]).T
    else:
        smoothed = arr.copy()
    weight, robust_range, noise = estimate_cluster_weights(smoothed)
    offset = float(np.nanmedian(smoothed))
    scale = float(np.nanquantile(smoothed, 0.95) - np.nanquantile(smoothed, 0.05))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    features = (smoothed - offset) / scale
    features = features * np.sqrt(weight)[None, :]
    return features, {
        "pelt_offset": offset,
        "pelt_scale": scale,
        "cluster_weight": weight,
        "robust_range": robust_range,
        "noise": noise,
    }


def segment_means(Y: np.ndarray, segments: Iterable[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Y must be a 2D array.")
    means: list[np.ndarray] = []
    lengths: list[int] = []
    for start, stop in segments:
        start_i = int(start)
        stop_i = int(stop)
        if stop_i <= start_i:
            continue
        means.append(np.nanmean(arr[start_i:stop_i], axis=0))
        lengths.append(stop_i - start_i)
    if not means:
        return np.zeros((0, arr.shape[1]), dtype=float), np.zeros(0, dtype=float)
    X = np.vstack(means)
    if not np.isfinite(X).all():
        col_fill = np.nanmedian(X, axis=0)
        global_fill = float(np.nanmedian(X)) if np.isfinite(np.nanmedian(X)) else 0.0
        col_fill = np.where(np.isfinite(col_fill), col_fill, global_fill)
        rows, cols = np.where(~np.isfinite(X))
        X[rows, cols] = col_fill[cols]
    return X, np.asarray(lengths, dtype=float)


def gaussian_nll(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return 0.5 * ((x - float(mu)) ** 2 / float(var) + np.log(float(var)))


def log_comb(n: int, k: int) -> float:
    return math.lgamma(int(n) + 1) - math.lgamma(int(k) + 1) - math.lgamma(int(n) - int(k) + 1)


def initialize_masks_threshold(X: np.ndarray, Kmax: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    M, Q = arr.shape
    max_active = Q if Kmax is None else min(int(Kmax), Q)
    lo = float(np.nanquantile(arr, 0.25))
    hi = float(np.nanquantile(arr, 0.75))
    theta = 0.5 * (lo + hi)
    masks = arr > theta
    if max_active < Q:
        for row in range(M):
            if int(masks[row].sum()) > max_active:
                idx = np.argsort(arr[row])[-max_active:]
                masks[row, :] = False
                masks[row, idx] = True
    return masks


def initialize_masks_gap(X: np.ndarray, Kmax: Optional[int] = None, min_gap: float = 0.0) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    M, Q = arr.shape
    max_active = Q if Kmax is None else min(int(Kmax), Q)
    masks = np.zeros((M, Q), dtype=bool)
    for row in range(M):
        order = np.argsort(arr[row])[::-1]
        sorted_values = arr[row, order]
        best_k = 0
        best_gap = float(min_gap)
        for k in range(1, min(max_active, Q - 1) + 1):
            gap = float(sorted_values[k - 1] - sorted_values[k])
            if gap > best_gap:
                best_gap = gap
                best_k = k
        if best_k > 0:
            masks[row, order[:best_k]] = True
    return masks


def estimate_params(
    X: np.ndarray,
    L: np.ndarray,
    masks: np.ndarray,
    Kmax: int,
    var_floor: float = 1e-4,
    old_params: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    lengths = np.asarray(L, dtype=float)
    active_masks = np.asarray(masks, dtype=bool)
    K = active_masks.sum(axis=1)
    mu0 = np.full(int(Kmax) + 1, np.nan)
    mu1 = np.full(int(Kmax) + 1, np.nan)
    var0 = np.full(int(Kmax) + 1, np.nan)
    var1 = np.full(int(Kmax) + 1, np.nan)
    for k in range(int(Kmax) + 1):
        seg_idx = np.where(K == k)[0]
        active_vals: list[np.ndarray] = []
        active_w: list[np.ndarray] = []
        inactive_vals: list[np.ndarray] = []
        inactive_w: list[np.ndarray] = []
        for row in seg_idx:
            active = active_masks[row]
            inactive = ~active
            if np.any(active):
                active_vals.append(arr[row, active])
                active_w.append(np.full(int(active.sum()), lengths[row]))
            if np.any(inactive):
                inactive_vals.append(arr[row, inactive])
                inactive_w.append(np.full(int(inactive.sum()), lengths[row]))
        if active_vals:
            vals = np.concatenate(active_vals)
            weights = np.concatenate(active_w)
            mu = float(np.average(vals, weights=weights))
            mu1[k] = mu
            var1[k] = max(float(np.average((vals - mu) ** 2, weights=weights)), float(var_floor))
        if inactive_vals:
            vals = np.concatenate(inactive_vals)
            weights = np.concatenate(inactive_w)
            mu = float(np.average(vals, weights=weights))
            mu0[k] = mu
            var0[k] = max(float(np.average((vals - mu) ** 2, weights=weights)), float(var_floor))
    global_mu = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
    global_var = max(float(np.nanvar(arr)) if np.isfinite(np.nanvar(arr)) else float(var_floor), float(var_floor))
    for k in range(int(Kmax) + 1):
        if np.isnan(mu0[k]):
            if old_params is not None and np.isfinite(old_params["mu0"][k]):
                mu0[k] = old_params["mu0"][k]
                var0[k] = old_params["var0"][k]
            else:
                mu0[k] = global_mu
                var0[k] = global_var
        if np.isnan(mu1[k]):
            if old_params is not None and np.isfinite(old_params["mu1"][k]):
                mu1[k] = old_params["mu1"][k]
                var1[k] = old_params["var1"][k]
            else:
                mu1[k] = global_mu
                var1[k] = global_var
    return mu0, mu1, var0, var1


def e_step(
    X: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    var0: np.ndarray,
    var1: np.ndarray,
    Kmax: int,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    M, Q = arr.shape
    max_active = min(int(Kmax), Q)
    masks = np.zeros((M, Q), dtype=bool)
    K_best = np.zeros(M, dtype=int)
    margins = np.zeros(M, dtype=float)
    objective = 0.0
    penalties = np.array(
        [float(lambda_active) * k + float(lambda_comb) * log_comb(Q, k) for k in range(max_active + 1)],
        dtype=float,
    )
    for row in range(M):
        x = arr[row]
        costs = np.full(max_active + 1, np.inf)
        candidate_masks = np.zeros((max_active + 1, Q), dtype=bool)
        for k in range(max_active + 1):
            if k == 0:
                low_loss = gaussian_nll(x, mu0[k], var0[k])
                costs[k] = float(np.sum(low_loss) + penalties[k])
            elif k == Q:
                high_loss = gaussian_nll(x, mu1[k], var1[k])
                costs[k] = float(np.sum(high_loss) + penalties[k])
                candidate_masks[k, :] = True
            else:
                low_loss = gaussian_nll(x, mu0[k], var0[k])
                high_loss = gaussian_nll(x, mu1[k], var1[k])
                delta = high_loss - low_loss
                active_idx = np.argpartition(delta, k)[:k]
                mask = np.zeros(Q, dtype=bool)
                mask[active_idx] = True
                if float(np.mean(x[mask]) - np.mean(x[~mask])) < float(min_separation):
                    continue
                costs[k] = float(np.sum(low_loss) + np.sum(delta[mask]) + penalties[k])
                candidate_masks[k] = mask
        order = np.argsort(costs)
        best = int(order[0])
        second = int(order[1]) if order.size > 1 else best
        masks[row] = candidate_masks[best]
        K_best[row] = best
        margins[row] = float(costs[second] - costs[best])
        objective += float(costs[best])
    return masks, K_best, objective, margins


def active_set_em(
    X: np.ndarray,
    L: Optional[np.ndarray] = None,
    Kmax: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
    var_floor: float = 1e-4,
    init_masks: Optional[np.ndarray] = None,
) -> ActiveSetEMResult:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    M, Q = arr.shape
    lengths = np.ones(M, dtype=float) if L is None else np.asarray(L, dtype=float).ravel()
    if lengths.shape[0] != M:
        raise ValueError("L must have one entry per segment.")
    max_active = Q if Kmax is None else min(int(Kmax), Q)
    masks = initialize_masks_threshold(arr, Kmax=max_active) if init_masks is None else np.asarray(init_masks, dtype=bool).copy()
    if masks.shape != arr.shape:
        raise ValueError("init_masks must have the same shape as X.")
    if max_active < Q:
        for row in range(M):
            if int(masks[row].sum()) > max_active:
                idx = np.argsort(arr[row])[-max_active:]
                masks[row, :] = False
                masks[row, idx] = True
    params: Optional[Dict[str, np.ndarray]] = None
    previous_objective: Optional[float] = None
    converged = False
    objective = float("inf")
    K = masks.sum(axis=1).astype(int)
    margin = np.zeros(M, dtype=float)
    mu0 = mu1 = var0 = var1 = np.zeros(max_active + 1, dtype=float)
    for iteration in range(int(max_iter)):
        mu0, mu1, var0, var1 = estimate_params(
            arr,
            lengths,
            masks,
            Kmax=max_active,
            var_floor=float(var_floor),
            old_params=params,
        )
        params = {"mu0": mu0, "mu1": mu1, "var0": var0, "var1": var1}
        new_masks, K, objective, margin = e_step(
            arr,
            mu0,
            mu1,
            var0,
            var1,
            Kmax=max_active,
            lambda_active=float(lambda_active),
            lambda_comb=float(lambda_comb),
            min_separation=float(min_separation),
        )
        assignment_changed = bool(np.any(new_masks != masks))
        if previous_objective is not None:
            rel_improvement = abs(previous_objective - objective) / max(abs(previous_objective), 1.0)
            if not assignment_changed or rel_improvement < float(tol):
                masks = new_masks
                converged = True
                break
        elif not assignment_changed:
            masks = new_masks
            converged = True
            break
        masks = new_masks
        previous_objective = float(objective)
    else:
        iteration = int(max_iter) - 1
    return ActiveSetEMResult(
        masks=masks,
        K=np.asarray(K, dtype=int),
        mu0=np.asarray(mu0, dtype=float),
        mu1=np.asarray(mu1, dtype=float),
        var0=np.asarray(var0, dtype=float),
        var1=np.asarray(var1, dtype=float),
        objective=float(objective),
        n_iter=int(iteration) + 1,
        converged=bool(converged),
        margin=np.asarray(margin, dtype=float),
    )


def active_set_em_multi_init(X: np.ndarray, L: Optional[np.ndarray] = None, Kmax: Optional[int] = None, **kwargs: Any) -> ActiveSetEMResult:
    min_gap = float(kwargs.get("min_separation", 0.0))
    inits = [
        initialize_masks_threshold(X, Kmax=Kmax),
        initialize_masks_gap(X, Kmax=Kmax, min_gap=min_gap),
    ]
    results = [active_set_em(X, L=L, Kmax=Kmax, init_masks=init, **kwargs) for init in inits]
    return min(results, key=lambda item: item.objective)


def merge_identical_adjacent(
    segments: list[tuple[int, int]],
    masks: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    active_masks = np.asarray(masks, dtype=bool)
    if len(segments) == 0:
        return [], active_masks.reshape((0, active_masks.shape[-1] if active_masks.ndim == 2 else 0))
    merged_segments: list[tuple[int, int]] = []
    merged_masks: list[np.ndarray] = []
    current_start, current_stop = segments[0]
    current_mask = active_masks[0].copy()
    for (start, stop), mask in zip(segments[1:], active_masks[1:]):
        if np.array_equal(mask, current_mask):
            current_stop = int(stop)
        else:
            merged_segments.append((int(current_start), int(current_stop)))
            merged_masks.append(current_mask)
            current_start, current_stop = int(start), int(stop)
            current_mask = mask.copy()
    merged_segments.append((int(current_start), int(current_stop)))
    merged_masks.append(current_mask)
    return merged_segments, np.asarray(merged_masks, dtype=bool)


def mask_cost_for_mean(
    x: np.ndarray,
    L: float,
    mask: np.ndarray,
    params: Dict[str, np.ndarray],
    *,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
) -> float:
    active = np.asarray(mask, dtype=bool)
    K = int(active.sum())
    low_loss = gaussian_nll(np.asarray(x, dtype=float), params["mu0"][K], params["var0"][K])
    high_loss = gaussian_nll(np.asarray(x, dtype=float), params["mu1"][K], params["var1"][K])
    penalty = float(lambda_active) * K + float(lambda_comb) * log_comb(active.size, K)
    return float(L) * float(np.sum(np.where(active, high_loss, low_loss)) + penalty)


def segment_cost_from_bins(
    Y: np.ndarray,
    segment: tuple[int, int],
    mask: np.ndarray,
    params: Dict[str, np.ndarray],
    *,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
) -> float:
    start, stop = int(segment[0]), int(segment[1])
    x = np.nanmean(np.asarray(Y, dtype=float)[start:stop], axis=0)
    return mask_cost_for_mean(
        x,
        float(stop - start),
        mask,
        params,
        lambda_active=lambda_active,
        lambda_comb=lambda_comb,
    )


def best_mask_for_segment_mean(
    x: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=float)
    Q = arr.shape[0]
    max_active = min(int(Kmax), Q)
    best_cost = np.inf
    best_mask = np.zeros(Q, dtype=bool)
    for K in range(max_active + 1):
        low_loss = gaussian_nll(arr, params["mu0"][K], params["var0"][K])
        high_loss = gaussian_nll(arr, params["mu1"][K], params["var1"][K])
        if K == 0:
            mask = np.zeros(Q, dtype=bool)
            cost = float(np.sum(low_loss))
        elif K == Q:
            mask = np.ones(Q, dtype=bool)
            cost = float(np.sum(high_loss))
        else:
            delta = high_loss - low_loss
            active_idx = np.argpartition(delta, K)[:K]
            mask = np.zeros(Q, dtype=bool)
            mask[active_idx] = True
            if float(np.mean(arr[mask]) - np.mean(arr[~mask])) < float(min_separation):
                continue
            cost = float(np.sum(low_loss) + np.sum(delta[mask]))
        cost += float(lambda_active) * K + float(lambda_comb) * log_comb(Q, K)
        if cost < best_cost:
            best_cost = cost
            best_mask = mask
    return best_mask, float(best_cost)


def cost_mask_for_segment_mean(
    x: np.ndarray,
    mask: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
) -> float:
    active = np.asarray(mask, dtype=bool)
    K = int(active.sum())
    if K > int(Kmax):
        return float("inf")
    low_loss = gaussian_nll(np.asarray(x, dtype=float), params["mu0"][K], params["var0"][K])
    high_loss = gaussian_nll(np.asarray(x, dtype=float), params["mu1"][K], params["var1"][K])
    return float(np.sum(np.where(active, high_loss, low_loss)) + float(lambda_active) * K + float(lambda_comb) * log_comb(active.size, K))


def top_candidate_masks_for_segment(
    x: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    n_candidates: int = 5,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
) -> list[tuple[np.ndarray, float]]:
    candidates: list[tuple[np.ndarray, float]] = []
    Q = np.asarray(x).shape[0]
    arr = np.asarray(x, dtype=float)

    def add_candidate(mask: np.ndarray, K: int, low_loss: np.ndarray, high_loss: np.ndarray) -> None:
        if K not in {0, Q}:
            if not np.any(mask) or not np.any(~mask):
                return
            if float(np.mean(arr[mask]) - np.mean(arr[~mask])) < float(min_separation):
                return
        cost = float(np.sum(np.where(mask, high_loss, low_loss)))
        cost += float(lambda_active) * K + float(lambda_comb) * log_comb(Q, K)
        candidates.append((mask.copy(), cost))

    for K in range(min(int(Kmax), Q) + 1):
        low_loss = gaussian_nll(arr, params["mu0"][K], params["var0"][K])
        high_loss = gaussian_nll(arr, params["mu1"][K], params["var1"][K])
        if K == 0:
            mask = np.zeros(Q, dtype=bool)
            add_candidate(mask, K, low_loss, high_loss)
        elif K == Q:
            mask = np.ones(Q, dtype=bool)
            add_candidate(mask, K, low_loss, high_loss)
        else:
            delta = high_loss - low_loss
            active_idx = np.argsort(delta)[:K]
            mask = np.zeros(Q, dtype=bool)
            mask[active_idx] = True
            add_candidate(mask, K, low_loss, high_loss)
            order = np.argsort(delta)
            fixed_active = order[: max(K - 1, 0)]
            for replacement in order[K : min(Q, K + int(n_candidates))]:
                alt_mask = np.zeros(Q, dtype=bool)
                alt_mask[fixed_active] = True
                alt_mask[replacement] = True
                if int(alt_mask.sum()) == K:
                    add_candidate(alt_mask, K, low_loss, high_loss)
    candidates = sorted(candidates, key=lambda item: item[1])
    unique: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[bool, ...]] = set()
    for mask, cost in candidates:
        key = tuple(bool(value) for value in mask.tolist())
        if key in seen:
            continue
        unique.append((mask.copy(), float(cost)))
        seen.add(key)
        if len(unique) >= int(n_candidates):
            break
    return unique


def build_candidate_sets(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    n_candidates: int = 5,
    base_masks: Optional[np.ndarray] = None,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
    include_all_low: bool = True,
) -> list[list[tuple[np.ndarray, float]]]:
    arr = np.asarray(X, dtype=float)
    M, Q = arr.shape
    candidate_sets = [
        top_candidate_masks_for_segment(
            arr[row],
            params,
            Kmax,
            n_candidates=n_candidates,
            lambda_active=lambda_active,
            lambda_comb=lambda_comb,
            min_separation=min_separation,
        )
        for row in range(M)
    ]

    def add_mask(row: int, mask: np.ndarray) -> None:
        active = np.asarray(mask, dtype=bool)
        key = tuple(bool(value) for value in active.tolist())
        existing = {tuple(bool(value) for value in cand_mask.tolist()) for cand_mask, _ in candidate_sets[row]}
        if key in existing:
            return
        cost = cost_mask_for_segment_mean(
            arr[row],
            active,
            params,
            Kmax,
            lambda_active=lambda_active,
            lambda_comb=lambda_comb,
        )
        if np.isfinite(cost):
            candidate_sets[row].append((active.copy(), float(cost)))

    if base_masks is not None:
        base = np.asarray(base_masks, dtype=bool)
        for row in range(M):
            add_mask(row, base[row])
            if row > 0:
                add_mask(row, base[row - 1])
            if row < M - 1:
                add_mask(row, base[row + 1])
    if include_all_low:
        all_low = np.zeros(Q, dtype=bool)
        for row in range(M):
            add_mask(row, all_low)
    return candidate_sets


def hamming_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.sum(np.asarray(mask_a, dtype=bool) != np.asarray(mask_b, dtype=bool)))


def transition_cost(
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    *,
    gamma_switch: float = 5.0,
    gamma_hamming: float = 1.0,
) -> float:
    if np.array_equal(mask_prev, mask_curr):
        return 0.0
    return float(gamma_switch) + float(gamma_hamming) * hamming_distance(mask_prev, mask_curr)


def smooth_active_set_sequence_dp(
    candidate_sets: list[list[tuple[np.ndarray, float]]],
    L: Optional[np.ndarray] = None,
    *,
    gamma_switch: float = 5.0,
    gamma_hamming: float = 1.0,
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    M = len(candidate_sets)
    if M == 0:
        return np.zeros((0, 0), dtype=bool), 0.0, {"smooth_total_cost": 0.0}
    lengths = np.ones(M, dtype=float) if L is None else np.asarray(L, dtype=float).ravel()
    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []
    dp.append(np.asarray([lengths[0] * cost for _, cost in candidate_sets[0]], dtype=float))
    back.append(np.full(len(candidate_sets[0]), -1, dtype=int))
    for row in range(1, M):
        prev_candidates = candidate_sets[row - 1]
        curr_candidates = candidate_sets[row]
        curr_dp = np.full(len(curr_candidates), np.inf, dtype=float)
        curr_back = np.full(len(curr_candidates), -1, dtype=int)
        for j, (mask_j, cost_j) in enumerate(curr_candidates):
            emission = lengths[row] * float(cost_j)
            best_val = np.inf
            best_idx = -1
            for i, (mask_i, _cost_i) in enumerate(prev_candidates):
                value = dp[row - 1][i] + transition_cost(
                    mask_i,
                    mask_j,
                    gamma_switch=gamma_switch,
                    gamma_hamming=gamma_hamming,
                ) + emission
                if value < best_val:
                    best_val = float(value)
                    best_idx = i
            curr_dp[j] = best_val
            curr_back[j] = best_idx
        dp.append(curr_dp)
        back.append(curr_back)
    path_idx = np.zeros(M, dtype=int)
    path_idx[-1] = int(np.argmin(dp[-1]))
    for row in range(M - 1, 0, -1):
        path_idx[row - 1] = back[row][path_idx[row]]
    masks = np.asarray([candidate_sets[row][path_idx[row]][0] for row in range(M)], dtype=bool)
    total_cost = float(dp[-1][path_idx[-1]])
    metadata = {
        "smooth_total_cost": total_cost,
        "candidate_counts": [len(items) for items in candidate_sets],
        "selected_candidate_indices": path_idx.astype(int).tolist(),
        "gamma_switch": float(gamma_switch),
        "gamma_hamming": float(gamma_hamming),
    }
    return masks, total_cost, metadata


def boundary_evidence(
    X: np.ndarray,
    masks: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
) -> list[Dict[str, Any]]:
    arr = np.asarray(X, dtype=float)
    active_masks = np.asarray(masks, dtype=bool)
    evidence: list[Dict[str, Any]] = []
    for row in range(1, active_masks.shape[0]):
        prev_mask = active_masks[row - 1]
        curr_mask = active_masks[row]
        if np.array_equal(prev_mask, curr_mask):
            continue
        left_keep = cost_mask_for_segment_mean(arr[row - 1], prev_mask, params, Kmax, lambda_active=lambda_active, lambda_comb=lambda_comb)
        left_swap = cost_mask_for_segment_mean(arr[row - 1], curr_mask, params, Kmax, lambda_active=lambda_active, lambda_comb=lambda_comb)
        right_keep = cost_mask_for_segment_mean(arr[row], curr_mask, params, Kmax, lambda_active=lambda_active, lambda_comb=lambda_comb)
        right_swap = cost_mask_for_segment_mean(arr[row], prev_mask, params, Kmax, lambda_active=lambda_active, lambda_comb=lambda_comb)
        evidence.append(
            {
                "boundary": int(row),
                "left_evidence": float(left_swap - left_keep),
                "right_evidence": float(right_swap - right_keep),
                "min_evidence": float(min(left_swap - left_keep, right_swap - right_keep)),
                "hamming": hamming_distance(prev_mask, curr_mask),
            }
        )
    return evidence


def remove_short_aba_flickers(
    Y: np.ndarray,
    segments: list[tuple[int, int]],
    masks: np.ndarray,
    params: Dict[str, np.ndarray],
    *,
    min_duration: int = 3,
    max_hamming: int = 2,
    cost_tolerance: float = 10.0,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    active_masks = np.asarray(masks, dtype=bool)
    removed = 0
    changed = True
    while changed:
        changed = False
        new_segments: list[tuple[int, int]] = []
        new_masks: list[np.ndarray] = []
        idx = 0
        while idx < len(segments):
            if 0 < idx < len(segments) - 1:
                prev_mask = active_masks[idx - 1]
                curr_mask = active_masks[idx]
                next_mask = active_masks[idx + 1]
                duration = int(segments[idx][1] - segments[idx][0])
                is_aba = bool(np.array_equal(prev_mask, next_mask))
                hamming = int(np.sum(curr_mask != prev_mask))
                if is_aba and duration <= int(min_duration) and hamming <= int(max_hamming):
                    keep_cost = segment_cost_from_bins(
                        Y,
                        segments[idx],
                        curr_mask,
                        params,
                        lambda_active=lambda_active,
                        lambda_comb=lambda_comb,
                    )
                    absorb_cost = segment_cost_from_bins(
                        Y,
                        segments[idx],
                        prev_mask,
                        params,
                        lambda_active=lambda_active,
                        lambda_comb=lambda_comb,
                    )
                    if absorb_cost - keep_cost <= float(cost_tolerance) and new_segments:
                        prev_start = new_segments[-1][0]
                        next_stop = segments[idx + 1][1]
                        new_segments[-1] = (int(prev_start), int(next_stop))
                        new_masks[-1] = prev_mask.copy()
                        idx += 2
                        removed += 1
                        changed = True
                        continue
            new_segments.append((int(segments[idx][0]), int(segments[idx][1])))
            new_masks.append(active_masks[idx].copy())
            idx += 1
        segments = new_segments
        active_masks = np.asarray(new_masks, dtype=bool)
    return segments, active_masks, removed


def merge_unsupported_boundaries(
    Y: np.ndarray,
    segments: list[tuple[int, int]],
    masks: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    beta_merge: float = 10.0,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
    max_iter: int = 100,
) -> tuple[list[tuple[int, int]], np.ndarray, Dict[str, Any]]:
    current_segments = [(int(start), int(stop)) for start, stop in segments]
    current_masks = np.asarray(masks, dtype=bool)
    merge_count = 0
    gains: list[float] = []
    for _ in range(int(max_iter)):
        if len(current_segments) <= 1:
            break
        boundary_gains = []
        for idx in range(len(current_segments) - 1):
            seg_a = current_segments[idx]
            seg_b = current_segments[idx + 1]
            cost_split = segment_cost_from_bins(
                Y,
                seg_a,
                current_masks[idx],
                params,
                lambda_active=lambda_active,
                lambda_comb=lambda_comb,
            ) + segment_cost_from_bins(
                Y,
                seg_b,
                current_masks[idx + 1],
                params,
                lambda_active=lambda_active,
                lambda_comb=lambda_comb,
            )
            merged_seg = (seg_a[0], seg_b[1])
            x_merged = np.nanmean(np.asarray(Y, dtype=float)[merged_seg[0] : merged_seg[1]], axis=0)
            merged_mask, merged_cost_per_bin = best_mask_for_segment_mean(
                x_merged,
                params,
                Kmax=Kmax,
                lambda_active=lambda_active,
                lambda_comb=lambda_comb,
                min_separation=min_separation,
            )
            cost_merge = float(merged_seg[1] - merged_seg[0]) * merged_cost_per_bin
            gain = float(cost_merge - cost_split)
            boundary_gains.append((gain, idx, merged_seg, merged_mask))
        gain, idx, merged_seg, merged_mask = min(boundary_gains, key=lambda item: item[0])
        if gain >= float(beta_merge):
            break
        new_segments: list[tuple[int, int]] = []
        new_masks: list[np.ndarray] = []
        pos = 0
        while pos < len(current_segments):
            if pos == idx:
                new_segments.append(merged_seg)
                new_masks.append(merged_mask)
                pos += 2
            else:
                new_segments.append(current_segments[pos])
                new_masks.append(current_masks[pos].copy())
                pos += 1
        current_segments, current_masks = merge_identical_adjacent(new_segments, np.asarray(new_masks, dtype=bool))
        merge_count += 1
        gains.append(gain)
    return current_segments, current_masks, {"unsupported_merge_count": merge_count, "unsupported_merge_gains": gains}


def merge_active_set_segments(
    Y: np.ndarray,
    segments: list[tuple[int, int]],
    masks: np.ndarray,
    params: Dict[str, np.ndarray],
    Kmax: int,
    *,
    beta_merge: float = 10.0,
    min_flicker_duration: int = 3,
    flicker_max_hamming: int = 2,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.0,
    max_iter: int = 100,
) -> tuple[list[tuple[int, int]], np.ndarray, Dict[str, Any]]:
    atomic_count = len(segments)
    merged_segments, merged_masks = merge_identical_adjacent(segments, masks)
    identical_removed = atomic_count - len(merged_segments)
    merged_segments, merged_masks, flicker_removed = remove_short_aba_flickers(
        Y,
        merged_segments,
        merged_masks,
        params,
        min_duration=min_flicker_duration,
        max_hamming=flicker_max_hamming,
        cost_tolerance=beta_merge,
        lambda_active=lambda_active,
        lambda_comb=lambda_comb,
    )
    merged_segments, merged_masks, unsupported_metadata = merge_unsupported_boundaries(
        Y,
        merged_segments,
        merged_masks,
        params,
        Kmax,
        beta_merge=beta_merge,
        lambda_active=lambda_active,
        lambda_comb=lambda_comb,
        min_separation=min_separation,
        max_iter=max_iter,
    )
    final_segments, final_masks = merge_identical_adjacent(merged_segments, merged_masks)
    metadata = {
        "merge_after_em": True,
        "atomic_segment_count": int(atomic_count),
        "final_segment_count": int(len(final_segments)),
        "identical_merge_count": int(identical_removed),
        "flicker_merge_count": int(flicker_removed),
        "beta_merge": float(beta_merge),
        "min_flicker_duration": int(min_flicker_duration),
        "flicker_max_hamming": int(flicker_max_hamming),
    }
    metadata.update(unsupported_metadata)
    metadata["total_merge_count"] = int(atomic_count - len(final_segments))
    return final_segments, final_masks, metadata


def expand_segment_masks(masks: np.ndarray, segments: list[tuple[int, int]], T: int) -> np.ndarray:
    active_masks = np.asarray(masks, dtype=bool)
    out = np.zeros((int(T), active_masks.shape[1]), dtype=bool)
    for row, (start, stop) in enumerate(segments):
        out[int(start) : int(stop)] = active_masks[row]
    return out


def merge_state_episodes(masks: np.ndarray, segments: list[tuple[int, int]]) -> list[Dict[str, Any]]:
    active_masks = np.asarray(masks, dtype=bool)
    if active_masks.shape[0] == 0:
        return []
    episodes: list[Dict[str, Any]] = []
    current = active_masks[0].copy()
    start = int(segments[0][0])
    stop = int(segments[0][1])
    for row in range(1, len(segments)):
        if np.array_equal(active_masks[row], current):
            stop = int(segments[row][1])
            continue
        episodes.append({"start": start, "stop": stop, "mask": current.copy(), "K": int(current.sum())})
        current = active_masks[row].copy()
        start = int(segments[row][0])
        stop = int(segments[row][1])
    episodes.append({"start": start, "stop": stop, "mask": current.copy(), "K": int(current.sum())})
    return episodes


def classify_clusters(
    masks: np.ndarray,
    L: np.ndarray,
    low_tol: float = 0.01,
    high_tol: float = 0.99,
    min_transitions: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    active_masks = np.asarray(masks, dtype=bool)
    lengths = np.asarray(L, dtype=float).ravel()
    if active_masks.shape[0] == 0:
        return np.zeros(active_masks.shape[1], dtype=object), np.zeros(active_masks.shape[1], dtype=float)
    total = float(np.sum(lengths))
    weights = lengths / total if total > 0.0 else np.full(active_masks.shape[0], 1.0 / active_masks.shape[0])
    occupancy = np.sum(active_masks * weights[:, None], axis=0)
    labels = np.full(active_masks.shape[1], "dynamic", dtype=object)
    for col in range(active_masks.shape[1]):
        trace = active_masks[:, col].astype(int)
        transitions = int(np.sum(trace[1:] != trace[:-1]))
        if occupancy[col] <= float(low_tol):
            labels[col] = "observed_pinned_low"
        elif occupancy[col] >= float(high_tol):
            labels[col] = "observed_pinned_high"
        elif transitions < int(min_transitions):
            labels[col] = "ambiguous"
        else:
            labels[col] = "dynamic"
    return labels, np.asarray(occupancy, dtype=float)


def _labels_from_masks(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(masks, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("masks must be a 2D array.")
    patterns, inverse = np.unique(arr.astype(np.uint8), axis=0, return_inverse=True)
    return inverse.astype(np.int64), patterns.astype(np.uint8)


def _transform_activity(Y: np.ndarray, transform: str) -> tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(Y, dtype=float)
    name = str(transform or "identity").lower()
    if name == "identity":
        return arr, {"transform": "identity"}
    if name == "sqrt":
        return np.sqrt(np.maximum(arr, 0.0)), {"transform": "sqrt"}
    if name == "asinh":
        positive = arr[arr > 0.0]
        r0 = float(np.nanmedian(positive)) if positive.size else 1.0
        if not np.isfinite(r0) or r0 <= 0.0:
            r0 = 1.0
        return np.arcsinh(arr / r0), {"transform": "asinh", "r0": r0}
    raise ValueError(f"Unknown transform: {transform}")


def detect_population_states(
    Y: np.ndarray,
    transform: str = "identity",
    segmentation: str = "fixed",
    fixed_width: int = 10,
    pelt_penalty: float = 10.0,
    pelt_min_size: int = 5,
    pelt_feature_mode: str = "weighted",
    pelt_smooth_width: int = 3,
    Kmax: Optional[int] = None,
    lambda_active: float = 0.0,
    lambda_comb: float = 0.1,
    min_separation: float = 0.05,
    var_floor: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-6,
    flat_range_threshold: float = 1e-12,
    low_tol: float = 0.01,
    high_tol: float = 0.99,
    min_transitions: int = 1,
    merge_after_em: bool = False,
    beta_merge: float = 0.0,
    min_flicker_duration: int = 3,
    flicker_max_hamming: int = 2,
    merge_max_iter: int = 100,
    sequence_smoothing: str = "none",
    dp_n_candidates: int = 5,
    gamma_switch: float = 5.0,
    gamma_hamming: float = 1.0,
) -> PopulationStateDetection:
    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Y must be a 2D array with shape (T, Q).")
    T, Q = arr.shape
    transformed, preprocessing = _transform_activity(arr, transform)
    finite_range = float(np.nanquantile(transformed, 0.95) - np.nanquantile(transformed, 0.05)) if transformed.size else 0.0
    scaled, offset, scale = robust_run_scale(transformed)
    preprocessing.update({"offset": offset, "scale": scale, "flat_range": finite_range})
    mode = str(segmentation or "fixed").lower()
    if mode == "fixed":
        segments = fixed_segments(T, int(fixed_width))
    elif mode == "pelt":
        feature_mode = str(pelt_feature_mode or "scaled").lower()
        if feature_mode == "weighted":
            pelt_features, pelt_info = make_pelt_features(transformed, smooth_width=int(pelt_smooth_width))
            preprocessing.update(pelt_info)
            preprocessing["pelt_feature_mode"] = "weighted"
            preprocessing["pelt_smooth_width"] = int(pelt_smooth_width)
        elif feature_mode == "scaled":
            pelt_features = scaled
            preprocessing["pelt_feature_mode"] = "scaled"
        else:
            raise ValueError(f"Unknown pelt_feature_mode: {pelt_feature_mode}")
        segments = get_segments_pelt(pelt_features, penalty=float(pelt_penalty), min_size=int(pelt_min_size))
    else:
        raise ValueError(f"Unknown segmentation: {segmentation}")
    atomic_segments = list(segments)
    X, L = segment_means(scaled, segments)
    merge_metadata: Dict[str, Any] = {"merge_after_em": bool(merge_after_em)}
    smoothing_metadata: Dict[str, Any] = {"sequence_smoothing": str(sequence_smoothing or "none").lower()}
    if finite_range < float(flat_range_threshold):
        masks = np.zeros((X.shape[0], Q), dtype=bool)
        result = ActiveSetEMResult(
            masks=masks,
            K=np.zeros(X.shape[0], dtype=int),
            mu0=np.array([0.0]),
            mu1=np.array([0.0]),
            var0=np.array([float(var_floor)]),
            var1=np.array([float(var_floor)]),
            objective=0.0,
            n_iter=0,
            converged=True,
            margin=np.zeros(X.shape[0], dtype=float),
        )
        status = "no_detectable_population_switching"
        final_segments = segments
        final_masks = masks
    else:
        max_active = Q if Kmax is None else min(int(Kmax), Q)
        result = active_set_em_multi_init(
            X,
            L,
            Kmax=max_active,
            max_iter=max_iter,
            tol=tol,
            lambda_active=lambda_active,
            lambda_comb=lambda_comb,
            min_separation=min_separation,
            var_floor=var_floor,
        )
        status = "ok"
        masks_for_final = result.masks
        smooth_mode = str(sequence_smoothing or "none").lower()
        if smooth_mode in {"dp", "potts", "potts_dp"}:
            params = {"mu0": result.mu0, "mu1": result.mu1, "var0": result.var0, "var1": result.var1}
            candidate_sets = build_candidate_sets(
                X,
                params,
                max_active,
                n_candidates=dp_n_candidates,
                base_masks=result.masks,
                lambda_active=lambda_active,
                lambda_comb=lambda_comb,
                min_separation=min_separation,
                include_all_low=True,
            )
            masks_for_final, _smooth_cost, smoothing_metadata = smooth_active_set_sequence_dp(
                candidate_sets,
                L=L,
                gamma_switch=gamma_switch,
                gamma_hamming=gamma_hamming,
            )
            smoothing_metadata.update(
                {
                    "sequence_smoothing": "dp",
                    "n_changed_atomic_masks": int(np.sum(np.any(masks_for_final != result.masks, axis=1))),
                    "boundary_evidence": boundary_evidence(
                        X,
                        masks_for_final,
                        params,
                        max_active,
                        lambda_active=lambda_active,
                        lambda_comb=lambda_comb,
                    ),
                }
            )
        elif smooth_mode not in {"none", "off", "false"}:
            raise ValueError(f"Unknown sequence_smoothing: {sequence_smoothing}")
        if bool(merge_after_em):
            params = {"mu0": result.mu0, "mu1": result.mu1, "var0": result.var0, "var1": result.var1}
            final_segments, final_masks, merge_metadata = merge_active_set_segments(
                scaled,
                segments,
                masks_for_final,
                params,
                max_active,
                beta_merge=beta_merge,
                min_flicker_duration=min_flicker_duration,
                flicker_max_hamming=flicker_max_hamming,
                lambda_active=lambda_active,
                lambda_comb=lambda_comb,
                min_separation=min_separation,
                max_iter=merge_max_iter,
            )
        else:
            final_segments, final_masks = merge_identical_adjacent(segments, masks_for_final)
            merge_metadata = {
                "merge_after_em": False,
                "identical_merge_count": int(len(segments) - len(final_segments)),
                "total_merge_count": int(len(segments) - len(final_segments)),
            }
    merge_metadata.update(smoothing_metadata)
    final_X, final_L = segment_means(scaled, final_segments)
    time_masks = expand_segment_masks(final_masks, final_segments, T)
    labels, _patterns = _labels_from_masks(time_masks)
    episodes = merge_state_episodes(final_masks, final_segments)
    cluster_labels, cluster_occupancy = classify_clusters(
        final_masks,
        final_L,
        low_tol=low_tol,
        high_tol=high_tol,
        min_transitions=min_transitions,
    )
    return PopulationStateDetection(
        result=result,
        segments=final_segments,
        X=final_X,
        L=final_L,
        preprocessing=preprocessing,
        labels=labels,
        time_masks=time_masks,
        episodes=episodes,
        cluster_labels=cluster_labels,
        cluster_occupancy=cluster_occupancy,
        status=status,
        atomic_segments=atomic_segments,
        atomic_masks=result.masks,
        merge_metadata=merge_metadata,
    )


def simulate_active_set_data(M: int = 200, Q: int = 10, noise: float = 0.05, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = np.zeros((int(M), int(Q)), dtype=bool)
    X = np.zeros((int(M), int(Q)), dtype=float)
    mu0 = {0: 0.05, 1: 0.05, 2: 0.04, 3: 0.04}
    mu1 = {1: 1.00, 2: 0.70, 3: 0.50}
    for row in range(int(M)):
        K = int(rng.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1]))
        if K > 0:
            active = rng.choice(int(Q), size=K, replace=False)
            masks[row, active] = True
        for col in range(int(Q)):
            mean = mu1[K] if masks[row, col] else mu0[K]
            X[row, col] = float(mean + float(noise) * rng.standard_normal())
    return X, np.ones(int(M), dtype=float), masks
