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
        segments = get_segments_pelt(scaled, penalty=float(pelt_penalty), min_size=int(pelt_min_size))
    else:
        raise ValueError(f"Unknown segmentation: {segmentation}")
    X, L = segment_means(scaled, segments)
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
    else:
        result = active_set_em_multi_init(
            X,
            L,
            Kmax=Kmax,
            max_iter=max_iter,
            tol=tol,
            lambda_active=lambda_active,
            lambda_comb=lambda_comb,
            min_separation=min_separation,
            var_floor=var_floor,
        )
        status = "ok"
    time_masks = expand_segment_masks(result.masks, segments, T)
    labels, _patterns = _labels_from_masks(time_masks)
    episodes = merge_state_episodes(result.masks, segments)
    cluster_labels, cluster_occupancy = classify_clusters(
        result.masks,
        L,
        low_tol=low_tol,
        high_tol=high_tol,
        min_transitions=min_transitions,
    )
    return PopulationStateDetection(
        result=result,
        segments=segments,
        X=X,
        L=L,
        preprocessing=preprocessing,
        labels=labels,
        time_masks=time_masks,
        episodes=episodes,
        cluster_labels=cluster_labels,
        cluster_occupancy=cluster_occupancy,
        status=status,
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
