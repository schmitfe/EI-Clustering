from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .active_set import detect_population_states
from .preprocessing import binarize_activity, build_feature_matrix
from .types import AnalysisInput, StateInferenceResult
from .utils import (
    active_cluster_patterns_to_labels,
    compute_dwell_times,
    compute_state_means,
    compute_state_templates,
    compute_transition_matrix,
    extract_segments,
    fill_short_gaps_binary,
    median_filter_labels,
    relabel_states_by_activity,
    remove_short_segments,
)


def _reference_matrix(data: AnalysisInput) -> np.ndarray:
    if data.X_rate is not None:
        return np.asarray(data.X_rate, dtype=float)
    if data.X_counts is not None:
        return np.asarray(data.X_counts, dtype=float)
    if data.X_binary is not None:
        return np.asarray(data.X_binary, dtype=float)
    raise ValueError("No reference matrix available.")


def _binary_matrix_for_threshold(data: AnalysisInput, preprocessing_cfg: Dict[str, Any], method_cfg: Dict[str, Any]) -> np.ndarray:
    if data.X_binary is not None and np.all(np.isin(data.X_binary, [0, 1])) and not method_cfg.get("force_rethreshold", False):
        return np.asarray(data.X_binary, dtype=np.uint8)
    source = data.X_rate if data.X_rate is not None else data.X_counts
    if source is None:
        raise ValueError("Threshold method requires X_binary, X_rate, or X_counts.")
    threshold_mode = method_cfg.get("threshold_mode", "auto")
    if str(threshold_mode).lower() == "auto":
        threshold_mode = preprocessing_cfg.get("binary_threshold_mode", "percentile")
    return binarize_activity(
        np.asarray(source, dtype=float),
        mode=str(threshold_mode),
        threshold=method_cfg.get("fixed_threshold"),
        percentile=float(method_cfg.get("threshold_percentile", preprocessing_cfg.get("binary_threshold_percentile", 80.0))),
        std_factor=float(method_cfg.get("threshold_std_factor", 1.0)),
        zscore_threshold=float(method_cfg.get("zscore_threshold", 0.0)),
        hysteresis=preprocessing_cfg.get("hysteresis"),
    )


def _remove_short_binary_runs_1d(values: np.ndarray, active_value: int, min_width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8).copy()
    width = int(min_width)
    if width <= 1 or arr.size == 0:
        return arr
    idx = 0
    while idx < arr.size:
        if arr[idx] != active_value:
            idx += 1
            continue
        start = idx
        while idx < arr.size and arr[idx] == active_value:
            idx += 1
        stop = idx
        if stop - start < width:
            arr[start:stop] = 1 - active_value
    return arr


def _clean_binary_matrix(X_binary: np.ndarray, method_cfg: Dict[str, Any]) -> np.ndarray:
    output = np.asarray(X_binary, dtype=np.uint8).copy()
    output = fill_short_gaps_binary(output, int(method_cfg.get("max_gap_bins", 0) or 0))
    min_active = int(method_cfg.get("min_active_bins", 1) or 1)
    min_inactive = int(method_cfg.get("min_inactive_bins", 1) or 1)
    for col in range(output.shape[1]):
        output[:, col] = _remove_short_binary_runs_1d(output[:, col], active_value=1, min_width=min_active)
        output[:, col] = _remove_short_binary_runs_1d(output[:, col], active_value=0, min_width=min_inactive)
    return output


def _finalize_result(
    method: str,
    labels: np.ndarray,
    data: AnalysisInput,
    config: Dict[str, Any],
    *,
    posterior_probs: Optional[np.ndarray] = None,
    log_likelihood: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    state_template_source: Optional[np.ndarray] = None,
) -> StateInferenceResult:
    unique_labels = np.unique(np.asarray(labels, dtype=np.int64))
    label_map = {int(label): idx for idx, label in enumerate(unique_labels.tolist())}
    compact_labels = np.array([label_map[int(label)] for label in np.asarray(labels, dtype=np.int64)], dtype=np.int64)
    reference = _reference_matrix(data)
    raw_means = compute_state_means(reference, compact_labels)
    relabeled, state_means = relabel_states_by_activity(compact_labels, raw_means)
    segments = extract_segments(relabeled, data.dt)
    dwell_times = compute_dwell_times(relabeled, data.dt)
    transition = compute_transition_matrix(relabeled, n_states=state_means.shape[0])
    template_source = state_template_source if state_template_source is not None else data.X_binary
    state_templates = None
    if template_source is not None:
        state_templates = compute_state_templates(template_source, relabeled, threshold=0.5)
    occupancy = {}
    total = float(relabeled.size) if relabeled.size else 1.0
    for state in range(state_means.shape[0]):
        occupancy[int(state)] = float(np.mean(relabeled == state)) if relabeled.size else 0.0
    return StateInferenceResult(
        method=method,
        labels=relabeled.astype(np.int64),
        segments=segments,
        dwell_times=dwell_times,
        transition_matrix=transition,
        state_means=state_means,
        state_templates=state_templates,
        state_occupancy=occupancy,
        posterior_probs=None if posterior_probs is None else np.asarray(posterior_probs, dtype=float),
        log_likelihood=None if log_likelihood is None else float(log_likelihood),
        config=dict(config),
        metadata=dict(metadata or {}),
    )


def run_threshold_filter(
    data: AnalysisInput,
    preprocessing_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> StateInferenceResult:
    X_binary = _binary_matrix_for_threshold(data, preprocessing_cfg, method_cfg)
    X_binary = _clean_binary_matrix(X_binary, method_cfg)
    labels, patterns = active_cluster_patterns_to_labels(X_binary)
    width = int(method_cfg.get("median_filter_width", 0) or 0)
    if width > 1:
        labels = median_filter_labels(labels, width=width)
    labels = remove_short_segments(labels, int(method_cfg.get("min_dwell_bins", 1) or 1), strategy="previous")
    return _finalize_result(
        "threshold_filter",
        labels,
        data,
        method_cfg,
        metadata={"n_patterns": int(patterns.shape[0])},
        state_template_source=X_binary,
    )


def run_kmeans_filter(
    data: AnalysisInput,
    preprocessing_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> StateInferenceResult:
    feature_type = str(method_cfg.get("feature_type", "sqrt_zscore"))
    features = build_feature_matrix(data, feature_type, preprocessing_cfg)
    n_states = max(1, min(int(method_cfg.get("n_states", 2)), features.shape[0]))
    kmeans = KMeans(
        n_clusters=n_states,
        n_init=int(method_cfg.get("n_init", 20)),
        max_iter=int(method_cfg.get("max_iter", 300)),
        random_state=int(method_cfg.get("random_state", 0)),
    )
    labels = kmeans.fit_predict(features).astype(np.int64)
    width = int(method_cfg.get("median_filter_width", 0) or 0)
    if width > 1:
        labels = median_filter_labels(labels, width=width)
    labels = remove_short_segments(
        labels,
        int(method_cfg.get("min_dwell_bins", 1) or 1),
        strategy=str(method_cfg.get("merge_strategy", "previous")),
        features=features,
        state_means=np.asarray(kmeans.cluster_centers_, dtype=float),
    )
    return _finalize_result(
        "kmeans_filter",
        labels,
        data,
        method_cfg,
        metadata={
            "inertia": float(kmeans.inertia_),
            "feature_type": feature_type,
            "n_requested_states": int(method_cfg.get("n_states", n_states)),
            "n_fitted_states": int(n_states),
        },
    )


def _segment_feature_matrix(
    X: np.ndarray,
    X_binary: np.ndarray,
    bounds: list[tuple[int, int]],
    config: Dict[str, Any],
) -> np.ndarray:
    rows = []
    use_mean = bool(config.get("mean", True))
    use_max = bool(config.get("max", False))
    use_active_fraction = bool(config.get("active_fraction", True))
    use_duration = bool(config.get("duration", False))
    use_start_end_difference = bool(config.get("start_end_difference", False))
    for start, stop in bounds:
        chunk = X[start:stop]
        binary_chunk = X_binary[start:stop]
        features = []
        if use_mean:
            features.append(chunk.mean(axis=0))
        if use_max:
            features.append(chunk.max(axis=0))
        if use_active_fraction:
            features.append(binary_chunk.mean(axis=0))
        if use_start_end_difference:
            features.append(chunk[-1] - chunk[0])
        if use_duration:
            features.append(np.array([stop - start], dtype=float))
        rows.append(np.concatenate(features, axis=0))
    return np.vstack(rows) if rows else np.zeros((0, X.shape[1]), dtype=float)


def _changepoint_bounds(bkps: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [int(value) for value in bkps[:-1]]
    stops = [int(value) for value in bkps]
    return list(zip(starts, stops))


def _fit_segment_kmeans_labels(
    data: AnalysisInput,
    features: np.ndarray,
    binary_source: np.ndarray,
    bounds: list[tuple[int, int]],
    method_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    segment_features = _segment_feature_matrix(
        _reference_matrix(data),
        np.asarray(binary_source, dtype=float),
        bounds,
        dict(method_cfg.get("segment_features", {})),
    )
    n_states = max(1, min(int(method_cfg.get("n_states", 2)), segment_features.shape[0]))
    kmeans = KMeans(
        n_clusters=n_states,
        n_init=int(method_cfg.get("n_init", 20)),
        random_state=int(method_cfg.get("random_state", 0)),
        max_iter=int(method_cfg.get("max_iter", 300)),
    )
    segment_labels = kmeans.fit_predict(segment_features)
    labels = np.zeros(data.n_timepoints, dtype=np.int64)
    for (start, stop), state in zip(bounds, segment_labels):
        labels[start:stop] = int(state)
    labels = remove_short_segments(
        labels,
        int(method_cfg.get("min_dwell_bins", 1) or 1),
        strategy=str(method_cfg.get("merge_strategy", "previous")),
        features=features,
        state_means=np.asarray(kmeans.cluster_centers_, dtype=float),
    )
    return labels, np.asarray(kmeans.cluster_centers_, dtype=float), float(kmeans.inertia_), int(n_states)


def _segmentation_cost(features: np.ndarray, cost_name: str, bkps: list[int]) -> float:
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("ruptures is required for changepoint_kmeans.") from exc
    cost = rpt.costs.cost_factory(cost_name).fit(features)
    start = 0
    total = 0.0
    for stop in bkps:
        total += float(cost.error(start, int(stop)))
        start = int(stop)
    return total


def _select_crops_candidate(candidates: list[Dict[str, Any]], method_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not candidates:
        raise RuntimeError("CROPS penalty path did not produce any candidates.")
    ordered = sorted(candidates, key=lambda item: (int(item["n_changepoints"]), float(item["cost"])))
    zero_costs = [float(item["cost"]) for item in ordered if int(item["n_changepoints"]) == 0]
    baseline_cost = zero_costs[0] if zero_costs else max(float(item["cost"]) for item in ordered)
    if baseline_cost > 0.0:
        for item in ordered:
            item["loss_reduction_fraction"] = (baseline_cost - float(item["cost"])) / baseline_cost
    else:
        for item in ordered:
            item["loss_reduction_fraction"] = 0.0
    min_loss_reduction = method_cfg.get("crops_min_loss_reduction")
    if min_loss_reduction is not None:
        threshold = float(min_loss_reduction)
        viable = [
            item
            for item in ordered
            if int(item["n_changepoints"]) == 0
            or float(item.get("loss_reduction_fraction", 0.0)) >= threshold
        ]
        if viable:
            ordered = viable
    min_selected_penalty = method_cfg.get("crops_min_selected_penalty")
    if min_selected_penalty is not None:
        threshold = float(min_selected_penalty)
        viable = [
            item
            for item in ordered
            if int(item["n_changepoints"]) == 0
            or float(item["penalty"]) >= threshold
        ]
        if viable:
            ordered = viable
    min_adjacent_distance = method_cfg.get("crops_min_adjacent_distance")
    if min_adjacent_distance is not None:
        threshold = float(min_adjacent_distance)
        viable = [
            item
            for item in ordered
            if int(item["n_changepoints"]) == 0
            or float(item.get("min_adjacent_distance", -np.inf)) >= threshold
        ]
        if viable:
            ordered = viable
    target = method_cfg.get("crops_target_segments")
    if target is not None:
        target_segments = max(1, int(target))
        return min(
            ordered,
            key=lambda item: (
                abs(int(item["n_segments"]) - target_segments),
                -float(item["penalty"]),
                float(item["cost"]),
            ),
        )
    selection = str(method_cfg.get("crops_selection", "elbow")).lower()
    if selection == "max_penalty":
        return max(ordered, key=lambda item: float(item["penalty"]))
    if selection == "min_penalty":
        return min(ordered, key=lambda item: float(item["penalty"]))
    max_cps = method_cfg.get("crops_max_changepoints_for_elbow")
    elbow_pool = ordered
    if max_cps is not None:
        threshold = int(max_cps)
        elbow_pool = [item for item in ordered if int(item["n_changepoints"]) <= threshold]
        if len(elbow_pool) < 3:
            elbow_pool = ordered
    if len(elbow_pool) < 3:
        return min(elbow_pool, key=lambda item: (abs(int(item["n_segments"]) - int(method_cfg.get("n_states", 2))), float(item["cost"])))
    points = np.array([[float(item["n_changepoints"]), float(item["cost"])] for item in elbow_pool], dtype=float)
    spans = np.ptp(points, axis=0)
    spans[spans == 0.0] = 1.0
    scaled = (points - points.min(axis=0, keepdims=True)) / spans[None, :]
    start = scaled[0]
    end = scaled[-1]
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm <= 0.0:
        return elbow_pool[0]
    distances = np.abs(np.cross(line, scaled - start) / norm)
    return elbow_pool[int(np.argmax(distances))]


def _candidate_adjacent_distances(matrix: np.ndarray, bkps: list[int]) -> tuple[float, float]:
    bounds = _changepoint_bounds(bkps)
    if len(bounds) <= 1:
        return float("inf"), float("inf")
    arr = np.asarray(matrix, dtype=float)
    means = []
    for start, stop in bounds:
        means.append(arr[int(start) : int(stop)].mean(axis=0))
    distances = [float(np.linalg.norm(means[idx + 1] - means[idx])) for idx in range(len(means) - 1)]
    return float(np.min(distances)), float(np.median(distances))


def _merge_bounds_by_similarity_and_dwell(
    bounds: list[tuple[int, int]],
    matrix: np.ndarray,
    *,
    min_distance: float,
    min_dwell_bins: int,
) -> tuple[list[tuple[int, int]], Dict[str, Any]]:
    if len(bounds) <= 1:
        return bounds, {"merge_count": 0, "merge_reasons": []}
    arr = np.asarray(matrix, dtype=float)
    merged = [(int(start), int(stop)) for start, stop in bounds]
    reasons: list[Dict[str, Any]] = []

    def mean_for(bound: tuple[int, int]) -> np.ndarray:
        start, stop = bound
        return arr[start:stop].mean(axis=0)

    changed = True
    while changed and len(merged) > 1:
        changed = False
        best_idx = None
        best_reason = None
        best_score = None
        for idx in range(len(merged) - 1):
            left = merged[idx]
            right = merged[idx + 1]
            left_len = left[1] - left[0]
            right_len = right[1] - right[0]
            distance = float(np.linalg.norm(mean_for(left) - mean_for(right)))
            short = min(left_len, right_len) < int(min_dwell_bins)
            similar = distance < float(min_distance)
            if not short and not similar:
                continue
            score = distance
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
                best_reason = {
                    "left": list(left),
                    "right": list(right),
                    "distance": distance,
                    "left_len": int(left_len),
                    "right_len": int(right_len),
                    "short": bool(short),
                    "similar": bool(similar),
                }
        if best_idx is not None:
            left = merged[best_idx]
            right = merged[best_idx + 1]
            merged[best_idx : best_idx + 2] = [(left[0], right[1])]
            reasons.append(dict(best_reason or {}))
            changed = True
    return merged, {"merge_count": len(reasons), "merge_reasons": reasons}


def _bounds_to_breakpoints(bounds: list[tuple[int, int]]) -> list[int]:
    return [int(stop) for _, stop in bounds]


def _resolve_cluster_indices(data: AnalysisInput, source: str) -> list[int]:
    label = str(source or "all").lower()
    if label in {"excitatory", "e"} and data.cluster_names is not None:
        indices = [idx for idx, name in enumerate(data.cluster_names) if str(name).startswith("E")]
        if indices:
            return indices
    return list(range(data.n_clusters))


def _threshold_segment_templates(
    means: np.ndarray,
    *,
    mode: str,
    percentile: float,
    fixed_threshold: Optional[float],
    min_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(means, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Segment template means must be 2D.")
    if arr.shape[0] == 0:
        return np.zeros_like(arr, dtype=np.uint8), np.zeros(arr.shape[1], dtype=float)
    mode = str(mode or "relative").lower()
    if mode == "fixed":
        if fixed_threshold is None:
            raise ValueError("template_threshold_mode='fixed' requires template_fixed_threshold.")
        thresholds = np.full(arr.shape[1], float(fixed_threshold), dtype=float)
    elif mode == "percentile":
        thresholds = np.percentile(arr, float(percentile), axis=0)
    elif mode == "relative":
        low = np.percentile(arr, 20.0, axis=0)
        high = np.percentile(arr, 95.0, axis=0)
        frac = float(fixed_threshold) if fixed_threshold is not None else 0.35
        thresholds = low + frac * np.maximum(high - low, 0.0)
    else:
        raise ValueError(f"Unsupported template_threshold_mode '{mode}'.")
    templates = (arr >= thresholds[None, :]).astype(np.uint8)
    if int(min_active) > 0:
        for idx in range(templates.shape[0]):
            if int(templates[idx].sum()) >= int(min_active):
                continue
            if arr.shape[1] > 0:
                top = np.argsort(arr[idx])[-int(min_active) :]
                templates[idx, top] = 1
    return templates, thresholds


def _merge_bounds_by_template_and_dwell(
    bounds: list[tuple[int, int]],
    matrix: np.ndarray,
    method_cfg: Dict[str, Any],
) -> tuple[list[tuple[int, int]], Dict[str, Any]]:
    if len(bounds) <= 1:
        return bounds, {"template_merge_count": 0, "template_merge_reasons": [], "segment_templates": []}
    arr = np.asarray(matrix, dtype=float)
    min_dwell = int(method_cfg.get("template_min_dwell_bins", method_cfg.get("merge_min_dwell_bins", 1)) or 1)
    threshold_mode = str(method_cfg.get("template_threshold_mode", "relative"))
    percentile = float(method_cfg.get("template_threshold_percentile", 80.0))
    fixed = method_cfg.get("template_fixed_threshold")
    fixed_value = None if fixed is None else float(fixed)
    min_active = int(method_cfg.get("template_min_active_clusters", 1) or 0)
    merged = [(int(start), int(stop)) for start, stop in bounds]
    reasons: list[Dict[str, Any]] = []

    def compute_templates(current_bounds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.vstack([arr[start:stop].mean(axis=0) for start, stop in current_bounds])
        templates, thresholds = _threshold_segment_templates(
            means,
            mode=threshold_mode,
            percentile=percentile,
            fixed_threshold=fixed_value,
            min_active=min_active,
        )
        return means, templates, thresholds

    thresholds = np.zeros(arr.shape[1], dtype=float)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        means, templates, thresholds = compute_templates(merged)
        for idx in range(len(merged) - 1):
            left = merged[idx]
            right = merged[idx + 1]
            left_len = left[1] - left[0]
            right_len = right[1] - right[0]
            same_template = bool(np.array_equal(templates[idx], templates[idx + 1]))
            short = min(left_len, right_len) < min_dwell
            if not same_template and not short:
                continue
            merged[idx : idx + 2] = [(left[0], right[1])]
            reasons.append(
                {
                    "left": list(left),
                    "right": list(right),
                    "same_template": same_template,
                    "short": bool(short),
                    "left_len": int(left_len),
                    "right_len": int(right_len),
                    "left_template": templates[idx].astype(int).tolist(),
                    "right_template": templates[idx + 1].astype(int).tolist(),
                }
            )
            changed = True
            break
    means, templates, thresholds = compute_templates(merged)
    metadata = {
        "template_merge_count": len(reasons),
        "template_merge_reasons": reasons,
        "template_thresholds": thresholds.tolist(),
        "segment_templates": templates.astype(int).tolist(),
        "segment_template_means": means.tolist(),
    }
    return merged, metadata


def _fit_template_labels(
    data: AnalysisInput,
    bounds: list[tuple[int, int]],
    matrix: np.ndarray,
    method_cfg: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(matrix, dtype=float)
    means = np.vstack([arr[start:stop].mean(axis=0) for start, stop in bounds]) if bounds else np.zeros((0, arr.shape[1]))
    templates, thresholds = _threshold_segment_templates(
        means,
        mode=str(method_cfg.get("template_threshold_mode", "relative")),
        percentile=float(method_cfg.get("template_threshold_percentile", 80.0)),
        fixed_threshold=None if method_cfg.get("template_fixed_threshold") is None else float(method_cfg.get("template_fixed_threshold")),
        min_active=int(method_cfg.get("template_min_active_clusters", 1) or 0),
    )
    template_to_state: dict[tuple[int, ...], int] = {}
    labels = np.zeros(data.n_timepoints, dtype=np.int64)
    segment_states: list[int] = []
    for (start, stop), template in zip(bounds, templates):
        key = tuple(int(value) for value in template.tolist())
        if key not in template_to_state:
            template_to_state[key] = len(template_to_state)
        state = template_to_state[key]
        labels[start:stop] = state
        segment_states.append(state)
    metadata = {
        "template_thresholds": thresholds.tolist(),
        "segment_templates": templates.astype(int).tolist(),
        "segment_template_states": segment_states,
        "unique_templates": [list(key) for key, _ in sorted(template_to_state.items(), key=lambda item: item[1])],
    }
    return labels, metadata


def _run_pelt_crops(
    features: np.ndarray,
    cost: str,
    method_cfg: Dict[str, Any],
    *,
    quality_matrix: Optional[np.ndarray] = None,
) -> tuple[list[int], Dict[str, Any]]:
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("ruptures is required for changepoint_kmeans.") from exc
    min_penalty = float(method_cfg.get("crops_min_penalty", 0.5))
    max_penalty = float(method_cfg.get("crops_max_penalty", max(1.0, np.log(max(features.shape[0], 2)) * max(1, features.shape[1]))))
    n_penalties = max(2, int(method_cfg.get("crops_n_penalties", 40)))
    if min_penalty <= 0.0 or max_penalty <= 0.0:
        raise ValueError("CROPS penalty bounds must be positive.")
    if min_penalty > max_penalty:
        min_penalty, max_penalty = max_penalty, min_penalty
    penalties = np.geomspace(min_penalty, max_penalty, n_penalties)
    seen: set[tuple[int, ...]] = set()
    candidates: list[Dict[str, Any]] = []
    for penalty in penalties:
        bkps = [int(value) for value in rpt.Pelt(model=cost).fit(features).predict(pen=float(penalty))]
        key = tuple(bkps)
        if key in seen:
            continue
        seen.add(key)
        min_adjacent, median_adjacent = (float("inf"), float("inf"))
        if quality_matrix is not None:
            min_adjacent, median_adjacent = _candidate_adjacent_distances(np.asarray(quality_matrix, dtype=float), bkps)
        candidates.append(
            {
                "penalty": float(penalty),
                "breakpoints": bkps,
                "n_changepoints": max(0, len(bkps) - 1),
                "n_segments": len(bkps),
                "cost": _segmentation_cost(features, cost, bkps),
                "min_adjacent_distance": min_adjacent,
                "median_adjacent_distance": median_adjacent,
            }
        )
    selected = _select_crops_candidate(candidates, method_cfg)
    metadata = {
        "crops_penalty_range": [min_penalty, max_penalty],
        "crops_n_penalties": n_penalties,
        "crops_selection": str(method_cfg.get("crops_selection", "elbow")),
        "crops_selected_penalty": float(selected["penalty"]),
        "crops_selected_cost": float(selected["cost"]),
        "crops_candidates": candidates,
    }
    return [int(value) for value in selected["breakpoints"]], metadata


def run_changepoint_kmeans(
    data: AnalysisInput,
    preprocessing_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> StateInferenceResult:
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency exercised in integration tests
        raise ModuleNotFoundError("ruptures is required for changepoint_kmeans.") from exc

    feature_type = str(method_cfg.get("feature_type", "smoothed_rates"))
    features = build_feature_matrix(data, feature_type, preprocessing_cfg)
    binary_source = data.X_binary
    if binary_source is None:
        binary_source = binarize_activity(
            _reference_matrix(data),
            mode=str(preprocessing_cfg.get("binary_threshold_mode", "percentile")),
            percentile=float(preprocessing_cfg.get("binary_threshold_percentile", 80.0)),
            hysteresis=preprocessing_cfg.get("hysteresis"),
        )
    algorithm = str(method_cfg.get("algorithm", "pelt")).lower()
    cost = str(method_cfg.get("cost", "rbf")).lower()
    crops_metadata: Dict[str, Any] = {}
    if algorithm in {"pelt_crops", "crops"}:
        quality_source = str(method_cfg.get("crops_quality_source", "all")).lower()
        quality_matrix = _reference_matrix(data)
        quality_indices = _resolve_cluster_indices(data, quality_source)
        quality_matrix = quality_matrix[:, quality_indices]
        bkps, crops_metadata = _run_pelt_crops(features, cost, method_cfg, quality_matrix=quality_matrix)
        algo = None
    elif algorithm == "pelt":
        algo = rpt.Pelt(model=cost).fit(features)
    elif algorithm == "binseg":
        algo = rpt.Binseg(model=cost).fit(features)
    elif algorithm == "window":
        width = int(method_cfg.get("window_width", max(10, features.shape[0] // 20)))
        algo = rpt.Window(width=width, model=cost).fit(features)
    else:
        raise ValueError(f"Unsupported changepoint algorithm '{algorithm}'.")
    if algorithm not in {"pelt_crops", "crops"}:
        penalty = method_cfg.get("penalty")
        n_bkps = method_cfg.get("n_bkps")
        if n_bkps is not None:
            bkps = algo.predict(n_bkps=int(n_bkps))
        else:
            if penalty is None:
                penalty = np.log(max(features.shape[0], 2)) * max(1, features.shape[1])
            bkps = algo.predict(pen=float(penalty))
    bkps = [int(value) for value in bkps]
    bounds = _changepoint_bounds(bkps)
    merge_metadata: Dict[str, Any] = {}
    if bool(method_cfg.get("merge_adjacent_segments", False)):
        if "quality_matrix" not in locals():
            quality_matrix = _reference_matrix(data)
        merge_min_distance = float(method_cfg.get("merge_min_adjacent_distance", method_cfg.get("crops_min_adjacent_distance", 0.0)) or 0.0)
        merge_min_dwell = int(method_cfg.get("merge_min_dwell_bins", 1) or 1)
        bounds, merge_metadata = _merge_bounds_by_similarity_and_dwell(
            bounds,
            np.asarray(quality_matrix, dtype=float),
            min_distance=merge_min_distance,
            min_dwell_bins=merge_min_dwell,
        )
        bkps = _bounds_to_breakpoints(bounds)
    template_metadata: Dict[str, Any] = {}
    if bool(method_cfg.get("template_state_assignment", False)):
        template_source = str(method_cfg.get("template_source", "excitatory")).lower()
        template_indices = _resolve_cluster_indices(data, template_source)
        template_matrix = _reference_matrix(data)[:, template_indices]
        if bool(method_cfg.get("template_merge_adjacent", True)):
            bounds, template_merge_metadata = _merge_bounds_by_template_and_dwell(
                bounds,
                template_matrix,
                method_cfg,
            )
            bkps = _bounds_to_breakpoints(bounds)
            template_metadata.update(template_merge_metadata)
        labels, template_label_metadata = _fit_template_labels(data, bounds, template_matrix, method_cfg)
        template_metadata.update(template_label_metadata)
        kmeans_inertia = 0.0
        n_states = int(np.unique(labels).size)
    else:
        labels, state_means, kmeans_inertia, n_states = _fit_segment_kmeans_labels(
            data,
            features,
            np.asarray(binary_source, dtype=float),
            bounds,
            method_cfg,
        )
    metadata = {
        "breakpoints": [int(value) for value in bkps],
        "feature_type": feature_type,
        "segment_kmeans_inertia": kmeans_inertia,
        "segment_count": int(len(bounds)),
        "n_requested_states": int(method_cfg.get("n_states", n_states)),
        "n_fitted_states": int(n_states),
    }
    metadata.update(crops_metadata)
    metadata.update(merge_metadata)
    metadata.update(template_metadata)
    return _finalize_result(
        "changepoint_kmeans",
        labels,
        data,
        method_cfg,
        metadata=metadata,
    )


def run_active_set_em(
    data: AnalysisInput,
    preprocessing_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> StateInferenceResult:
    source = str(method_cfg.get("source", "auto")).lower()
    if source == "auto":
        if data.X_rate is not None:
            Y = np.asarray(data.X_rate, dtype=float)
        elif data.X_counts is not None:
            Y = np.asarray(data.X_counts, dtype=float)
        elif data.X_binary is not None:
            Y = np.asarray(data.X_binary, dtype=float)
        else:
            raise ValueError("active_set_em requires X_rate, X_counts, or X_binary.")
    elif source == "rate":
        if data.X_rate is None:
            raise ValueError("active_set_em source='rate' requires X_rate.")
        Y = np.asarray(data.X_rate, dtype=float)
    elif source == "counts":
        if data.X_counts is None:
            raise ValueError("active_set_em source='counts' requires X_counts.")
        Y = np.asarray(data.X_counts, dtype=float)
    elif source == "binary":
        if data.X_binary is None:
            raise ValueError("active_set_em source='binary' requires X_binary.")
        Y = np.asarray(data.X_binary, dtype=float)
    else:
        raise ValueError(f"Unsupported active_set_em source '{source}'.")
    transform = str(method_cfg.get("transform", "identity"))
    if transform == "auto":
        transform = "sqrt" if data.source_type == "snn" and source != "binary" else "identity"
    detection = detect_population_states(
        Y,
        transform=transform,
        segmentation=str(method_cfg.get("segmentation", "fixed")),
        fixed_width=int(method_cfg.get("fixed_width", 10)),
        pelt_penalty=float(method_cfg.get("pelt_penalty", 10.0)),
        pelt_min_size=int(method_cfg.get("pelt_min_size", 5)),
        pelt_feature_mode=str(method_cfg.get("pelt_feature_mode", "weighted")),
        pelt_smooth_width=int(method_cfg.get("pelt_smooth_width", 3)),
        Kmax=None if method_cfg.get("Kmax") is None else int(method_cfg.get("Kmax")),
        lambda_active=float(method_cfg.get("lambda_active", 0.0)),
        lambda_comb=float(method_cfg.get("lambda_comb", 0.1)),
        min_separation=float(method_cfg.get("min_separation", 0.05)),
        var_floor=float(method_cfg.get("var_floor", 1e-4)),
        max_iter=int(method_cfg.get("max_iter", 100)),
        tol=float(method_cfg.get("tol", 1e-6)),
        flat_range_threshold=float(method_cfg.get("flat_range_threshold", 1e-12)),
        low_tol=float(method_cfg.get("low_tol", 0.01)),
        high_tol=float(method_cfg.get("high_tol", 0.99)),
        min_transitions=int(method_cfg.get("min_transitions", 1)),
        merge_after_em=bool(method_cfg.get("merge_after_em", False)),
        beta_merge=float(method_cfg.get("beta_merge", 0.0)),
        min_flicker_duration=int(method_cfg.get("min_flicker_duration", 3)),
        flicker_max_hamming=int(method_cfg.get("flicker_max_hamming", 2)),
        merge_max_iter=int(method_cfg.get("merge_max_iter", 100)),
        sequence_smoothing=str(method_cfg.get("sequence_smoothing", "none")),
        dp_n_candidates=int(method_cfg.get("dp_n_candidates", 5)),
        gamma_switch=float(method_cfg.get("gamma_switch", 5.0)),
        gamma_hamming=float(method_cfg.get("gamma_hamming", 1.0)),
    )
    res = detection.result
    final_masks = np.asarray([episode["mask"] for episode in detection.episodes], dtype=bool) if detection.episodes else np.zeros((0, Y.shape[1]), dtype=bool)
    final_K = final_masks.sum(axis=1).astype(int) if final_masks.size else np.zeros(0, dtype=int)
    metadata = {
        "source": source,
        "status": detection.status,
        "preprocessing": detection.preprocessing,
        "segments": [[int(start), int(stop)] for start, stop in detection.segments],
        "segment_active_masks": final_masks.astype(int),
        "segment_K": final_K,
        "segment_lengths": detection.L,
        "segment_means_scaled": detection.X,
        "atomic_segments": [[int(start), int(stop)] for start, stop in (detection.atomic_segments or [])],
        "atomic_active_masks": res.masks.astype(int),
        "atomic_K": res.K,
        "CP_pelt": max(0, len(detection.atomic_segments or []) - 1),
        "CP_final": max(0, len(detection.episodes) - 1),
        "mu0_by_K": res.mu0,
        "mu1_by_K": res.mu1,
        "var0_by_K": res.var0,
        "var1_by_K": res.var1,
        "em_objective": float(res.objective),
        "em_converged": bool(res.converged),
        "em_n_iter": int(res.n_iter),
        "em_margin": res.margin,
        "cluster_labels": detection.cluster_labels,
        "cluster_occupancy": detection.cluster_occupancy,
        "episodes": detection.episodes,
        "merge": dict(detection.merge_metadata or {}),
        "diagnostic_note": "Run-level robust scaling only; no per-cluster z-scoring.",
    }
    return _finalize_result(
        "active_set_em",
        detection.labels,
        data,
        method_cfg,
        log_likelihood=-float(res.objective),
        metadata=metadata,
        state_template_source=detection.time_masks.astype(np.uint8),
    )


def _instantiate_hmm(emission: str, n_states: int, emission_dim: int):
    from dynamax.hidden_markov_model import BernoulliHMM, GaussianHMM, PoissonHMM

    if emission == "poisson":
        return PoissonHMM(num_states=n_states, emission_dim=emission_dim)
    if emission == "bernoulli":
        return BernoulliHMM(num_states=n_states, emission_dim=emission_dim)
    if emission == "gaussian":
        return GaussianHMM(num_states=n_states, emission_dim=emission_dim)
    raise ValueError(f"Unsupported HMM emission '{emission}'.")


def _call_with_fallbacks(target, *args, **kwargs):
    attempts = [
        lambda: target(*args, **kwargs),
        lambda: target(*args),
        lambda: target(**kwargs),
    ]
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No call attempts were executed.")


def run_hmm(
    data: AnalysisInput,
    preprocessing_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> StateInferenceResult:
    try:
        import jax.numpy as jnp
        import jax.random as jr
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "The HMM pipeline requires dynamax and jax. Install them to enable analysis.methods.hmm."
        ) from exc

    emission = str(method_cfg.get("emission", "auto")).lower()
    if emission == "auto":
        if data.X_binary is not None and np.all(np.isin(data.X_binary, [0, 1])):
            emission = "bernoulli"
        elif data.X_counts is not None:
            emission = "poisson"
        else:
            emission = "gaussian"
    if emission == "bernoulli":
        observations = np.asarray(
            data.X_binary
            if data.X_binary is not None
            else binarize_activity(
                _reference_matrix(data),
                mode=str(preprocessing_cfg.get("binary_threshold_mode", "percentile")),
                percentile=float(preprocessing_cfg.get("binary_threshold_percentile", 80.0)),
                hysteresis=preprocessing_cfg.get("hysteresis"),
            ),
            dtype=float,
        )
    elif emission == "poisson":
        observations = np.asarray(data.X_counts if data.X_counts is not None else _reference_matrix(data), dtype=float)
    else:
        observations = np.asarray(build_feature_matrix(data, str(method_cfg.get("feature_type", "zscore_rates")), preprocessing_cfg), dtype=float)

    n_states = int(method_cfg.get("n_states", 2))
    num_iters = int(method_cfg.get("num_iters", 100))
    num_seeds = int(method_cfg.get("num_seeds", 1))
    base_seed = int(method_cfg.get("random_state", 0))
    best_payload: Optional[Dict[str, Any]] = None
    for offset in range(num_seeds):
        seed = base_seed + offset
        model = _instantiate_hmm(emission, n_states=n_states, emission_dim=observations.shape[1])
        key = jr.PRNGKey(seed)
        params, props = _call_with_fallbacks(model.initialize, key=key, method="kmeans", emissions=jnp.asarray(observations))
        fitted = _call_with_fallbacks(model.fit_em, params, props, jnp.asarray(observations), num_iters=num_iters)
        if isinstance(fitted, tuple) and len(fitted) >= 2:
            fitted_params = fitted[0]
        else:
            fitted_params = fitted
        log_likelihood = float(np.asarray(_call_with_fallbacks(model.marginal_log_prob, fitted_params, jnp.asarray(observations))))
        posterior = _call_with_fallbacks(model.smoother, fitted_params, jnp.asarray(observations))
        posterior_probs = None
        if hasattr(posterior, "smoothed_probs"):
            posterior_probs = np.asarray(posterior.smoothed_probs)
        labels = np.asarray(_call_with_fallbacks(model.most_likely_states, fitted_params, jnp.asarray(observations)), dtype=np.int64)
        payload = {
            "labels": labels,
            "posterior_probs": posterior_probs,
            "log_likelihood": log_likelihood,
            "seed": seed,
            "emission": emission,
        }
        if best_payload is None or log_likelihood > float(best_payload["log_likelihood"]):
            best_payload = payload
    if best_payload is None:
        raise RuntimeError("HMM fitting did not produce any candidate model.")
    metadata = {
        "emission": emission,
        "best_seed": int(best_payload["seed"]),
        "dwell_time_note": "HMM dwell times follow geometric statistics and should be treated as a baseline.",
    }
    return _finalize_result(
        "hmm",
        np.asarray(best_payload["labels"], dtype=np.int64),
        data,
        method_cfg,
        posterior_probs=best_payload["posterior_probs"],
        log_likelihood=float(best_payload["log_likelihood"]),
        metadata=metadata,
    )
