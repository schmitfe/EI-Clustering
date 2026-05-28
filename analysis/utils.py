from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def _label_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.asarray(labels, dtype=np.int64).ravel()
    if arr.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(arr) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [arr.size]))
    return [(int(start), int(stop), int(arr[start])) for start, stop in zip(starts, stops)]


def extract_segments(labels: np.ndarray, dt: float) -> pd.DataFrame:
    rows = []
    for start, stop, state in _label_runs(labels):
        rows.append(
            {
                "start_bin": start,
                "stop_bin": stop,
                "start_time": start * float(dt),
                "stop_time": stop * float(dt),
                "duration_bins": stop - start,
                "duration_time": (stop - start) * float(dt),
                "state": state,
            }
        )
    return pd.DataFrame(rows, columns=[
        "start_bin",
        "stop_bin",
        "start_time",
        "stop_time",
        "duration_bins",
        "duration_time",
        "state",
    ])


def compute_dwell_times(labels: np.ndarray, dt: float) -> Dict[int, np.ndarray]:
    segments = extract_segments(labels, dt)
    if segments.empty:
        return {}
    return {
        int(state): group["duration_time"].to_numpy(dtype=float)
        for state, group in segments.groupby("state", sort=True)
    }


def compute_transition_matrix(labels: np.ndarray, n_states: Optional[int] = None) -> np.ndarray:
    runs = _label_runs(labels)
    if not runs:
        size = int(n_states or 0)
        return np.zeros((size, size), dtype=float)
    states = np.array([run[2] for run in runs], dtype=np.int64)
    size = int(n_states) if n_states is not None else int(states.max()) + 1
    counts = np.zeros((size, size), dtype=float)
    for src, dst in zip(states[:-1], states[1:]):
        counts[int(src), int(dst)] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return matrix


def compute_state_means(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    lab = np.asarray(labels, dtype=np.int64).ravel()
    size = int(lab.max()) + 1 if lab.size else 0
    means = np.zeros((size, arr.shape[1]), dtype=float)
    for state in range(size):
        mask = lab == state
        if np.any(mask):
            means[state] = arr[mask].mean(axis=0)
    return means


def compute_state_templates(X: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    means = compute_state_means(X, labels)
    limit = 0.5 if threshold is None else float(threshold)
    return (means >= limit).astype(np.uint8)


def relabel_states_by_activity(labels: np.ndarray, state_means: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if state_means.size == 0:
        return np.asarray(labels, dtype=np.int64), np.asarray(state_means, dtype=float)
    ordering = np.argsort(-np.sum(np.asarray(state_means, dtype=float), axis=1), kind="stable")
    inverse = np.empty_like(ordering)
    inverse[ordering] = np.arange(ordering.size)
    relabeled = inverse[np.asarray(labels, dtype=np.int64)]
    return relabeled.astype(np.int64), np.asarray(state_means, dtype=float)[ordering]


def median_filter_labels(labels: np.ndarray, width: int) -> np.ndarray:
    width = int(width)
    if width <= 1:
        return np.asarray(labels, dtype=np.int64).copy()
    if width % 2 == 0:
        width += 1
    return median_filter(np.asarray(labels, dtype=np.int64), size=width, mode="nearest").astype(np.int64)


def _fill_short_runs_1d(values: np.ndarray, target: int, max_width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8).copy()
    if max_width <= 0 or arr.size == 0:
        return arr
    idx = 0
    while idx < arr.size:
        if arr[idx] != target:
            idx += 1
            continue
        start = idx
        while idx < arr.size and arr[idx] == target:
            idx += 1
        stop = idx
        if stop - start > max_width:
            continue
        left = start > 0 and arr[start - 1] != target
        right = stop < arr.size and arr[stop] != target
        if left and right:
            arr[start:stop] = arr[start - 1]
    return arr


def fill_short_gaps_binary(X_binary: np.ndarray, max_gap_bins: int) -> np.ndarray:
    arr = np.asarray(X_binary, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("X_binary must be a 2D array.")
    output = arr.copy()
    for col in range(output.shape[1]):
        output[:, col] = _fill_short_runs_1d(output[:, col], target=0, max_width=int(max_gap_bins))
    return output


def _merge_segment(
    labels: np.ndarray,
    start: int,
    stop: int,
    strategy: str,
    *,
    features: Optional[np.ndarray] = None,
    state_means: Optional[np.ndarray] = None,
) -> None:
    left = int(labels[start - 1]) if start > 0 else None
    right = int(labels[stop]) if stop < labels.size else None
    if left is None and right is None:
        return
    if left is None:
        replacement = right
    elif right is None:
        replacement = left
    elif strategy == "next":
        replacement = right
    elif strategy in {"nearest", "nearest_centroid"} and features is not None and state_means is not None:
        segment_mean = features[start:stop].mean(axis=0)
        left_dist = np.linalg.norm(segment_mean - state_means[left])
        right_dist = np.linalg.norm(segment_mean - state_means[right])
        replacement = left if left_dist <= right_dist else right
    else:
        replacement = left
    labels[start:stop] = int(replacement)


def remove_short_segments(
    labels: np.ndarray,
    min_dwell_bins: int,
    strategy: str = "previous",
    *,
    features: Optional[np.ndarray] = None,
    state_means: Optional[np.ndarray] = None,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).copy()
    min_dwell = int(min_dwell_bins)
    if min_dwell <= 1 or arr.size == 0:
        return arr
    changed = True
    while changed:
        changed = False
        for start, stop, _state in _label_runs(arr):
            if stop - start >= min_dwell:
                continue
            _merge_segment(arr, start, stop, str(strategy).lower(), features=features, state_means=state_means)
            changed = True
            break
    return arr


def active_cluster_patterns_to_labels(X_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X_binary, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("X_binary must be 2D.")
    patterns, inverse = np.unique(arr, axis=0, return_inverse=True)
    return inverse.astype(np.int64), patterns.astype(np.uint8)


def labels_to_active_cluster_patterns(labels: np.ndarray, state_templates: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64).ravel()
    templates = np.asarray(state_templates, dtype=np.uint8)
    if templates.ndim != 2:
        raise ValueError("state_templates must be 2D.")
    if lab.size == 0:
        return np.zeros((0, templates.shape[1]), dtype=np.uint8)
    return templates[lab]
