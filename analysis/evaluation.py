from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score

from .types import StateInferenceResult
from .utils import compute_dwell_times, compute_transition_matrix


def _match_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    true_arr = np.asarray(true_labels, dtype=np.int64).ravel()
    pred_arr = np.asarray(pred_labels, dtype=np.int64).ravel()
    true_states = np.unique(true_arr)
    pred_states = np.unique(pred_arr)
    conf = confusion_matrix(true_arr, pred_arr, labels=np.arange(max(true_states.max(), pred_states.max()) + 1))
    if conf.size == 0:
        return pred_arr.copy(), {}, conf
    rows, cols = linear_sum_assignment(conf.max() - conf)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    remapped = np.array([mapping.get(int(label), int(label)) for label in pred_arr], dtype=np.int64)
    matched_conf = confusion_matrix(true_arr, remapped, labels=np.arange(max(true_arr.max(), remapped.max()) + 1))
    return remapped, mapping, matched_conf


def _boundary_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, tolerance: int) -> Dict[str, float]:
    true_bounds = np.flatnonzero(np.diff(np.asarray(true_labels, dtype=np.int64)) != 0) + 1
    pred_bounds = np.flatnonzero(np.diff(np.asarray(pred_labels, dtype=np.int64)) != 0) + 1
    tol = int(tolerance)
    if pred_bounds.size == 0:
        precision = 1.0 if true_bounds.size == 0 else 0.0
    else:
        hits = sum(np.any(np.abs(true_bounds - value) <= tol) for value in pred_bounds)
        precision = hits / float(pred_bounds.size)
    if true_bounds.size == 0:
        recall = 1.0 if pred_bounds.size == 0 else 0.0
    else:
        hits = sum(np.any(np.abs(pred_bounds - value) <= tol) for value in true_bounds)
        recall = hits / float(true_bounds.size)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def evaluate_result(
    result: StateInferenceResult,
    *,
    true_labels: np.ndarray,
    dt: float,
    boundary_tolerances: Iterable[int] = (1, 2, 5),
) -> Dict[str, object]:
    """Evaluate one inferred state sequence against ground truth labels."""

    pred = np.asarray(result.labels, dtype=np.int64)
    truth = np.asarray(true_labels, dtype=np.int64)
    remapped, mapping, matched_conf = _match_labels(truth, pred)
    metrics: Dict[str, object] = {
        "adjusted_rand_index": float(adjusted_rand_score(truth, pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(truth, pred)),
        "hungarian_accuracy": float(np.mean(remapped == truth)),
        "label_mapping": {str(key): int(value) for key, value in mapping.items()},
        "confusion_matrix": matched_conf.tolist(),
        "state_occupancy": {str(key): float(value) for key, value in result.state_occupancy.items()},
    }
    metrics["boundaries"] = {
        str(int(tol)): _boundary_metrics(truth, remapped, int(tol))
        for tol in boundary_tolerances
    }
    true_dwell = compute_dwell_times(truth, dt)
    pred_dwell = compute_dwell_times(remapped, dt)
    dwell_metrics: Dict[str, object] = {}
    all_states = sorted(set(true_dwell) | set(pred_dwell))
    for state in all_states:
        truth_values = np.asarray(true_dwell.get(state, np.zeros(0)), dtype=float)
        pred_values = np.asarray(pred_dwell.get(state, np.zeros(0)), dtype=float)
        if truth_values.size and pred_values.size:
            ks_value = float(ks_2samp(truth_values, pred_values).statistic)
            pred_mean = float(pred_values.mean())
            true_mean = float(truth_values.mean())
        else:
            ks_value = float("nan")
            pred_mean = float(pred_values.mean()) if pred_values.size else float("nan")
            true_mean = float(truth_values.mean()) if truth_values.size else float("nan")
        dwell_metrics[str(state)] = {
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "mean_abs_error": float(abs(pred_mean - true_mean)) if np.isfinite(pred_mean) and np.isfinite(true_mean) else float("nan"),
            "ks_statistic": ks_value,
        }
    metrics["dwell_times"] = dwell_metrics
    true_transition = compute_transition_matrix(truth)
    n_transition_states = int(true_transition.shape[0])
    if remapped.size:
        n_transition_states = max(n_transition_states, int(remapped.max()) + 1)
    true_transition = compute_transition_matrix(truth, n_states=n_transition_states)
    pred_transition = compute_transition_matrix(remapped, n_states=n_transition_states)
    diff = pred_transition - true_transition
    metrics["transition_matrix"] = {
        "frobenius_error": float(np.linalg.norm(diff)),
        "mean_absolute_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
    }
    return metrics


def compare_results(results: Mapping[str, StateInferenceResult]) -> Dict[str, object]:
    """Compute pairwise method agreement for a result collection."""

    method_names = list(results.keys())
    pairwise: Dict[str, object] = {}
    for left, right in combinations(method_names, 2):
        left_labels = np.asarray(results[left].labels, dtype=np.int64)
        right_labels = np.asarray(results[right].labels, dtype=np.int64)
        key = f"{left}__{right}"
        pairwise[key] = {
            "adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
            "normalized_mutual_info": float(normalized_mutual_info_score(left_labels, right_labels)),
        }
    return {"pairwise": pairwise}
