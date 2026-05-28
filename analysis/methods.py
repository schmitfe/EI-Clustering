from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
    if algorithm == "pelt":
        algo = rpt.Pelt(model=cost).fit(features)
    elif algorithm == "binseg":
        algo = rpt.Binseg(model=cost).fit(features)
    elif algorithm == "window":
        width = int(method_cfg.get("window_width", max(10, features.shape[0] // 20)))
        algo = rpt.Window(width=width, model=cost).fit(features)
    else:
        raise ValueError(f"Unsupported changepoint algorithm '{algorithm}'.")
    penalty = method_cfg.get("penalty")
    n_bkps = method_cfg.get("n_bkps")
    if n_bkps is not None:
        bkps = algo.predict(n_bkps=int(n_bkps))
    else:
        if penalty is None:
            penalty = np.log(max(features.shape[0], 2)) * max(1, features.shape[1])
        bkps = algo.predict(pen=float(penalty))
    starts = [0] + [int(value) for value in bkps[:-1]]
    stops = [int(value) for value in bkps]
    bounds = list(zip(starts, stops))
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
    return _finalize_result(
        "changepoint_kmeans",
        labels,
        data,
        method_cfg,
        metadata={
            "breakpoints": [int(value) for value in bkps],
            "feature_type": feature_type,
            "segment_kmeans_inertia": float(kmeans.inertia_),
            "segment_count": int(len(bounds)),
            "n_requested_states": int(method_cfg.get("n_states", n_states)),
            "n_fitted_states": int(n_states),
        },
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
