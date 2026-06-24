from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from spiketools import spiketimes_to_binary

from .types import AnalysisInput


def bin_spikes_by_cluster(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    neuron_to_cluster: Sequence[int],
    dt: float,
    *,
    t_start: float = 0.0,
    t_stop: Optional[float] = None,
    n_clusters: Optional[int] = None,
) -> np.ndarray:
    """Bin spike trains into a `(T, n_clusters)` count matrix.

    This delegates the actual time binning to `spiketools.spiketimes_to_binary`
    so the analysis pipeline uses the repository's canonical spike binning
    behavior instead of maintaining a parallel implementation.
    """

    times = np.asarray(spike_times, dtype=float).ravel()
    ids = np.asarray(spike_ids, dtype=np.int64).ravel()
    mapping = np.asarray(neuron_to_cluster, dtype=np.int64).ravel()
    if times.shape != ids.shape:
        raise ValueError("spike_times and spike_ids must have the same shape.")
    if mapping.size == 0:
        raise ValueError("neuron_to_cluster must not be empty.")
    if np.any(ids < 0) or np.any(ids >= mapping.size):
        raise ValueError("spike_ids contain indices outside neuron_to_cluster.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_stop is None:
        t_stop = float(times.max()) if times.size else float(t_start)
    if t_stop < t_start:
        raise ValueError("t_stop must be greater than or equal to t_start.")
    total_bins = int(np.ceil((float(t_stop) - float(t_start)) / float(dt)))
    total_bins = max(total_bins, 1)
    cluster_count = int(n_clusters) if n_clusters is not None else int(mapping.max()) + 1
    counts = np.zeros((total_bins, cluster_count), dtype=np.int64)
    if times.size == 0:
        return counts
    valid = (times >= float(t_start)) & (times < float(t_stop))
    if not np.any(valid):
        return counts
    cluster_ids = mapping[ids[valid]]
    canonical_spikes = np.vstack([times[valid], cluster_ids.astype(float, copy=False)])
    # `spiketools.spiketimes_to_binary` switches to centered bins when dt <= 1.
    # Rescale the time axis so the effective bin width is > 1 while preserving
    # the original left-edge binning semantics used by this analysis package.
    scale = max(2.0 / float(dt), 1.0) if float(dt) <= 1.0 else 1.0
    canonical_spikes = canonical_spikes.copy()
    canonical_spikes[0] *= scale
    tlim = [float(t_start) * scale, float(t_stop) * scale]
    binned, _ = spiketimes_to_binary(
        canonical_spikes,
        tlim=tlim,
        dt=float(dt) * scale,
    )
    counts_spiketools = np.asarray(binned, dtype=np.int64).T
    if counts_spiketools.shape[0] != total_bins:
        if counts_spiketools.shape[0] > total_bins:
            counts_spiketools = counts_spiketools[:total_bins]
        else:
            counts_spiketools = np.pad(
                counts_spiketools,
                ((0, total_bins - counts_spiketools.shape[0]), (0, 0)),
                mode="constant",
            )
    copy_clusters = min(cluster_count, counts_spiketools.shape[1])
    counts[:, :copy_clusters] = counts_spiketools[:, :copy_clusters]
    return counts


def compute_cluster_rates(
    X_counts: np.ndarray,
    dt: float,
    *,
    cluster_sizes: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Convert binned counts to rates in Hz."""

    counts = np.asarray(X_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("X_counts must be a 2D array.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    rates = counts / float(dt)
    if cluster_sizes is not None:
        denom = np.asarray(cluster_sizes, dtype=float).ravel()
        if denom.shape[0] != counts.shape[1]:
            raise ValueError("cluster_sizes must match the number of clusters.")
        rates = rates / np.maximum(denom[None, :], 1.0)
    return rates


def smooth_rates(X_rate: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Apply Gaussian temporal smoothing to cluster rates."""

    arr = np.asarray(X_rate, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X_rate must be a 2D array.")
    if sigma_bins <= 0:
        return arr.copy()
    return gaussian_filter1d(arr, sigma=float(sigma_bins), axis=0, mode="nearest")


def sqrt_transform_counts(X_counts: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(X_counts, dtype=float), a_min=0.0, a_max=None))


def zscore_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(np.asarray(X, dtype=float))


def _apply_hysteresis(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    states = np.zeros_like(X, dtype=np.uint8)
    for col in range(X.shape[1]):
        active = False
        for row in range(X.shape[0]):
            value = X[row, col]
            if active:
                active = bool(value >= low[col])
            else:
                active = bool(value >= high[col])
            states[row, col] = 1 if active else 0
    return states


def binarize_activity(
    X: np.ndarray,
    *,
    mode: str = "percentile",
    threshold: Optional[float] = None,
    percentile: float = 80.0,
    std_factor: float = 1.0,
    zscore_threshold: float = 0.0,
    hysteresis: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Threshold continuous activity into a binary active/inactive matrix."""

    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    if np.all(np.isin(arr, [0.0, 1.0])):
        return arr.astype(np.uint8, copy=True)
    mode = str(mode or "percentile").lower()
    if mode == "auto":
        mode = "percentile"
    if mode == "fixed":
        if threshold is None:
            raise ValueError("A fixed threshold requires `threshold`.")
        thresh = np.full(arr.shape[1], float(threshold), dtype=float)
        return (arr >= thresh[None, :]).astype(np.uint8)
    if mode == "percentile":
        thresh = np.percentile(arr, float(percentile), axis=0)
        return (arr >= thresh[None, :]).astype(np.uint8)
    if mode == "mean_plus_std":
        thresh = np.mean(arr, axis=0) + float(std_factor) * np.std(arr, axis=0)
        return (arr >= thresh[None, :]).astype(np.uint8)
    if mode == "zscore":
        zscored = zscore_features(arr)
        return (zscored >= float(zscore_threshold)).astype(np.uint8)
    if mode == "hysteresis":
        settings = dict(hysteresis or {})
        low_percentile = float(settings.get("low_percentile", 60.0))
        high_percentile = float(settings.get("high_percentile", 80.0))
        low = np.percentile(arr, low_percentile, axis=0)
        high = np.percentile(arr, high_percentile, axis=0)
        return _apply_hysteresis(arr, low=low, high=high)
    raise ValueError(f"Unsupported threshold mode '{mode}'.")


def make_temporal_window_features(X: np.ndarray, window_bins: int) -> np.ndarray:
    """Concatenate lagged copies of a feature matrix along the feature axis."""

    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    window = int(window_bins)
    if window <= 0:
        return arr.copy()
    padded = np.pad(arr, ((window, window), (0, 0)), mode="edge")
    blocks = [padded[offset : offset + arr.shape[0]] for offset in range(2 * window + 1)]
    return np.concatenate(blocks, axis=1)


def validate_analysis_input(data: AnalysisInput) -> AnalysisInput:
    """Validate shapes and required metadata for a common analysis input."""

    matrices = [arr for arr in (data.X_counts, data.X_binary, data.X_rate) if arr is not None]
    if not matrices:
        raise ValueError("AnalysisInput must define at least one of X_counts, X_binary, or X_rate.")
    reference_shape = None
    for matrix in matrices:
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError("All feature matrices must be 2D.")
        reference_shape = reference_shape or arr.shape
        if arr.shape != reference_shape:
            raise ValueError("All feature matrices in AnalysisInput must have the same shape.")
    if data.dt <= 0:
        raise ValueError("AnalysisInput.dt must be positive.")
    if data.true_labels is not None:
        labels = np.asarray(data.true_labels)
        if labels.ndim != 1 or labels.shape[0] != reference_shape[0]:
            raise ValueError("true_labels must be a length-T vector.")
    if data.true_active_clusters is not None:
        active = np.asarray(data.true_active_clusters)
        if active.shape != reference_shape:
            raise ValueError("true_active_clusters must match the feature-matrix shape.")
    if data.cluster_names is not None and len(data.cluster_names) != reference_shape[1]:
        raise ValueError("cluster_names length must match the number of clusters.")
    if data.cluster_cell_types is not None and len(data.cluster_cell_types) != reference_shape[1]:
        raise ValueError("cluster_cell_types length must match the number of clusters.")
    if data.cluster_group_ids is not None and len(data.cluster_group_ids) != reference_shape[1]:
        raise ValueError("cluster_group_ids length must match the number of clusters.")
    return data


def subset_analysis_input(
    data: AnalysisInput,
    *,
    indices: Sequence[int],
) -> AnalysisInput:
    """Return a shallow copy of `AnalysisInput` restricted to selected clusters."""

    selected = np.asarray(indices, dtype=np.int64).ravel()
    if selected.size == 0:
        raise ValueError("Cluster selection removed all clusters.")
    if np.any(selected < 0) or np.any(selected >= data.n_clusters):
        raise ValueError("Cluster selection contains out-of-range indices.")
    selected_list = selected.tolist()
    return validate_analysis_input(
        replace(
            data,
            X_counts=None if data.X_counts is None else np.asarray(data.X_counts)[:, selected],
            X_binary=None if data.X_binary is None else np.asarray(data.X_binary)[:, selected],
            X_rate=None if data.X_rate is None else np.asarray(data.X_rate)[:, selected],
            cluster_ids=selected_list if data.cluster_ids is None else [data.cluster_ids[idx] for idx in selected_list],
            cluster_names=None if data.cluster_names is None else [str(data.cluster_names[idx]) for idx in selected_list],
            cluster_cell_types=None if data.cluster_cell_types is None else [str(data.cluster_cell_types[idx]) for idx in selected_list],
            cluster_group_ids=None if data.cluster_group_ids is None else [int(data.cluster_group_ids[idx]) for idx in selected_list],
            true_active_clusters=None
            if data.true_active_clusters is None
            else np.asarray(data.true_active_clusters)[:, selected],
            metadata={
                **dict(data.metadata),
                "selected_cluster_indices": selected_list,
            },
        )
    )


def apply_cluster_selection(data: AnalysisInput, selection_cfg: Dict[str, Any]) -> AnalysisInput:
    """Apply config-driven cluster selection to an analysis input."""

    cfg = dict(selection_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return data
    mode = str(cfg.get("mode", "all") or "all").lower()
    if mode == "all":
        return data
    if mode in {"excitatory", "inhibitory"}:
        if data.cluster_cell_types is None:
            raise ValueError(f"Cluster selection mode '{mode}' requires cluster_cell_types metadata.")
        target = "E" if mode == "excitatory" else "I"
        indices = [idx for idx, cell_type in enumerate(data.cluster_cell_types) if str(cell_type).upper().startswith(target)]
        return subset_analysis_input(data, indices=indices)
    if mode == "indices":
        return subset_analysis_input(data, indices=cfg.get("indices", []))
    if mode == "names":
        if data.cluster_names is None:
            raise ValueError("Cluster selection by names requires cluster_names metadata.")
        wanted = {str(name) for name in cfg.get("names", [])}
        indices = [idx for idx, name in enumerate(data.cluster_names) if str(name) in wanted]
        return subset_analysis_input(data, indices=indices)
    if mode in {"group_ids", "cluster_ids"}:
        if data.cluster_group_ids is None:
            raise ValueError(f"Cluster selection mode '{mode}' requires cluster_group_ids metadata.")
        wanted = {int(value) for value in cfg.get("group_ids", cfg.get("cluster_ids", []))}
        indices = [idx for idx, group_id in enumerate(data.cluster_group_ids) if int(group_id) in wanted]
        return subset_analysis_input(data, indices=indices)
    raise ValueError(f"Unsupported cluster selection mode '{mode}'.")


def _population_filter_source(data: AnalysisInput, cfg: Dict[str, Any]) -> np.ndarray:
    source = str(cfg.get("source", "auto") or "auto").lower()
    if source == "auto":
        return np.asarray(data.X_rate if data.X_rate is not None else data.preferred_matrix(), dtype=float)
    if source == "rate":
        if data.X_rate is None:
            raise ValueError("population_filter.source=rate requires X_rate.")
        return np.asarray(data.X_rate, dtype=float)
    if source == "counts":
        if data.X_counts is None:
            raise ValueError("population_filter.source=counts requires X_counts.")
        return np.asarray(data.X_counts, dtype=float)
    if source == "binary":
        if data.X_binary is None:
            raise ValueError("population_filter.source=binary requires X_binary.")
        return np.asarray(data.X_binary, dtype=float)
    raise ValueError(f"Unsupported population_filter.source '{source}'.")


def infer_network_state_mask(X: np.ndarray, cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Infer low/high network-state bins from average population activity."""

    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be 2D.")
    sigma = float(cfg.get("smoothing_sigma_bins", 0) or 0)
    network_mean = arr.mean(axis=1)
    if sigma > 0:
        network_mean = gaussian_filter1d(network_mean, sigma=sigma, mode="nearest")
    method = str(cfg.get("network_state_method", "kmeans2") or "kmeans2").lower()
    if method == "kmeans2":
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=int(cfg.get("random_state", 0)))
        labels = kmeans.fit_predict(network_mean[:, None]).astype(np.int64)
        centers = np.asarray(kmeans.cluster_centers_, dtype=float).ravel()
        high_state = int(np.argmax(centers))
        return labels == high_state, network_mean
    if method == "percentile":
        percentile = float(cfg.get("high_percentile", 75.0))
        threshold = float(np.percentile(network_mean, percentile))
        return network_mean >= threshold, network_mean
    raise ValueError(f"Unsupported network_state_method '{method}'.")


def estimate_population_switching(
    data: AnalysisInput,
    filter_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Score how strongly each population modulates with the inferred network state."""

    X = _population_filter_source(data, filter_cfg)
    high_mask, network_mean = infer_network_state_mask(X, filter_cfg)
    if not np.any(high_mask) or np.all(high_mask):
        raise ValueError("Population filter could not split the network into low/high states.")
    low_mask = ~high_mask
    rows = []
    min_mean_delta = float(filter_cfg.get("min_mean_delta", 0.0) or 0.0)
    min_effect_size = float(filter_cfg.get("min_effect_size", 0.0) or 0.0)
    min_correlation = float(filter_cfg.get("min_correlation", 0.0) or 0.0)
    sign = str(filter_cfg.get("modulation_sign", "positive") or "positive").lower()
    for idx in range(X.shape[1]):
        values = X[:, idx]
        mean_low = float(values[low_mask].mean())
        mean_high = float(values[high_mask].mean())
        std_low = float(values[low_mask].std(ddof=0))
        std_high = float(values[high_mask].std(ddof=0))
        pooled = np.sqrt(0.5 * (std_low ** 2 + std_high ** 2))
        delta = mean_high - mean_low
        effect_size = float(delta / pooled) if pooled > 1e-12 else float("inf" if delta > 0 else 0.0)
        corr = float(np.corrcoef(values, network_mean)[0, 1]) if np.std(values) > 1e-12 and np.std(network_mean) > 1e-12 else 0.0
        modulation_index = float(delta / (mean_high + mean_low + 1e-12))
        if sign == "positive":
            sign_ok = delta >= 0
        elif sign == "negative":
            sign_ok = delta <= 0
        else:
            sign_ok = True
        keep = sign_ok and (abs(delta) >= min_mean_delta) and (abs(effect_size) >= min_effect_size or abs(corr) >= min_correlation)
        rows.append(
            {
                "cluster_index": idx,
                "cluster_name": None if data.cluster_names is None else str(data.cluster_names[idx]),
                "cell_type": None if data.cluster_cell_types is None else str(data.cluster_cell_types[idx]),
                "group_id": None if data.cluster_group_ids is None else int(data.cluster_group_ids[idx]),
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta": float(delta),
                "effect_size": float(effect_size),
                "correlation": corr,
                "modulation_index": modulation_index,
                "sign_ok": bool(sign_ok),
                "keep": bool(keep),
            }
        )
    frame = pd.DataFrame(rows)
    if filter_cfg.get("keep_top_k") is not None:
        keep_top_k = max(1, int(filter_cfg["keep_top_k"]))
        ranked = frame.sort_values(["sign_ok", "effect_size", "delta", "correlation"], ascending=False).head(keep_top_k)
        frame["keep"] = frame["cluster_index"].isin(ranked["cluster_index"].tolist())
    min_keep = int(filter_cfg.get("min_keep", 1) or 1)
    if int(frame["keep"].sum()) < min_keep:
        ranked = frame.sort_values(["sign_ok", "effect_size", "delta", "correlation"], ascending=False).head(min_keep)
        frame["keep"] = frame["cluster_index"].isin(ranked["cluster_index"].tolist())
    return frame


def apply_population_filter(
    data: AnalysisInput,
    filter_cfg: Dict[str, Any],
) -> tuple[AnalysisInput, Optional[pd.DataFrame]]:
    """Exclude weakly modulated populations before normalization and state inference."""

    cfg = dict(filter_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return data, None
    diagnostics = estimate_population_switching(data, cfg)
    kept = diagnostics.loc[diagnostics["keep"], "cluster_index"].to_list()
    filtered = subset_analysis_input(data, indices=kept)
    filtered = replace(
        filtered,
        metadata={
            **dict(filtered.metadata),
            "population_filter": {
                "kept_count": int(len(kept)),
                "excluded_count": int(data.n_clusters - len(kept)),
                "kept_cluster_indices": [int(value) for value in kept],
            },
        },
    )
    return validate_analysis_input(filtered), diagnostics


def build_feature_matrix(
    data: AnalysisInput,
    feature_type: str,
    preprocessing_cfg: Dict[str, Any],
) -> np.ndarray:
    """Create a feature matrix according to the configured preprocessing mode."""

    feature_type = str(feature_type).lower()
    counts = None if data.X_counts is None else np.asarray(data.X_counts, dtype=float)
    rates = None if data.X_rate is None else np.asarray(data.X_rate, dtype=float)
    binary = None if data.X_binary is None else np.asarray(data.X_binary, dtype=float)
    sigma = float(preprocessing_cfg.get("smoothing_sigma_bins", 0) or 0)
    temporal_window = int(preprocessing_cfg.get("temporal_window_bins", 0) or 0)

    if feature_type in {"counts", "raw_counts"}:
        if counts is None:
            if rates is None:
                raise ValueError("Count features requested but X_counts is unavailable.")
            base = rates
        else:
            base = counts
    elif feature_type == "sqrt_counts":
        if counts is None:
            if rates is None:
                raise ValueError("sqrt_counts requires X_counts.")
            base = sqrt_transform_counts(rates)
        else:
            base = sqrt_transform_counts(counts)
    elif feature_type in {"zscore_counts", "counts_zscore"}:
        if counts is None:
            if rates is None:
                raise ValueError("zscore_counts requires X_counts.")
            base = zscore_features(rates)
        else:
            base = zscore_features(counts)
    elif feature_type in {"sqrt_zscore", "sqrt_counts_zscore"}:
        if counts is None:
            if rates is None:
                raise ValueError("sqrt_zscore requires X_counts.")
            base = zscore_features(sqrt_transform_counts(rates))
        else:
            base = zscore_features(sqrt_transform_counts(counts))
    elif feature_type in {"rates", "raw_rates"}:
        if rates is None:
            raise ValueError("Rate features requested but X_rate is unavailable.")
        base = rates
    elif feature_type in {"zscore_rates", "rates_zscore"}:
        if rates is None:
            raise ValueError("zscore_rates requires X_rate.")
        base = zscore_features(rates)
    elif feature_type == "smoothed_rates":
        if rates is None:
            raise ValueError("smoothed_rates requires X_rate.")
        base = smooth_rates(rates, sigma_bins=sigma)
    elif feature_type == "binary":
        if binary is None:
            source = rates if rates is not None else counts
            if source is None:
                raise ValueError("binary features require X_binary or another numeric source.")
            base = binarize_activity(
                source,
                mode=str(preprocessing_cfg.get("binary_threshold_mode", "percentile")),
                percentile=float(preprocessing_cfg.get("binary_threshold_percentile", 80.0)),
                hysteresis=preprocessing_cfg.get("hysteresis"),
            ).astype(float)
        else:
            base = binary
    elif feature_type == "temporal_window":
        source = rates if rates is not None else counts
        if source is None:
            raise ValueError("temporal_window features require X_rate or X_counts.")
        base = source
    else:
        raise ValueError(f"Unsupported feature type '{feature_type}'.")

    return make_temporal_window_features(base, temporal_window) if feature_type == "temporal_window" or temporal_window > 0 else np.asarray(base, dtype=float)
