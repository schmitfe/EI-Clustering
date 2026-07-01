"""Source-agnostic active-set episode inference and canonicalization.

The entry point operates on :class:`AnalysisInput`, so binary and spiking
simulations share the same state-estimation pipeline once converted by
``analysis.io``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .methods import run_active_set_em
from .preprocessing import subset_analysis_input
from .types import AnalysisInput
from .utils import extract_segments


@dataclass
class EpisodeInferenceOutput:
    analyzed_data: AnalysisInput
    raw_result: Any
    result: Any
    episodes: list[dict[str, Any]]
    summary: dict[str, Any]
    inventory: list[dict[str, Any]]
    emissions: list[dict[str, Any]]


def _get(options: Any, name: str, default: Any = None) -> Any:
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def active_set_configs(options: Any, dt_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the common preprocessing and active-set EM configurations."""
    preprocessing = {
        "dt": float(dt_seconds),
        "use_counts": False,
        "use_rates": True,
        "smoothing_sigma_bins": 0,
        "sqrt_transform": False,
        "zscore": False,
        "temporal_window_bins": 0,
    }
    method = {
        "source": "rate",
        "transform": "auto",
        "segmentation": str(_get(options, "segmentation", "pelt")),
        "fixed_width": int(_get(options, "fixed_width", 1)),
        "pelt_penalty": float(_get(options, "pelt_penalty", 10.0)),
        "changepoint_backend": str(_get(options, "changepoint_backend", "skchange")),
        "changepoint_method": str(_get(options, "changepoint_method", "pelt")),
        "pelt_min_size": int(_get(options, "pelt_min_size", 1)),
        "pelt_jump": int(_get(options, "pelt_jump", 2)),
        "pelt_refine": bool(_get(options, "pelt_refine", True)),
        "changepoint_bandwidth": int(_get(options, "changepoint_bandwidth", 20)),
        "changepoint_max_interval_length": int(_get(options, "changepoint_max_interval_length", 200)),
        "changepoint_parallel_backend": str(_get(options, "changepoint_parallel_backend", "None")),
        "changepoint_parallel_jobs": int(_get(options, "changepoint_parallel_jobs", 1)),
        "pelt_feature_mode": "weighted",
        "pelt_smooth_width": int(_get(options, "pelt_smooth_width", 1)),
        "Kmax": None if _get(options, "kmax") is None else int(_get(options, "kmax")),
        "lambda_active": 0.0,
        "lambda_comb": 0.1,
        "min_separation": 0.05,
        "var_floor": 0.0001,
        "max_iter": int(_get(options, "em_max_iter", 30)),
        "tol": 1e-6,
        "flat_range_threshold": 1e-12,
        "merge_after_em": bool(_get(options, "merge_after_em", False)),
        "beta_merge": float(_get(options, "beta_merge", 0.0)),
        "min_flicker_duration": int(_get(options, "min_flicker_duration", 0)),
        "flicker_max_hamming": int(_get(options, "flicker_max_hamming", 2)),
        "merge_max_iter": int(_get(options, "merge_max_iter", 10)),
        "sequence_smoothing": "none",
    }
    return preprocessing, method


def select_analysis_populations(
    data: AnalysisInput,
    *,
    population_source: str = "excitatory",
    indices: Sequence[int] | None = None,
) -> AnalysisInput:
    """Select inference populations without depending on simulator type."""
    if indices is not None:
        return subset_analysis_input(data, indices=[int(value) for value in indices])
    if population_source == "all":
        return data
    if population_source != "excitatory":
        raise ValueError(f"Unsupported population source: {population_source}")
    if data.cluster_cell_types is not None:
        selected = [
            idx for idx, value in enumerate(data.cluster_cell_types)
            if str(value).upper().startswith("E")
        ]
    else:
        selected = [
            idx for idx, value in enumerate(data.cluster_names or [])
            if str(value).upper().startswith("E")
        ]
    if not selected:
        raise ValueError("No excitatory populations could be identified in AnalysisInput metadata")
    return subset_analysis_input(data, indices=selected)


def episode_rows(
    result: Any, *, condition: str, title: str, seed: int, exclude_edges: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments = result.segments.copy()
    if segments.empty:
        dwell = np.zeros(0, dtype=float)
    else:
        segments["is_edge"] = False
        segments.loc[segments.index[[0, -1]], "is_edge"] = True
        source = segments.loc[~segments["is_edge"]] if exclude_edges and len(segments) > 2 else segments
        dwell = source["duration_time"].to_numpy(dtype=float)
    rows = [
        {
            "condition": condition,
            "title": title,
            "seed": int(seed),
            "episode": int(index),
            "state": int(row["state"]),
            "state_key": str(row.get("state_key", row["state"])),
            "K": int(row.get("K", -1)) if pd.notna(row.get("K", -1)) else -1,
            "clusters": str(row.get("clusters", "")),
            "start_time_s": float(row["start_time"]),
            "stop_time_s": float(row["stop_time"]),
            "dwell_time_s": float(row["duration_time"]),
            "is_edge": bool(row.get("is_edge", False)),
        }
        for index, row in segments.iterrows()
    ]
    summary = {
        "condition": condition,
        "title": title,
        "seed": int(seed),
        "n_states": int(result.n_states),
        "n_episodes": int(len(segments)),
        "n_state_changes": max(0, int(len(segments)) - 1),
        "mean_dwell_time_s": float(np.mean(dwell)) if dwell.size else np.nan,
        "std_dwell_time_s": float(np.std(dwell, ddof=1)) if dwell.size > 1 else 0.0 if dwell.size else np.nan,
        "median_dwell_time_s": float(np.median(dwell)) if dwell.size else np.nan,
        "n_dwells_used": int(dwell.size),
        "status": str(result.metadata.get("status", "ok")),
        "cp_pelt": int(result.metadata.get("CP_pelt", 0)),
        "cp_final": int(result.metadata.get("CP_final", 0)),
    }
    return rows, summary


def cluster_names(data: AnalysisInput) -> list[str]:
    names = list(data.cluster_names or [])
    return [str(name) for name in names] if len(names) == data.n_clusters else [
        f"C{idx + 1}" for idx in range(int(data.n_clusters))
    ]


def _robust_baseline(values: np.ndarray, floor: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, max(float(floor), 1e-12)
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center))) if arr.size > 1 else 0.0
    return center, max(1.4826 * mad, float(floor), 1e-12)


def canonical_state_key(
    mask: np.ndarray,
    rates: np.ndarray,
    names: list[str],
    *,
    kmax: int,
    z_threshold: float,
    similarity: float,
    noise_floor: float,
) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    diagnostics = {
        "truncated": False, "pruned": False, "n_proposed": int(active.size), "n_retained": 0,
        "inactive_baseline": np.nan, "inactive_sigma": np.nan,
    }
    rate_arr = np.asarray(rates, dtype=float)
    inactive = np.setdiff1d(np.arange(rate_arr.size), active, assume_unique=False)
    baseline, sigma = _robust_baseline(rate_arr[inactive] if inactive.size else rate_arr, noise_floor)
    diagnostics.update(inactive_baseline=baseline, inactive_sigma=sigma)
    if active.size:
        active = active[np.argsort(rate_arr[active])[::-1]]
        strongest = float(rate_arr[active[0]])
        retained = [
            int(idx) for rank, idx in enumerate(active.tolist())
            if float(rate_arr[idx]) >= baseline + z_threshold * sigma
            and (rank == 0 or float(rate_arr[idx]) >= similarity * strongest)
        ]
        diagnostics["pruned"] = len(retained) != int(active.size)
        active = np.asarray(retained, dtype=int)
    if active.size > max(0, int(kmax)):
        active = np.sort(active[np.argsort(rate_arr[active])[::-1]][: max(0, int(kmax))])
        diagnostics["truncated"] = True
    else:
        active = np.sort(active)
    diagnostics["n_retained"] = int(active.size)
    selected = tuple(int(idx) for idx in active.tolist())
    return ("LOW" if not selected else "+".join(names[idx] for idx in selected)), selected, diagnostics


def canonicalize_active_set_result(
    result: Any,
    data: AnalysisInput,
    full_data: AnalysisInput,
    *,
    condition: str,
    title: str,
    seed: int,
    exclude_edges: bool,
    kmax: int,
    z_threshold: float,
    similarity: float,
    noise_floor: float,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episodes_meta = list(result.metadata.get("episodes", []))
    if not episodes_meta:
        rows, summary = episode_rows(
            result, condition=condition, title=title, seed=seed, exclude_edges=exclude_edges
        )
        return result, rows, summary, [], []
    names, full_names = cluster_names(data), cluster_names(full_data)
    labels = np.zeros(int(data.n_timepoints), dtype=np.int64)
    keys: list[str] = []
    active_sets: list[tuple[int, ...]] = []
    diagnostics: list[dict[str, Any]] = []
    for episode in episodes_meta:
        start, stop = int(episode["start"]), int(episode["stop"])
        rates = np.nanmean(np.asarray(data.X_rate, dtype=float)[start:stop], axis=0)
        key, active, diagnostic = canonical_state_key(
            np.asarray(episode["mask"], dtype=bool), rates, names, kmax=kmax,
            z_threshold=z_threshold, similarity=similarity, noise_floor=noise_floor,
        )
        keys.append(key); active_sets.append(active); diagnostics.append(diagnostic)
    ordered = sorted(set(keys), key=lambda item: (0 if item == "LOW" else item.count("+") + 1, item))
    key_to_id = {key: idx for idx, key in enumerate(ordered)}
    for episode, key in zip(episodes_meta, keys):
        labels[int(episode["start"]):int(episode["stop"])] = key_to_id[key]
    segments = extract_segments(labels, data.dt)
    active_by_key = {key: active_sets[keys.index(key)] for key in ordered}
    if not segments.empty:
        segments["state_key"] = [ordered[int(state)] for state in segments["state"]]
        segments["K"] = [len(active_by_key[key]) for key in segments["state_key"]]
        segments["clusters"] = [",".join(names[idx] for idx in active_by_key[key]) for key in segments["state_key"]]
    state_means = np.zeros((len(ordered), data.n_clusters), dtype=float)
    full_means = np.zeros((len(ordered), full_data.n_clusters), dtype=float)
    occupancy: dict[int, float] = {}
    for state_id in range(len(ordered)):
        selected = labels == state_id
        occupancy[state_id] = float(np.mean(selected)) if labels.size else 0.0
        if np.any(selected):
            state_means[state_id] = np.nanmean(np.asarray(data.X_rate)[selected], axis=0)
            full_means[state_id] = np.nanmean(np.asarray(full_data.X_rate)[selected], axis=0)
    canonical = SimpleNamespace(
        method="active_set_em_canonical", labels=labels, segments=segments, n_states=len(ordered),
        state_means=state_means, state_occupancy=occupancy,
        metadata={
            **dict(result.metadata), "status": str(result.metadata.get("status", "ok")),
            "canonical_state_keys": ordered, "canonical_cluster_names": names,
            "canonical_kmax": int(kmax), "canonical_z_threshold": float(z_threshold),
            "canonical_similarity": float(similarity), "canonical_noise_floor": float(noise_floor),
            "raw_n_states": int(result.n_states), "raw_n_episodes": int(len(result.segments)),
            "canonical_pruned_episodes": int(sum(bool(item["pruned"]) for item in diagnostics)),
            "canonical_truncated_episodes": int(sum(bool(item["truncated"]) for item in diagnostics)),
        },
    )
    rows, summary = episode_rows(
        canonical, condition=condition, title=title, seed=seed, exclude_edges=exclude_edges
    )
    summary.update(
        raw_n_states=int(result.n_states), raw_n_episodes=int(len(result.segments)),
        canonical_pruned_episodes=int(sum(bool(item["pruned"]) for item in diagnostics)),
        canonical_truncated_episodes=int(sum(bool(item["truncated"]) for item in diagnostics)),
        canonical_z_threshold=float(z_threshold), canonical_similarity=float(similarity),
    )
    inventory: list[dict[str, Any]] = []
    emissions: list[dict[str, Any]] = []
    for state_id, key in enumerate(ordered):
        selected_segments = segments.loc[segments["state"] == state_id] if not segments.empty else pd.DataFrame()
        dwell = selected_segments["duration_time"].to_numpy(dtype=float) if not selected_segments.empty else np.zeros(0)
        active = active_by_key[key]
        inventory.append({
            "condition": condition, "title": title, "seed": int(seed), "state": state_id,
            "state_key": key, "K": len(active), "clusters": ",".join(names[idx] for idx in active),
            "visits": int(dwell.size), "occupancy_time_s": float(np.sum(labels == state_id) * data.dt),
            "occupancy_fraction": occupancy[state_id],
            "mean_dwell_time_s": float(np.mean(dwell)) if dwell.size else np.nan,
            "std_dwell_time_s": float(np.std(dwell, ddof=1)) if dwell.size > 1 else 0.0 if dwell.size else np.nan,
            "median_dwell_time_s": float(np.median(dwell)) if dwell.size else np.nan,
            "first_seen_s": float(selected_segments["start_time"].min()) if not selected_segments.empty else np.nan,
        })
        emission = {"condition": condition, "title": title, "seed": int(seed), "state": state_id, "state_key": key, "K": len(active)}
        emission.update({f"rate_{name}": float(value) for name, value in zip(full_names, full_means[state_id])})
        emissions.append(emission)
    canonical.metadata["canonical_full_cluster_names"] = full_names
    canonical.metadata["canonical_full_state_means"] = full_means
    return canonical, rows, summary, inventory, emissions


def infer_active_set_episodes(
    full_data: AnalysisInput,
    options: Any,
    *,
    condition: str,
    title: str,
    seed: int,
    population_source: str = "excitatory",
    population_indices: Sequence[int] | None = None,
    canonicalize: bool = True,
    exclude_edges: bool = True,
) -> EpisodeInferenceOutput:
    """Run the complete state-estimation pipeline on any ``AnalysisInput``."""
    data = select_analysis_populations(
        full_data, population_source=population_source, indices=population_indices
    )
    preprocessing, method = active_set_configs(options, full_data.dt)
    raw = run_active_set_em(data, preprocessing, method)
    if canonicalize:
        result, episodes, summary, inventory, emissions = canonicalize_active_set_result(
            raw, data, full_data, condition=condition, title=title, seed=seed,
            exclude_edges=exclude_edges, kmax=int(_get(options, "canonical_kmax", 2)),
            z_threshold=float(_get(options, "canonical_z_threshold", 3.0)),
            similarity=float(_get(options, "canonical_similarity", 0.5)),
            noise_floor=float(_get(options, "canonical_noise_floor", 1e-6)),
        )
    else:
        result = raw
        episodes, summary = episode_rows(
            raw, condition=condition, title=title, seed=seed, exclude_edges=exclude_edges
        )
        inventory, emissions = [], []
    return EpisodeInferenceOutput(data, raw, result, episodes, summary, inventory, emissions)
