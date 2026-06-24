#!/usr/bin/env python3
"""Estimate Figure 3 state dwell times across independently seeded networks."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Figure3 import (  # noqa: E402
    BinaryRunSettings,
    COLUMN_SPECS,
    PipelineSweepSettings,
    _build_column_parameter,
    _build_trace_task,
    _normalize_label,
    _parse_column_override_entries,
    _prepare_focus_markers_with_retry,
    _resolve_binary_config,
    _resolve_column_title,
    _resolve_focus_counts,
    _simulate_binary_task,
    _time_axis_scale_from_taus,
    _validate_column_keys,
)
from analysis.io import analysis_input_from_binary_trace  # noqa: E402
from analysis.methods import run_active_set_em  # noqa: E402
from analysis.preprocessing import subset_analysis_input  # noqa: E402
from analysis.utils import extract_segments  # noqa: E402
from figure_cli import parse_int_values  # noqa: E402
from plotting import plot_spike_raster  # noqa: E402
from sim_config import add_override_arguments, load_from_args, write_yaml_config  # noqa: E402

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional progress display
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated Figure 3 binary networks and summarize robust active-set-EM "
            "state dwell times across quenched-disorder realizations."
        )
    )
    add_override_arguments(parser)
    parser.add_argument("--n-simulations", type=int, default=10, help="Networks per Figure 3 condition.")
    parser.add_argument("--duration", type=float, default=30.0, help="Recorded duration per network in seconds.")
    parser.add_argument("--seed-start", type=int, default=100, help="First network seed.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel BinaryNetwork simulations.")
    parser.add_argument(
        "--analysis-jobs",
        type=int,
        help="Parallel state-estimation workers; defaults to --jobs.",
    )
    parser.add_argument("--erf-jobs", type=int, default=1, help="Parallel workers for mean-field preparation.")
    parser.add_argument("--warmup-steps", type=int, help="Override binary warmup steps.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        help="Binary rate/state sampling interval in updates; defaults to approximately 10 ms.",
    )
    parser.add_argument("--batch-size", type=int, help="Override binary batch size.")
    parser.add_argument("--focus-counts", nargs="+", help="Focus counts, e.g. 5:1:-1.")
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=[spec.label for spec in COLUMN_SPECS],
        help="Figure 3 conditions to run; defaults to all columns.",
    )
    parser.add_argument("--stability-filter", choices=("stable", "unstable", "any"), default="any")
    parser.add_argument("--column-override", action="append", default=[], metavar="label:path=value")
    parser.add_argument("--column-title", action="append", default=[], metavar="label:title")
    parser.add_argument("--delta-rep-mf", type=float, default=0.025)
    parser.add_argument("--rep-retry-mf", type=int, default=10)
    parser.add_argument("--rep-rng-seed", type=int)
    parser.add_argument("--analysis-only", action="store_true", help="Reuse existing matching traces.")
    parser.add_argument(
        "--skip-missing-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing traces in analysis-only mode and write missing_traces.csv.",
    )
    parser.add_argument("--overwrite-simulation", action="store_true")
    parser.add_argument(
        "--segmentation",
        choices=("fixed", "pelt"),
        default="pelt",
        help="Atomic segmentation before active-set EM.",
    )
    parser.add_argument(
        "--fixed-width",
        type=int,
        default=1,
        help="Atomic segment width in analysis bins when --segmentation=fixed.",
    )
    parser.add_argument("--pelt-penalty", type=float, default=10.0)
    parser.add_argument(
        "--changepoint-backend",
        choices=("skchange", "sktime", "ruptures"),
        default="skchange",
        help="Library used for changepoint detection.",
    )
    parser.add_argument(
        "--changepoint-method",
        choices=("pelt", "moving_window", "seeded_binseg", "binseg"),
        default="pelt",
        help="Changepoint estimator. moving_window and seeded_binseg avoid exact PELT scaling.",
    )
    parser.add_argument(
        "--pelt-min-size",
        type=int,
        default=1,
        help="Minimum state duration in analysis bins; one bin is approximately 10 ms by default.",
    )
    parser.add_argument(
        "--pelt-jump",
        type=int,
        default=2,
        help="Evaluate coarse PELT changepoints every N analysis bins.",
    )
    parser.add_argument(
        "--pelt-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine coarse PELT boundaries locally at full analysis-bin resolution.",
    )
    parser.add_argument("--pelt-smooth-width", type=int, default=1)
    parser.add_argument("--changepoint-bandwidth", type=int, default=20)
    parser.add_argument("--changepoint-max-interval-length", type=int, default=200)
    parser.add_argument(
        "--changepoint-parallel-backend",
        choices=("None", "loky", "threading", "multiprocessing"),
        default="None",
        help="sktime/skchange estimator-level parallel backend; useful for moving_window, not PELT.",
    )
    parser.add_argument("--changepoint-parallel-jobs", type=int, default=1)
    parser.add_argument("--em-max-iter", type=int, default=30)
    parser.add_argument("--kmax", type=int, help="Maximum simultaneously active populations considered by EM.")
    parser.add_argument(
        "--population-source",
        choices=("excitatory", "all"),
        default="excitatory",
        help="Population rates used for state inference; rasters still show both E and I neurons.",
    )
    parser.add_argument(
        "--state-canonicalization",
        choices=("none", "active_set"),
        default="active_set",
        help="Merge EM episodes that have the same inferred active population set.",
    )
    parser.add_argument(
        "--canonical-kmax",
        type=int,
        default=2,
        help="Maximum active populations kept in a canonical state key; larger EM masks are truncated to top rates.",
    )
    parser.add_argument(
        "--canonical-z-threshold",
        type=float,
        default=3.0,
        help="Retain active clusters only if they exceed the inactive-cluster baseline by this many robust SDs.",
    )
    parser.add_argument(
        "--canonical-similarity",
        type=float,
        default=0.5,
        help="Retain secondary active clusters only if their rate is at least this fraction of the dominant active rate.",
    )
    parser.add_argument(
        "--canonical-noise-floor",
        type=float,
        default=1e-6,
        help="Lower bound for the robust inactive-cluster SD used by canonical pruning.",
    )
    parser.add_argument("--beta-merge", type=float, default=0.0)
    parser.add_argument(
        "--merge-after-em",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run costly unsupported-boundary merging after EM; disabled to preserve short dwells.",
    )
    parser.add_argument(
        "--min-flicker-duration",
        type=int,
        default=0,
        help="Remove weak A-B-A flickers up to this many bins; zero preserves all short states.",
    )
    parser.add_argument(
        "--flicker-max-hamming",
        type=int,
        default=2,
        help="Maximum active-set identity changes for removing a short A-B-A flicker.",
    )
    parser.add_argument(
        "--merge-max-iter",
        type=int,
        default=10,
        help="Maximum expensive unsupported-boundary merge passes per trace.",
    )
    parser.add_argument(
        "--inspection-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one inferred-state overlay raster per network.",
    )
    parser.add_argument(
        "--inspection-duration",
        type=float,
        help="Only show the first N seconds in inspection rasters; defaults to the full run.",
    )
    parser.add_argument("--raster-stride", type=int, default=20, help="Plot every Nth neuron in inspection rasters.")
    parser.add_argument(
        "--max-raster-events",
        type=int,
        default=2_000_000,
        help="Maximum events per inspection raster after time/neuron filtering; 0 disables the limit.",
    )
    parser.add_argument(
        "--exclude-edge-dwells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude first and last, potentially censored, episodes from per-network means.",
    )
    parser.add_argument("--output-dir", default="plots/Figure3_dwell_times")
    parser.add_argument("--output-prefix", default="Figures/Figure3_dwell_times")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _analysis_configs(args: argparse.Namespace, dt_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    preprocessing = {
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
        "segmentation": str(args.segmentation),
        "fixed_width": int(args.fixed_width),
        "pelt_penalty": float(args.pelt_penalty),
        "changepoint_backend": str(args.changepoint_backend),
        "changepoint_method": str(args.changepoint_method),
        "pelt_min_size": int(args.pelt_min_size),
        "pelt_jump": int(args.pelt_jump),
        "pelt_refine": bool(args.pelt_refine),
        "changepoint_bandwidth": int(args.changepoint_bandwidth),
        "changepoint_max_interval_length": int(args.changepoint_max_interval_length),
        "changepoint_parallel_backend": str(args.changepoint_parallel_backend),
        "changepoint_parallel_jobs": int(args.changepoint_parallel_jobs),
        "pelt_feature_mode": "weighted",
        "pelt_smooth_width": int(args.pelt_smooth_width),
        "Kmax": None if args.kmax is None else int(args.kmax),
        "lambda_active": 0.0,
        "lambda_comb": 0.1,
        "min_separation": 0.05,
        "var_floor": 0.0001,
        "max_iter": int(args.em_max_iter),
        "tol": 1e-6,
        "flat_range_threshold": 1e-12,
        "merge_after_em": bool(args.merge_after_em),
        "beta_merge": float(args.beta_merge),
        "min_flicker_duration": int(args.min_flicker_duration),
        "flicker_max_hamming": int(args.flicker_max_hamming),
        "merge_max_iter": int(args.merge_max_iter),
        "sequence_smoothing": "none",
    }
    return {"dt": float(dt_seconds), **preprocessing}, method


def _episode_rows(
    result: Any,
    *,
    condition: str,
    title: str,
    seed: int,
    exclude_edges: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments = result.segments.copy()
    if segments.empty:
        dwell = np.zeros(0, dtype=float)
    else:
        segments["is_edge"] = False
        segments.loc[segments.index[[0, -1]], "is_edge"] = True
        dwell_source = segments.loc[~segments["is_edge"]] if exclude_edges and len(segments) > 2 else segments
        dwell = dwell_source["duration_time"].to_numpy(dtype=float)
    rows = []
    for episode_index, row in segments.iterrows():
        rows.append(
            {
                "condition": condition,
                "title": title,
                "seed": int(seed),
                "episode": int(episode_index),
                "state": int(row["state"]),
                "state_key": str(row.get("state_key", row["state"])),
                "K": int(row.get("K", -1)) if pd.notna(row.get("K", -1)) else -1,
                "clusters": str(row.get("clusters", "")),
                "start_time_s": float(row["start_time"]),
                "stop_time_s": float(row["stop_time"]),
                "dwell_time_s": float(row["duration_time"]),
                "is_edge": bool(row.get("is_edge", False)),
            }
        )
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


def _cluster_names(data: Any) -> list[str]:
    names = list(data.cluster_names or [])
    if len(names) == data.n_clusters:
        return [str(name) for name in names]
    return [f"C{idx + 1}" for idx in range(int(data.n_clusters))]


def _robust_baseline(values: np.ndarray, floor: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, max(float(floor), 1e-12)
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center))) if arr.size > 1 else 0.0
    sigma = max(1.4826 * mad, float(floor), 1e-12)
    return center, sigma


def _canonical_state_key(
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
        "truncated": False,
        "pruned": False,
        "n_proposed": int(active.size),
        "n_retained": 0,
        "inactive_baseline": np.nan,
        "inactive_sigma": np.nan,
    }
    rate_arr = np.asarray(rates, dtype=float)
    inactive = np.setdiff1d(np.arange(rate_arr.size), active, assume_unique=False)
    baseline_values = rate_arr[inactive] if inactive.size else rate_arr
    baseline, sigma = _robust_baseline(baseline_values, floor=float(noise_floor))
    diagnostics["inactive_baseline"] = float(baseline)
    diagnostics["inactive_sigma"] = float(sigma)

    if active.size:
        active = active[np.argsort(rate_arr[active])[::-1]]
        strongest_rate = float(rate_arr[active[0]])
        threshold = float(baseline) + float(z_threshold) * float(sigma)
        retained: list[int] = []
        for rank, idx in enumerate(active.tolist()):
            rate = float(rate_arr[int(idx)])
            above_background = rate >= threshold
            similar_to_primary = rank == 0 or rate >= float(similarity) * strongest_rate
            if above_background and similar_to_primary:
                retained.append(int(idx))
        diagnostics["pruned"] = len(retained) != int(active.size)
        active = np.asarray(retained, dtype=int)
    if active.size > max(0, int(kmax)):
        order = active[np.argsort(rate_arr[active])[::-1]]
        active = np.sort(order[: max(0, int(kmax))])
        diagnostics["truncated"] = True
    else:
        active = np.sort(active)
    diagnostics["n_retained"] = int(active.size)
    active_tuple = tuple(int(idx) for idx in active.tolist())
    if not active_tuple:
        return "LOW", active_tuple, diagnostics
    return "+".join(names[idx] for idx in active_tuple), active_tuple, diagnostics


def _make_canonical_result(
    result: Any,
    data: Any,
    full_data: Any,
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
        rows, summary = _episode_rows(
            result,
            condition=condition,
            title=title,
            seed=seed,
            exclude_edges=exclude_edges,
        )
        return result, rows, summary, [], []

    names = _cluster_names(data)
    full_names = _cluster_names(full_data)
    labels = np.zeros(int(data.n_timepoints), dtype=np.int64)
    episode_keys: list[str] = []
    episode_active: list[tuple[int, ...]] = []
    episode_diagnostics: list[dict[str, Any]] = []
    for episode in episodes_meta:
        start = int(episode["start"])
        stop = int(episode["stop"])
        mean_rates = np.nanmean(np.asarray(data.X_rate, dtype=float)[start:stop], axis=0)
        key, active, diagnostics = _canonical_state_key(
            np.asarray(episode["mask"], dtype=bool),
            mean_rates,
            names,
            kmax=int(kmax),
            z_threshold=float(z_threshold),
            similarity=float(similarity),
            noise_floor=float(noise_floor),
        )
        episode_keys.append(key)
        episode_active.append(active)
        episode_diagnostics.append(diagnostics)

    ordered_keys = sorted(set(episode_keys), key=lambda item: (0 if item == "LOW" else item.count("+") + 1, item))
    key_to_id = {key: idx for idx, key in enumerate(ordered_keys)}
    for episode, key in zip(episodes_meta, episode_keys):
        labels[int(episode["start"]) : int(episode["stop"])] = int(key_to_id[key])

    segments = extract_segments(labels, data.dt)
    id_to_key = {idx: key for key, idx in key_to_id.items()}
    key_to_active = {
        key: episode_active[episode_keys.index(key)]
        for key in ordered_keys
    }
    if not segments.empty:
        segments["state_key"] = [id_to_key[int(state)] for state in segments["state"]]
        segments["K"] = [len(key_to_active[str(key)]) for key in segments["state_key"]]
        segments["clusters"] = [
            ",".join(names[idx] for idx in key_to_active[str(key)]) for key in segments["state_key"]
        ]

    state_means = np.zeros((len(ordered_keys), data.n_clusters), dtype=float)
    full_state_means = np.zeros((len(ordered_keys), full_data.n_clusters), dtype=float)
    occupancy: dict[int, float] = {}
    for state_id, key in enumerate(ordered_keys):
        selected = labels == state_id
        occupancy[state_id] = float(np.mean(selected)) if labels.size else 0.0
        if np.any(selected):
            state_means[state_id] = np.nanmean(np.asarray(data.X_rate, dtype=float)[selected], axis=0)
            full_state_means[state_id] = np.nanmean(np.asarray(full_data.X_rate, dtype=float)[selected], axis=0)
    canonical = SimpleNamespace(
        method="active_set_em_canonical",
        labels=labels,
        segments=segments,
        n_states=len(ordered_keys),
        state_means=state_means,
        state_occupancy=occupancy,
        metadata={
            **dict(result.metadata),
            "status": str(result.metadata.get("status", "ok")),
            "canonical_state_keys": ordered_keys,
            "canonical_cluster_names": names,
            "canonical_kmax": int(kmax),
            "canonical_z_threshold": float(z_threshold),
            "canonical_similarity": float(similarity),
            "canonical_noise_floor": float(noise_floor),
            "raw_n_states": int(result.n_states),
            "raw_n_episodes": int(len(result.segments)),
            "canonical_pruned_episodes": int(sum(bool(item["pruned"]) for item in episode_diagnostics)),
            "canonical_truncated_episodes": int(sum(bool(item["truncated"]) for item in episode_diagnostics)),
        },
    )
    rows, summary = _episode_rows(
        canonical,
        condition=condition,
        title=title,
        seed=seed,
        exclude_edges=exclude_edges,
    )
    summary["raw_n_states"] = int(result.n_states)
    summary["raw_n_episodes"] = int(len(result.segments))
    summary["canonical_pruned_episodes"] = int(sum(bool(item["pruned"]) for item in episode_diagnostics))
    summary["canonical_truncated_episodes"] = int(sum(bool(item["truncated"]) for item in episode_diagnostics))
    summary["canonical_z_threshold"] = float(z_threshold)
    summary["canonical_similarity"] = float(similarity)

    inventory_rows: list[dict[str, Any]] = []
    for state_id, key in enumerate(ordered_keys):
        selected_segments = segments.loc[segments["state"] == state_id] if not segments.empty else pd.DataFrame()
        dwell = selected_segments["duration_time"].to_numpy(dtype=float) if not selected_segments.empty else np.zeros(0)
        active = key_to_active[key]
        inventory_rows.append(
            {
                "condition": condition,
                "title": title,
                "seed": int(seed),
                "state": int(state_id),
                "state_key": key,
                "K": int(len(active)),
                "clusters": ",".join(names[idx] for idx in active),
                "visits": int(dwell.size),
                "occupancy_time_s": float(np.sum(labels == state_id) * data.dt),
                "occupancy_fraction": float(occupancy[state_id]),
                "mean_dwell_time_s": float(np.mean(dwell)) if dwell.size else np.nan,
                "std_dwell_time_s": float(np.std(dwell, ddof=1)) if dwell.size > 1 else 0.0 if dwell.size else np.nan,
                "median_dwell_time_s": float(np.median(dwell)) if dwell.size else np.nan,
                "first_seen_s": float(selected_segments["start_time"].min()) if not selected_segments.empty else np.nan,
            }
        )

    emission_rows: list[dict[str, Any]] = []
    for state_id, key in enumerate(ordered_keys):
        row = {
            "condition": condition,
            "title": title,
            "seed": int(seed),
            "state": int(state_id),
            "state_key": key,
            "K": int(len(key_to_active[key])),
        }
        for cluster_name, value in zip(full_names, full_state_means[state_id]):
            row[f"rate_{cluster_name}"] = float(value)
        emission_rows.append(row)
    canonical.metadata["canonical_full_cluster_names"] = full_names
    canonical.metadata["canonical_full_state_means"] = full_state_means
    return canonical, rows, summary, inventory_rows, emission_rows


def _save_canonical_artifacts(
    output_dir: Path,
    canonical: Any,
    inventory_rows: list[dict[str, Any]],
    emission_rows: list[dict[str, Any]],
    *,
    condition: str,
    seed: int,
) -> None:
    target_dir = output_dir / "canonical"
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{condition}_seed_{int(seed)}"
    canonical.segments.to_csv(target_dir / f"{prefix}_segments.csv", index=False)
    pd.DataFrame(inventory_rows).to_csv(target_dir / f"{prefix}_state_inventory.csv", index=False)
    pd.DataFrame(emission_rows).to_csv(target_dir / f"{prefix}_state_emissions.csv", index=False)
    np.save(target_dir / f"{prefix}_state_sequence.npy", np.asarray(canonical.labels, dtype=np.int64))
    np.savez_compressed(
        target_dir / f"{prefix}_state_emissions.npz",
        state_ids=np.arange(int(canonical.n_states), dtype=np.int64),
        state_keys=np.asarray(canonical.metadata.get("canonical_state_keys", []), dtype=str),
        analyzed_cluster_names=np.asarray(canonical.metadata.get("canonical_cluster_names", []), dtype=str),
        full_cluster_names=np.asarray(canonical.metadata.get("canonical_full_cluster_names", []), dtype=str),
        analyzed_state_means=np.asarray(canonical.state_means, dtype=float),
        full_state_means=np.asarray(canonical.metadata.get("canonical_full_state_means", []), dtype=float),
        labels=np.asarray(canonical.labels, dtype=np.int64),
    )


def _plot(summary: pd.DataFrame, condition_order: list[str], output_prefix: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(condition_order), figsize=(4.0 * len(condition_order), 4.2), sharey=True)
    axes_arr = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, condition in zip(axes_arr, condition_order):
        group = summary.loc[summary["condition"] == condition].sort_values("seed")
        values = group["mean_dwell_time_s"].to_numpy(dtype=float)
        within_std = group["std_dwell_time_s"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        x = np.arange(1, len(group) + 1, dtype=float)
        jittered_x = x[finite] + rng.uniform(-0.08, 0.08, finite.sum())
        ax.errorbar(
            jittered_x,
            values[finite],
            yerr=np.where(np.isfinite(within_std[finite]), within_std[finite], 0.0),
            fmt="o",
            color="#444444",
            ecolor="#AAAAAA",
            markersize=4,
            capsize=2,
            linewidth=0.8,
            zorder=3,
        )
        if finite.any():
            mean = float(np.mean(values[finite]))
            std = float(np.std(values[finite], ddof=1)) if finite.sum() > 1 else 0.0
            ax.axhspan(max(0.0, mean - std), mean + std, color="#4C78A8", alpha=0.2, label="network mean +/- SD")
            ax.axhline(mean, color="#4C78A8", linewidth=2)
            ax.errorbar(len(group) + 1.0, mean, yerr=std, fmt="o", color="#D62728", capsize=4, label="aggregate")
        ax.set_title(str(group["title"].iloc[0]) if not group.empty else condition)
        ax.set_xlabel("Network realization")
        ax.set_xticks([1, max(1, len(group)), len(group) + 1])
        ax.set_xticklabels(["1", str(max(1, len(group))), "mean"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_arr[0].set_ylabel("Mean state dwell time [s]")
    handles, labels = axes_arr[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)


def _plot_state_raster_overlay(
    trace_path: str | Path,
    result: Any,
    *,
    parameter: dict[str, Any],
    updates_per_second: float,
    title: str,
    output_path: Path,
    duration: float | None,
    raster_stride: int,
    max_raster_events: int,
    dpi: int,
) -> None:
    with np.load(trace_path, allow_pickle=False) as payload:
        spike_times = np.asarray(payload["spike_times"], dtype=float) / float(updates_per_second)
        spike_ids = np.asarray(payload["spike_ids"], dtype=np.int64)
    segments = result.segments
    inferred_stop = float(segments["stop_time"].max()) if not segments.empty else 0.0
    t_stop = inferred_stop if duration is None else min(float(duration), inferred_stop)
    if t_stop <= 0.0:
        t_stop = float(duration or 1.0)
    stride = max(1, int(raster_stride))
    keep = (spike_times >= 0.0) & (spike_times <= t_stop) & ((spike_ids % stride) == 0)
    spike_times = spike_times[keep]
    spike_ids = spike_ids[keep]
    event_limit = int(max_raster_events)
    if event_limit > 0 and spike_times.size > event_limit:
        selected = np.linspace(0, spike_times.size - 1, event_limit, dtype=np.int64)
        spike_times = spike_times[selected]
        spike_ids = spike_ids[selected]
    n_exc = int(parameter.get("N_E", 0) or 0)
    n_inh = int(parameter.get("N_I", 0) or 0)
    cmap = plt.get_cmap("tab20")

    fig, (ax_raster, ax_state) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.55], "hspace": 0.06},
    )
    for _, segment in segments.iterrows():
        start = max(0.0, float(segment["start_time"]))
        stop = min(t_stop, float(segment["stop_time"]))
        if stop <= start:
            continue
        state = int(segment["state"])
        key = str(segment.get("state_key", state))
        color = cmap(state % cmap.N)
        ax_raster.axvspan(start, stop, color=color, alpha=0.12, linewidth=0)
        ax_state.axvspan(start, stop, color=color, alpha=0.9, linewidth=0)
        ax_raster.axvline(start, color=color, alpha=0.6, linewidth=0.6)
        if stop - start >= 0.03 * t_stop:
            ax_state.text((start + stop) / 2.0, 0.5, key, ha="center", va="center", fontsize=7)
    plot_spike_raster(
        ax_raster,
        spike_times,
        spike_ids,
        n_exc=n_exc,
        n_inh=n_inh,
        stride=1,
        t_start=0.0,
        t_end=t_stop,
        marker=".",
        marker_size=1.2,
    )
    ax_raster.set_ylabel("Neuron")
    ax_raster.set_title(title)
    ax_state.set_xlim(0.0, t_stop)
    ax_state.set_ylim(0.0, 1.0)
    ax_state.set_yticks([])
    ax_state.set_ylabel("State", rotation=0, ha="right", va="center")
    ax_state.set_xlabel("Time [s]")
    for ax in (ax_raster, ax_state):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_activity_overlay(
    data: Any,
    result: Any,
    *,
    title: str,
    output_path: Path,
    duration: float | None,
    dpi: int,
) -> None:
    rates = np.asarray(data.X_rate, dtype=float)
    if rates.ndim != 2 or rates.shape[0] == 0:
        return
    segments = result.segments
    inferred_stop = float(segments["stop_time"].max()) if not segments.empty else float(rates.shape[0] * data.dt)
    t_stop = inferred_stop if duration is None else min(float(duration), inferred_stop)
    stop_bin = max(1, min(rates.shape[0], int(np.ceil(t_stop / float(data.dt)))))
    shown = rates[:stop_bin]
    t_stop = min(t_stop, stop_bin * float(data.dt))
    vmax = float(np.nanquantile(shown, 0.99)) if shown.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = None
    names = _cluster_names(data)
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.025],
        height_ratios=[5, 0.55],
        hspace=0.06,
        wspace=0.04,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_state = fig.add_subplot(grid[1, 0], sharex=ax_heat)
    cax = fig.add_subplot(grid[0, 1])
    spacer_ax = fig.add_subplot(grid[1, 1])
    spacer_ax.axis("off")
    image = ax_heat.imshow(
        shown.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(0.0, t_stop, -0.5, shown.shape[1] - 0.5),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    for _, segment in segments.iterrows():
        start = max(0.0, float(segment["start_time"]))
        stop = min(t_stop, float(segment["stop_time"]))
        if stop <= start:
            continue
        state = int(segment["state"])
        key = str(segment.get("state_key", state))
        color = cmap(state % cmap.N)
        ax_heat.axvline(start, color=color, alpha=0.75, linewidth=0.65)
        ax_state.axvspan(start, stop, color=color, alpha=0.9, linewidth=0)
        if stop - start >= 0.03 * t_stop:
            ax_state.text((start + stop) / 2.0, 0.5, key, ha="center", va="center", fontsize=6)
    ax_heat.set_ylabel("Cluster")
    ax_heat.set_title(title)
    if shown.shape[1] <= 50:
        ax_heat.set_yticks(np.arange(shown.shape[1]))
        ax_heat.set_yticklabels(names[: shown.shape[1]], fontsize=7)
    else:
        ax_heat.set_yticks([])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Population rate")
    ax_heat.tick_params(axis="x", labelbottom=False)
    ax_state.set_xlim(0.0, t_stop)
    ax_state.set_ylim(0.0, 1.0)
    ax_state.set_yticks([])
    ax_state.set_ylabel("State", rotation=0, ha="right", va="center")
    ax_state.set_xlabel("Time [s]")
    for ax in (ax_heat, ax_state):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _analyze_context(
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    analysis_start = time.perf_counter()
    context = task["context"]
    dt_seconds = float(context["binary_cfg"]["sample_interval"]) / float(context["updates_per_second"])
    preprocessing_cfg, method_cfg = _analysis_configs(task["args"], dt_seconds)
    full_data = analysis_input_from_binary_trace(
        context["trace_path"],
        parameter=context["parameter"],
        analysis_cfg={"dt": dt_seconds},
    )
    data = full_data
    if str(task["args"].population_source) == "excitatory":
        if data.cluster_cell_types is not None:
            indices = [
                idx
                for idx, cell_type in enumerate(data.cluster_cell_types)
                if str(cell_type).upper().startswith("E")
            ]
        else:
            indices = [
                idx
                for idx, name in enumerate(data.cluster_names or [])
                if str(name).upper().startswith("E")
            ]
        data = subset_analysis_input(data, indices=indices)
    result = run_active_set_em(data, preprocessing_cfg, method_cfg)
    if str(task["args"].state_canonicalization) == "active_set":
        analysis_result, episodes, summary, inventory_rows, emission_rows = _make_canonical_result(
            result,
            data,
            full_data,
            condition=context["condition"],
            title=context["title"],
            seed=context["seed"],
            exclude_edges=bool(task["args"].exclude_edge_dwells),
            kmax=int(task["args"].canonical_kmax),
            z_threshold=float(task["args"].canonical_z_threshold),
            similarity=float(task["args"].canonical_similarity),
            noise_floor=float(task["args"].canonical_noise_floor),
        )
    else:
        analysis_result = result
        episodes, summary = _episode_rows(
            result,
            condition=context["condition"],
            title=context["title"],
            seed=context["seed"],
            exclude_edges=bool(task["args"].exclude_edge_dwells),
        )
        inventory_rows = []
        emission_rows = []
    if inventory_rows or emission_rows:
        _save_canonical_artifacts(
            Path(task["output_dir"]),
            analysis_result,
            inventory_rows,
            emission_rows,
            condition=context["condition"],
            seed=context["seed"],
        )
    inspection_path = Path(task["output_dir"]) / "inspection" / f"{context['condition']}_seed_{context['seed']}.png"
    activity_inspection_path = (
        Path(task["output_dir"])
        / "inspection"
        / "cluster_activity"
        / f"{context['condition']}_seed_{context['seed']}.png"
    )
    if bool(task["args"].inspection_plots):
        _plot_state_raster_overlay(
            context["trace_path"],
            analysis_result,
            parameter=context["parameter"],
            updates_per_second=float(context["updates_per_second"]),
            title=f"{context['title']}, seed={context['seed']}",
            output_path=inspection_path,
            duration=task["args"].inspection_duration,
            raster_stride=int(task["args"].raster_stride),
            max_raster_events=int(task["args"].max_raster_events),
            dpi=int(task["args"].dpi),
        )
        _plot_cluster_activity_overlay(
            full_data,
            analysis_result,
            title=f"{context['title']}, seed={context['seed']}",
            output_path=activity_inspection_path,
            duration=task["args"].inspection_duration,
            dpi=int(task["args"].dpi),
        )
    summary["trace_path"] = str(context["trace_path"])
    summary["inspection_plot"] = str(inspection_path) if bool(task["args"].inspection_plots) else ""
    summary["activity_inspection_plot"] = (
        str(activity_inspection_path) if bool(task["args"].inspection_plots) else ""
    )
    summary["analysis_seconds"] = float(time.perf_counter() - analysis_start)
    summary["population_source"] = str(task["args"].population_source)
    summary["n_analyzed_populations"] = int(data.n_clusters)
    summary["state_canonicalization"] = str(task["args"].state_canonicalization)
    return episodes, summary, inventory_rows, emission_rows


def _worker_count(requested: int, total: int, label: str) -> int:
    workers = min(max(1, int(requested)), max(1, int(total)))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        available = max(1, int(slurm_cpus))
        if workers > available:
            print(
                f"Reducing {label} workers from {workers} to SLURM_CPUS_PER_TASK={available}.",
                flush=True,
            )
            workers = available
    return workers


def _progress(iterable: Any, *, total: int, description: str) -> Any:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=description, unit="run", dynamic_ncols=True)

    def text_progress() -> Any:
        start = time.perf_counter()
        width = 24
        for completed, item in enumerate(iterable, start=1):
            elapsed = time.perf_counter() - start
            fraction = completed / max(1, total)
            filled = min(width, int(round(width * fraction)))
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r{description}: [{bar}] {completed}/{total} "
                f"elapsed={elapsed / 60.0:.1f}m rate={rate:.2f}/s",
                end="\n" if completed == total else "",
                flush=True,
            )
            yield item

    return text_progress()


def _write_analysis_checkpoints(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    network_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    emission_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(episode_rows).to_csv(output_dir / "dwell_episodes.partial.csv", index=False)
    pd.DataFrame(network_rows).to_csv(output_dir / "network_dwell_summary.partial.csv", index=False)
    pd.DataFrame(inventory_rows).to_csv(output_dir / "state_inventory.partial.csv", index=False)
    pd.DataFrame(emission_rows).to_csv(output_dir / "state_emissions.partial.csv", index=False)


def _pool_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("spawn")
    except ValueError:  # pragma: no cover
        return mp.get_context()


def run(args: argparse.Namespace) -> Path:
    if args.n_simulations <= 0:
        raise ValueError("--n-simulations must be positive.")
    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    base_parameter = load_from_args(args)
    if base_parameter.get("R_Eplus") is None:
        base_parameter["R_Eplus"] = 7.25
    column_overrides = _parse_column_override_entries(args.column_override)
    column_titles = {
        _normalize_label(raw.split(":", 1)[0]): raw.split(":", 1)[1]
        for raw in args.column_title
        if ":" in raw
    }
    known_labels = {_normalize_label(spec.label) for spec in COLUMN_SPECS}
    _validate_column_keys(column_overrides, known_labels, "column overrides")
    _validate_column_keys(column_titles, known_labels, "column titles")
    parsed_focus_counts = parse_int_values(args.focus_counts, option_name="--focus-counts")
    selected_labels = set(args.columns or [spec.label for spec in COLUMN_SPECS])
    selected_specs = [spec for spec in COLUMN_SPECS if spec.label in selected_labels]
    sweep_cfg = PipelineSweepSettings(jobs=max(1, int(args.erf_jobs)), plot_erfs=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for spec in selected_specs:
        parameter = _build_column_parameter(base_parameter, spec, column_override_map=column_overrides)
        title = _resolve_column_title(parameter, spec, title_map=column_titles)
        focus_counts = _resolve_focus_counts(parameter, parsed_focus_counts)
        _markers, candidates, _folder, _bundle = _prepare_focus_markers_with_retry(
            parameter,
            focus_counts,
            args.stability_filter,
            sweep_cfg,
            delta_rep=float(args.delta_rep_mf),
            rep_retry=int(args.rep_retry_mf),
            rng_seed=args.rep_rng_seed,
            column_label=spec.label,
        )
        updates_per_second, _label = _time_axis_scale_from_taus(parameter)
        simulation_steps = max(1, int(round(float(args.duration) * updates_per_second)))
        sample_interval = (
            int(args.sample_interval)
            if args.sample_interval is not None
            else max(1, int(round(0.010 * updates_per_second)))
        )
        binary_cfg = _resolve_binary_config(
            parameter,
            BinaryRunSettings(
                warmup_steps=args.warmup_steps,
                simulation_steps=simulation_steps,
                sample_interval=sample_interval,
                batch_size=args.batch_size,
                output_name=f"figure3_dwell_{spec.label}",
            ),
        )
        sample_count = int(np.ceil(simulation_steps / sample_interval))
        print(
            f"Condition {spec.label}: {simulation_steps:,} updates, sample_interval={sample_interval:,} "
            f"({sample_count:,} analysis samples).",
            flush=True,
        )
        for offset in range(int(args.n_simulations)):
            seed = int(args.seed_start) + offset
            trace_path, task = _build_trace_task(
                parameter,
                binary_cfg,
                candidates=candidates,
                seed=seed,
                analysis_only=bool(args.analysis_only),
                overwrite_simulation=bool(args.overwrite_simulation),
            )
            if task is not None:
                tasks.append(task)
            contexts.append(
                {
                    "condition": spec.label,
                    "title": title,
                    "parameter": parameter,
                    "binary_cfg": binary_cfg,
                    "seed": seed,
                    "trace_path": trace_path,
                    "updates_per_second": updates_per_second,
                }
            )
    trace_rows = [
        {
            "condition": context["condition"],
            "seed": int(context["seed"]),
            "trace_path": str(context["trace_path"]),
            "exists": bool(Path(context["trace_path"]).exists()),
        }
        for context in contexts
    ]
    pd.DataFrame(trace_rows).to_csv(output_dir / "trace_manifest.csv", index=False)

    if tasks:
        workers = _worker_count(int(args.jobs), len(tasks), "simulation")
        print(f"Running {len(tasks)} binary simulations with {workers} workers.", flush=True)
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
                futures = [pool.submit(_simulate_binary_task, task) for task in tasks]
                completed_futures = _progress(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    description="Simulations",
                )
                for completed, future in enumerate(completed_futures, start=1):
                    future.result()
        else:
            for completed, task in enumerate(_progress(tasks, total=len(tasks), description="Simulations"), start=1):
                _simulate_binary_task(task)
    else:
        print("All matching binary traces already exist; skipping simulations.", flush=True)

    episode_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    emission_rows: list[dict[str, Any]] = []
    existing_contexts: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for context in contexts:
        if Path(context["trace_path"]).exists():
            existing_contexts.append(context)
        else:
            missing_rows.append(
                {
                    "condition": context["condition"],
                    "seed": int(context["seed"]),
                    "trace_path": str(context["trace_path"]),
                }
            )
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(output_dir / "missing_traces.csv", index=False)
        message = f"{len(missing_rows)} expected trace(s) are missing; see {output_dir / 'missing_traces.csv'}."
        if bool(args.analysis_only) and bool(args.skip_missing_traces):
            print(f"{message} Skipping them.", flush=True)
        else:
            raise FileNotFoundError(message)
    if not existing_contexts:
        raise FileNotFoundError("No existing traces are available for analysis.")
    analysis_tasks = [{"context": context, "args": args, "output_dir": str(output_dir)} for context in existing_contexts]
    requested_analysis_jobs = args.analysis_jobs if args.analysis_jobs is not None else args.jobs
    analysis_workers = _worker_count(int(requested_analysis_jobs), len(analysis_tasks), "analysis")
    print(f"Running {len(analysis_tasks)} state-estimation analyses with {analysis_workers} workers.", flush=True)
    if analysis_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=analysis_workers, mp_context=_pool_context()) as pool:
            futures = [pool.submit(_analyze_context, task) for task in analysis_tasks]
            completed_futures = _progress(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                description="State analyses",
            )
            for completed, future in enumerate(completed_futures, start=1):
                episodes, summary, inventory, emissions = future.result()
                episode_rows.extend(episodes)
                network_rows.append(summary)
                inventory_rows.extend(inventory)
                emission_rows.extend(emissions)
                _write_analysis_checkpoints(output_dir, episode_rows, network_rows, inventory_rows, emission_rows)
    else:
        for completed, task in enumerate(
            _progress(analysis_tasks, total=len(analysis_tasks), description="State analyses"),
            start=1,
        ):
            episodes, summary, inventory, emissions = _analyze_context(task)
            episode_rows.extend(episodes)
            network_rows.append(summary)
            inventory_rows.extend(inventory)
            emission_rows.extend(emissions)
            _write_analysis_checkpoints(output_dir, episode_rows, network_rows, inventory_rows, emission_rows)

    episodes_df = pd.DataFrame(episode_rows)
    networks_df = pd.DataFrame(network_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    emissions_df = pd.DataFrame(emission_rows)
    episodes_df.to_csv(output_dir / "dwell_episodes.csv", index=False)
    networks_df.to_csv(output_dir / "network_dwell_summary.csv", index=False)
    inventory_df.to_csv(output_dir / "state_inventory.csv", index=False)
    emissions_df.to_csv(output_dir / "state_emissions.csv", index=False)
    (output_dir / "dwell_episodes.partial.csv").unlink(missing_ok=True)
    (output_dir / "network_dwell_summary.partial.csv").unlink(missing_ok=True)
    (output_dir / "state_inventory.partial.csv").unlink(missing_ok=True)
    (output_dir / "state_emissions.partial.csv").unlink(missing_ok=True)
    aggregate = (
        networks_df.groupby(["condition", "title"], as_index=False)
        .agg(
            network_count=("seed", "count"),
            mean_dwell_time_s=("mean_dwell_time_s", "mean"),
            std_across_networks_s=("mean_dwell_time_s", "std"),
            mean_state_changes=("n_state_changes", "mean"),
            std_state_changes=("n_state_changes", "std"),
        )
    )
    aggregate.to_csv(output_dir / "aggregate_dwell_summary.csv", index=False)
    write_yaml_config(vars(args), output_dir / "run_config.yaml")
    _plot(networks_df, [spec.label for spec in selected_specs], Path(args.output_prefix), int(args.dpi))
    return output_dir


def main() -> None:
    output_dir = run(parse_args())
    print(f"Wrote Figure 3 dwell-time analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
