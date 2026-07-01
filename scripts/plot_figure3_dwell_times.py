#!/usr/bin/env python3
"""Estimate Figure 3 state dwell times across independently seeded networks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
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
from analysis.episode_inference import (  # noqa: E402
    cluster_names as _cluster_names,
    infer_active_set_episodes,
)
from analyze_weights import analyze_weights  # noqa: E402
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
    parser.add_argument(
        "--simulation-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-network matrix/state/emission collection files in output_dir/SimulationCollection.",
    )
    parser.add_argument(
        "--collection-include-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Embed the actual dense or CSR weight matrix arrays in each SimulationCollection npz.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    return value


def _matching_weights_path(trace_path: str | Path) -> Path:
    path = Path(trace_path)
    return path.with_name(f"{path.stem}_weights.npz")


def _load_weight_arrays(weights_path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(weights_path, allow_pickle=True) as payload:
        for key in (
            "weight_format",
            "weight_mode",
            "weight_dtype",
            "weight_shape",
            "weights",
            "weights_data",
            "weights_indices",
            "weights_indptr",
            "population_names",
            "population_start_ids",
            "population_end_ids",
            "population_sizes",
            "population_cell_types",
            "population_cluster_indices",
        ):
            if key in payload:
                arrays[f"weight_{key}"] = payload[key]
    return arrays


def _state_count_summary(inventory_rows: list[dict[str, Any]]) -> dict[str, int]:
    n_states = len(inventory_rows)
    one_cluster = sum(1 for row in inventory_rows if int(row.get("K", 0) or 0) == 1)
    multi_cluster = sum(1 for row in inventory_rows if int(row.get("K", 0) or 0) > 1)
    low = sum(1 for row in inventory_rows if int(row.get("K", 0) or 0) == 0)
    return {
        "n_states": int(n_states),
        "n_low_states": int(low),
        "n_one_cluster_states": int(one_cluster),
        "n_multi_cluster_states": int(multi_cluster),
    }


def _save_simulation_collection(
    output_dir: Path,
    *,
    context: dict[str, Any],
    args: argparse.Namespace,
    result: Any,
    full_data: Any,
    summary: dict[str, Any],
    inventory_rows: list[dict[str, Any]],
    emission_rows: list[dict[str, Any]],
    include_weights: bool,
) -> str:
    collection_dir = output_dir / "SimulationCollection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    condition = str(context["condition"])
    seed = int(context["seed"])
    collection_path = collection_dir / f"{condition}_seed{seed:06d}.npz"
    weights_path = _matching_weights_path(context["trace_path"])

    payload: dict[str, Any] = {
        "condition": np.array(condition),
        "title": np.array(str(context["title"])),
        "seed": np.array(seed, dtype=np.int64),
        "trace_path": np.array(str(context["trace_path"])),
        "weights_path": np.array(str(weights_path)),
        "weights_available": np.array(bool(weights_path.exists())),
        "state_sequence": np.asarray(result.labels, dtype=np.int64),
        "state_segments_start_bin": result.segments["start_bin"].to_numpy(dtype=np.int64)
        if not result.segments.empty
        else np.zeros(0, dtype=np.int64),
        "state_segments_stop_bin": result.segments["stop_bin"].to_numpy(dtype=np.int64)
        if not result.segments.empty
        else np.zeros(0, dtype=np.int64),
        "state_segments_start_time_s": result.segments["start_time"].to_numpy(dtype=float)
        if not result.segments.empty
        else np.zeros(0, dtype=float),
        "state_segments_stop_time_s": result.segments["stop_time"].to_numpy(dtype=float)
        if not result.segments.empty
        else np.zeros(0, dtype=float),
        "state_segments_duration_s": result.segments["duration_time"].to_numpy(dtype=float)
        if not result.segments.empty
        else np.zeros(0, dtype=float),
        "state_segments_state": result.segments["state"].to_numpy(dtype=np.int64)
        if not result.segments.empty
        else np.zeros(0, dtype=np.int64),
        "state_segments_state_key": result.segments["state_key"].to_numpy(dtype=str)
        if "state_key" in result.segments
        else np.asarray([], dtype=str),
        "state_keys": np.asarray(result.metadata.get("canonical_state_keys", []), dtype=str),
        "analyzed_cluster_names": np.asarray(result.metadata.get("canonical_cluster_names", []), dtype=str),
        "full_cluster_names": np.asarray(result.metadata.get("canonical_full_cluster_names", _cluster_names(full_data)), dtype=str),
        "state_emissions_analyzed": np.asarray(result.state_means, dtype=float),
        "state_emissions_full": np.asarray(result.metadata.get("canonical_full_state_means", np.zeros((0, 0))), dtype=float),
        "state_inventory_json": np.array(json.dumps(_json_default(inventory_rows), sort_keys=True)),
        "state_emissions_json": np.array(json.dumps(_json_default(emission_rows), sort_keys=True)),
        "network_summary_json": np.array(json.dumps(_json_default(summary), sort_keys=True)),
        "simulation_parameter_json": np.array(json.dumps(_json_default(context["parameter"]), sort_keys=True)),
        "binary_config_json": np.array(json.dumps(_json_default(context["binary_cfg"]), sort_keys=True)),
        "run_args_json": np.array(json.dumps(_json_default(vars(args)), sort_keys=True)),
        "sample_dt_seconds": np.array(float(full_data.dt)),
    }
    count_summary = _state_count_summary(inventory_rows)
    for key, value in count_summary.items():
        payload[key] = np.array(int(value), dtype=np.int64)

    if weights_path.exists():
        weight_analysis = analyze_weights(str(weights_path))
        for key, value in weight_analysis.items():
            payload[f"matrix_analysis_{key}"] = np.asarray(value)
        if include_weights:
            payload.update(_load_weight_arrays(weights_path))

    np.savez_compressed(collection_path, **payload)
    return str(collection_path)


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
    full_data = analysis_input_from_binary_trace(
        context["trace_path"],
        parameter=context["parameter"],
        analysis_cfg={"dt": dt_seconds},
    )
    inference = infer_active_set_episodes(
        full_data,
        task["args"],
        condition=context["condition"],
        title=context["title"],
        seed=context["seed"],
        population_source=str(task["args"].population_source),
        canonicalize=str(task["args"].state_canonicalization) == "active_set",
        exclude_edges=bool(task["args"].exclude_edge_dwells),
    )
    data = inference.analyzed_data
    analysis_result = inference.result
    episodes = inference.episodes
    summary = inference.summary
    inventory_rows = inference.inventory
    emission_rows = inference.emissions
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
    if bool(task["args"].simulation_collection):
        collection_path = _save_simulation_collection(
            Path(task["output_dir"]),
            context=context,
            args=task["args"],
            result=analysis_result,
            full_data=full_data,
            summary=summary,
            inventory_rows=inventory_rows,
            emission_rows=emission_rows,
            include_weights=bool(task["args"].collection_include_weights),
        )
        summary["simulation_collection"] = collection_path
    else:
        summary["simulation_collection"] = ""
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
