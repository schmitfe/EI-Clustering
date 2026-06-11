#!/usr/bin/env python3
"""Estimate Figure 3 state dwell times across independently seeded networks."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
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
    _load_trace_payload,
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
from figure_cli import parse_int_values  # noqa: E402
from plotting import plot_spike_raster  # noqa: E402
from sim_config import add_override_arguments, load_from_args, write_yaml_config  # noqa: E402


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
    parser.add_argument("--erf-jobs", type=int, default=1, help="Parallel workers for mean-field preparation.")
    parser.add_argument("--warmup-steps", type=int, help="Override binary warmup steps.")
    parser.add_argument("--sample-interval", type=int, help="Override binary sample interval.")
    parser.add_argument("--batch-size", type=int, help="Override binary batch size.")
    parser.add_argument("--focus-counts", nargs="+", help="Focus counts, e.g. 5:1:-1.")
    parser.add_argument("--stability-filter", choices=("stable", "unstable", "any"), default="any")
    parser.add_argument("--column-override", action="append", default=[], metavar="label:path=value")
    parser.add_argument("--column-title", action="append", default=[], metavar="label:title")
    parser.add_argument("--delta-rep-mf", type=float, default=0.025)
    parser.add_argument("--rep-retry-mf", type=int, default=10)
    parser.add_argument("--rep-rng-seed", type=int)
    parser.add_argument("--analysis-only", action="store_true", help="Reuse existing matching traces.")
    parser.add_argument("--overwrite-simulation", action="store_true")
    parser.add_argument("--pelt-penalty", type=float, default=10.0)
    parser.add_argument("--pelt-min-size", type=int, default=3)
    parser.add_argument("--pelt-smooth-width", type=int, default=3)
    parser.add_argument("--beta-merge", type=float, default=0.0)
    parser.add_argument("--min-flicker-duration", type=int, default=3)
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
        "segmentation": "pelt",
        "pelt_penalty": float(args.pelt_penalty),
        "pelt_min_size": int(args.pelt_min_size),
        "pelt_feature_mode": "weighted",
        "pelt_smooth_width": int(args.pelt_smooth_width),
        "Kmax": None,
        "lambda_active": 0.0,
        "lambda_comb": 0.1,
        "min_separation": 0.05,
        "var_floor": 0.0001,
        "max_iter": 100,
        "tol": 1e-6,
        "flat_range_threshold": 1e-12,
        "merge_after_em": True,
        "beta_merge": float(args.beta_merge),
        "min_flicker_duration": int(args.min_flicker_duration),
        "flicker_max_hamming": 2,
        "merge_max_iter": 100,
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
    dpi: int,
) -> None:
    payload = _load_trace_payload(str(trace_path))
    spike_times = np.asarray(payload["spike_times"], dtype=float) / float(updates_per_second)
    spike_ids = np.asarray(payload["spike_ids"], dtype=np.int64)
    segments = result.segments
    inferred_stop = float(segments["stop_time"].max()) if not segments.empty else 0.0
    t_stop = inferred_stop if duration is None else min(float(duration), inferred_stop)
    if t_stop <= 0.0:
        t_stop = float(duration or 1.0)
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
        color = cmap(state % cmap.N)
        ax_raster.axvspan(start, stop, color=color, alpha=0.12, linewidth=0)
        ax_state.axvspan(start, stop, color=color, alpha=0.9, linewidth=0)
        ax_raster.axvline(start, color=color, alpha=0.6, linewidth=0.6)
        if stop - start >= 0.03 * t_stop:
            ax_state.text((start + stop) / 2.0, 0.5, str(state), ha="center", va="center", fontsize=7)
    plot_spike_raster(
        ax_raster,
        spike_times,
        spike_ids,
        n_exc=n_exc,
        n_inh=n_inh,
        stride=max(1, int(raster_stride)),
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
    sweep_cfg = PipelineSweepSettings(jobs=max(1, int(args.erf_jobs)), plot_erfs=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for spec in COLUMN_SPECS:
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
        binary_cfg = _resolve_binary_config(
            parameter,
            BinaryRunSettings(
                warmup_steps=args.warmup_steps,
                simulation_steps=simulation_steps,
                sample_interval=args.sample_interval,
                batch_size=args.batch_size,
                output_name=f"figure3_dwell_{spec.label}",
            ),
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

    if tasks:
        workers = min(max(1, int(args.jobs)), len(tasks))
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_simulate_binary_task, tasks))
        else:
            for task in tasks:
                _simulate_binary_task(task)

    episode_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    for context in contexts:
        dt_seconds = float(context["binary_cfg"]["sample_interval"]) / float(context["updates_per_second"])
        preprocessing_cfg, method_cfg = _analysis_configs(args, dt_seconds)
        data = analysis_input_from_binary_trace(
            context["trace_path"],
            parameter=context["parameter"],
            analysis_cfg={"dt": dt_seconds},
        )
        result = run_active_set_em(data, preprocessing_cfg, method_cfg)
        inspection_path = output_dir / "inspection" / f"{context['condition']}_seed_{context['seed']}.png"
        if bool(args.inspection_plots):
            _plot_state_raster_overlay(
                context["trace_path"],
                result,
                parameter=context["parameter"],
                updates_per_second=float(context["updates_per_second"]),
                title=f"{context['title']}, seed={context['seed']}",
                output_path=inspection_path,
                duration=args.inspection_duration,
                raster_stride=int(args.raster_stride),
                dpi=int(args.dpi),
            )
        episodes, summary = _episode_rows(
            result,
            condition=context["condition"],
            title=context["title"],
            seed=context["seed"],
            exclude_edges=bool(args.exclude_edge_dwells),
        )
        summary["trace_path"] = str(context["trace_path"])
        summary["inspection_plot"] = str(inspection_path) if bool(args.inspection_plots) else ""
        episode_rows.extend(episodes)
        network_rows.append(summary)

    episodes_df = pd.DataFrame(episode_rows)
    networks_df = pd.DataFrame(network_rows)
    episodes_df.to_csv(output_dir / "dwell_episodes.csv", index=False)
    networks_df.to_csv(output_dir / "network_dwell_summary.csv", index=False)
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
    _plot(networks_df, [spec.label for spec in COLUMN_SPECS], Path(args.output_prefix), int(args.dpi))
    return output_dir


def main() -> None:
    output_dir = run(parse_args())
    print(f"Wrote Figure 3 dwell-time analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
