from __future__ import annotations

import argparse
import os
import pickle
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ei_pipeline import (
    _key_to_r_eplus,
    _taggable_configuration,
    resolve_r_eplus,
    run_analysis,
    run_simulation,
)
from plot_config import DEFAULT_PLOT_CONFIG, PlotConfig
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg

PROB_KEYS = ("p0_ee", "p0_ei", "p0_ie", "p0_ii")
MARKER_STRIDE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 2x3 grid of fixpoint plots from cached ei_pipeline analyses."
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--rows",
        type=float,
        nargs="+",
        default=[0.3, 0.1],
        help="Average connectivity values defining the row order (default: %(default)s).",
    )
    parser.add_argument(
        "--columns",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Kappa values defining the column order (default: %(default)s).",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x-axis limit (default: autoscale).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper x-axis limit (default: autoscale).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=-0.075,
        help="Lower y-axis limit (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=1.075,
        help="Upper y-axis limit (default: %(default)s).",
    )
    parser.add_argument(
        "--marker-focus-count",
        type=int,
        default=1,
        help="Focus count rendered as markers (default: %(default)s).",
    )
    parser.add_argument(
        "--line-focus-counts",
        type=int,
        nargs="*",
        default=None,
        help="Other focus counts rendered as stable lines (omit to disable).",
    )
    parser.add_argument(
        "--extra-focus-counts",
        type=int,
        nargs="+",
        help="Additional focus counts to include during simulation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("plots", "fixpoint_grid.png"),
        help="Destination for the generated figure (default: %(default)s).",
    )
    parser.add_argument(
        "--write-pdf",
        action="store_true",
        default=True,
        help="Also export a PDF copy (default: enabled).",
    )
    parser.add_argument("--v-start", type=float, default=0.0)
    parser.add_argument("--v-end", type=float, default=1.0)
    parser.add_argument("--v-steps", type=int, default=1000)
    parser.add_argument("--retry-step", type=float, default=None)
    parser.add_argument("--r-eplus", type=float, action="append")
    parser.add_argument("--r-eplus-start", type=float, default=1.0, help="Default start for R_Eplus sweep (default: %(default)s).")
    parser.add_argument("--r-eplus-end", type=float, default=20.0, help="Default end for R_Eplus sweep (default: %(default)s).")
    parser.add_argument("--r-eplus-step", type=float, default=0.25, help="Default step for R_Eplus sweep (default: %(default)s).")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite-simulation", action="store_true")
    parser.add_argument("--overwrite-analysis", action="store_true")
    return parser.parse_args()


def _mean_connectivity(parameter: Mapping[str, float]) -> float | None:
    keys = ("N_E", "N_I", "p0_ee", "p0_ei", "p0_ie", "p0_ii")
    try:
        N_E, N_I, p_ee, p_ei, p_ie, p_ii = (float(parameter[key]) for key in keys)
    except (KeyError, TypeError, ValueError):
        return None
    denom = (N_E + N_I) ** 2
    if denom <= 0:
        return None
    numerator = (N_E ** 2) * p_ee + (N_E * N_I) * (p_ei + p_ie) + (N_I ** 2) * p_ii
    return numerator / denom


def _scale_probabilities(parameter: Mapping[str, float], target_avg: float) -> Dict[str, float]:
    base_avg = _mean_connectivity(parameter)
    if base_avg is None or base_avg <= 0:
        raise ValueError("Base configuration has invalid average connectivity.")
    factor = float(target_avg) / base_avg
    scaled = dict(parameter)
    for key in PROB_KEYS:
        if key not in parameter:
            continue
        scaled_value = float(parameter[key]) * factor
        scaled[key] = min(1.0, max(0.0, scaled_value))
    return scaled


def _coerce_focus_block(fixpoints: Mapping, focus_count: int):
    if not fixpoints:
        return {}
    if focus_count in fixpoints:
        return fixpoints[focus_count]
    if str(focus_count) in fixpoints:
        return fixpoints[str(focus_count)]
    return {}


def _collect_fixpoint_points(fixpoints: Mapping, focus_count: int) -> List[Tuple[float, float, str]]:
    block = _coerce_focus_block(fixpoints, focus_count)
    entries: List[Tuple[float, float, str]] = []
    for key, fixpoint_group in block.items():
        r_value = _key_to_r_eplus(key)
        if not np.isfinite(r_value):
            continue
        for point, info in fixpoint_group.items():
            included = info.get("included", True)
            if not included:
                continue
            stability = str(info.get("stability", "unstable"))
            entries.append((float(r_value), float(point), stability))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries


def _sparsify_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points or len(points) <= MARKER_STRIDE or MARKER_STRIDE <= 1:
        return points
    sampled = points[::MARKER_STRIDE]
    if points[-1] not in sampled:
        sampled.append(points[-1])
    return sampled


def _load_summary(path: str) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _summary_path(parameter: Mapping[str, float]) -> str:
    filtered = _taggable_configuration(parameter)
    analysis_tag = sim_tag_from_cfg(filtered)
    conn_label = str(parameter.get("connection_type", "bernoulli")).lower().replace(" ", "_")
    encoded_kappa = f"{float(parameter.get('kappa', 0.0)):.2f}".replace(".", "_")
    return os.path.join(
        "data",
        f"all_fixpoints_{conn_label}_kappa{encoded_kappa}_Rj{parameter.get('R_j', 0.0)}_{analysis_tag}.pkl",
    )


def _prepare_line_color_map(focus_counts: Sequence[int], palette: Sequence[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for idx, focus_count in enumerate(sorted(set(int(fc) for fc in focus_counts))):
        color = palette[idx % len(palette)]
        mapping[int(focus_count)] = color
    return mapping


def _ensure_fixpoint_summary(
    args: argparse.Namespace,
    parameter: Dict[str, float],
    focus_counts: Sequence[int],
) -> str:
    parameter = dict(parameter)
    parameter["focus_counts"] = list(sorted(set(int(fc) for fc in focus_counts)))
    summary_path = _summary_path(parameter)
    if not args.overwrite_analysis and os.path.exists(summary_path):
        return summary_path
    pipeline_args = SimpleNamespace(
        v_start=args.v_start,
        v_end=args.v_end,
        v_steps=args.v_steps,
        retry_step=args.retry_step,
        overwrite_simulation=args.overwrite_simulation,
        jobs=args.jobs,
        r_eplus=args.r_eplus,
        r_eplus_start=args.r_eplus_start,
        r_eplus_end=args.r_eplus_end,
        r_eplus_step=args.r_eplus_step,
    )
    r_values = resolve_r_eplus(pipeline_args, parameter)
    focus_counts = list(sorted(set(int(fc) for fc in focus_counts)))
    folder = run_simulation(pipeline_args, parameter, r_values, focus_counts)
    if folder is None:
        raise RuntimeError("Simulation did not produce any data folder.")
    run_analysis(folder, parameter, focus_counts, plot_erfs=False)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Expected summary file not found at {summary_path}")
    return summary_path


def plot_panel(
    ax: plt.Axes,
    *,
    summary: Mapping,
    marker_focus: int,
    line_focus_counts: Sequence[int],
    config: PlotConfig,
    letter: str,
    color_map: Dict[int, str],
) -> None:
    fixpoints = summary.get("fixpoints", {})
    marker_points = _collect_fixpoint_points(fixpoints, marker_focus)
    stable_marker = [(x, y) for x, y, status in marker_points if status == "stable"]
    unstable_marker = [(x, y) for x, y, status in marker_points if status != "stable"]
    stable_marker = _sparsify_points(stable_marker)
    unstable_marker = _sparsify_points(unstable_marker)
    if stable_marker:
        xs, ys = zip(*stable_marker)
        ax.scatter(
            xs,
            ys,
            s=35,
            color=config.palette.get("focus_stable", "#000000"),
        )
    if unstable_marker:
        xs, ys = zip(*unstable_marker)
        ax.scatter(
            xs,
            ys,
            s=35,
            facecolors="none",
            edgecolors=config.palette.get("focus_unstable", "#000000"),
        )
    for focus_count in line_focus_counts or []:
        points = _collect_fixpoint_points(fixpoints, focus_count)
        stable_points = [(x, y) for x, y, status in points if status == "stable"]
        if not stable_points:
            continue
        xs, ys = zip(*stable_points)
        color = color_map.get(focus_count, config.palette.get("line", "#2ca02c"))
        ax.plot(xs, ys, color=color, linewidth=1.5)
    label_x, label_y = config.panel_label_coords
    label_ha, label_va = config.panel_label_align
    panel_label = ax.text(
        label_x,
        label_y,
        f"{letter})",
        transform=ax.transAxes,
        ha=label_ha,
        va=label_va,
        fontweight="bold",
        clip_on=False,
    )
    panel_label.set_in_layout(False)


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    config = DEFAULT_PLOT_CONFIG
    config.apply()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    row_order = [float(val) for val in args.rows]
    col_order = [float(val) for val in args.columns]
    connection_type = base_parameter.get("connection_type")
    focus_counts = {args.marker_focus_count}
    if args.line_focus_counts:
        focus_counts.update(args.line_focus_counts)
    if args.extra_focus_counts:
        focus_counts.update(args.extra_focus_counts)
    focus_counts = sorted({max(1, int(fc)) for fc in focus_counts})
    n_rows = len(row_order)
    n_cols = len(col_order)
    fig_height = config.figure_height or config.figure_width * n_rows / n_cols * 0.65
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(config.figure_width, fig_height))
    axes = np.atleast_2d(axes)
    all_focus_counts = list(args.line_focus_counts or [])
    color_map = _prepare_line_color_map(all_focus_counts, config.line_colors)
    letters = [chr(ord("a") + idx) for idx in range(n_rows * n_cols)]
    for r_idx, avg_conn in enumerate(row_order):
        scaled_parameter = _scale_probabilities(base_parameter, avg_conn)
        actual_avg = _mean_connectivity(scaled_parameter) or avg_conn
        for c_idx, kappa in enumerate(col_order):
            ax = axes[r_idx, c_idx]
            letter = letters[r_idx * n_cols + c_idx]
            parameter = dict(scaled_parameter)
            parameter["kappa"] = float(kappa)
            summary_path = _ensure_fixpoint_summary(args, parameter, focus_counts)
            summary = _load_summary(summary_path)
            meta_param = summary.get("metadata", {}).get("analysis_parameter", {})
            if meta_param.get("connection_type"):
                connection_type = meta_param.get("connection_type")
            plot_panel(
                ax,
                summary=summary,
                marker_focus=args.marker_focus_count,
                line_focus_counts=args.line_focus_counts or [],
                config=config,
                letter=letter,
                color_map=color_map,
            )
            if r_idx == n_rows - 1:
                ax.set_xlabel(r"$R_{E+}$", labelpad=2)
            if c_idx == 0:
                ax.set_ylabel(r"$v_{\mathrm{out}}$", labelpad=2)
                ax.yaxis.set_label_coords(-0.07, 0.5)
                row_label = ax.text(
                    -0.08,
                    0.5,
                    rf"$\bar{{p}} = {actual_avg:.2f}$",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=config.label_size,
                    clip_on=False,
                )
                row_label.set_in_layout(False)
            if r_idx == 0:
                ax.set_title(rf"$\kappa = {kappa:.2f}$")
            ticks = [tick for tick in ax.get_yticks() if abs(tick - 0.5) > 1e-6]
            ax.set_yticks(ticks)
            if args.x_min is not None or args.x_max is not None:
                ax.set_xlim(left=args.x_min, right=args.x_max)
            ax.set_ylim(bottom=args.y_min, top=args.y_max)
    fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.98])
    fig.savefig(args.output, dpi=600)
    if args.write_pdf:
        pdf_path = os.path.splitext(args.output)[0] + ".pdf"
        fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Stored fixpoint grid figure at {args.output}")


if __name__ == "__main__":
    main()
