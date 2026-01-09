from __future__ import annotations

import argparse
import os
import pickle
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plotting import FontCfg, style_axes
from ei_pipeline import (
    _key_to_r_eplus,
    _taggable_configuration,
    resolve_r_eplus,
    run_analysis,
    run_simulation,
)
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

def _cmyk_to_rgb_hex(c: float, m: float, y: float, k: float) -> str:
    r = 1.0 - min(1.0, c + k)
    g = 1.0 - min(1.0, m + k)
    b = 1.0 - min(1.0, y + k)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


FOCUS_STABLE_COLOR = _cmyk_to_rgb_hex(0.0, 0.0, 0.0, 1.0)
FOCUS_UNSTABLE_COLOR = _cmyk_to_rgb_hex(0.0, 0.0, 0.0, 1.0)
LINE_COLORS = (
    _cmyk_to_rgb_hex(0.8, 0.1, 0.0, 0.1),
    _cmyk_to_rgb_hex(0.0, 0.6, 0.2, 0.1),
    _cmyk_to_rgb_hex(0.1, 0.2, 0.8, 0.1),
    _cmyk_to_rgb_hex(0.0, 0.4, 0.8, 0.2),
    _cmyk_to_rgb_hex(0.6, 0.0, 0.1, 0.2),
)
DEFAULT_LINE_COLOR = LINE_COLORS[0]
PANEL_LABEL_COORDS = (-0.12, 1.02)
PANEL_LABEL_ALIGN = ("right", "bottom")
PANEL_LABEL_ABOVE_COORDS = (0.0, 1.02)
PANEL_LABEL_ABOVE_ALIGN = ("center", "bottom")

PROB_KEYS = ("p0_ee", "p0_ei", "p0_ie", "p0_ii")
MARKER_STRIDE = 1
BRANCH_GROUP_TOL = 1e-6
BRANCH_MAX_JUMP = 0.3
BRANCH_MAX_MISSING_STEPS = 2


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
        "--row-parameter",
        action="append",
        default=[],
        metavar="ROW:key=value",
        help=(
            "Row-specific parameter override (0-based row index). "
            "Example: --row-parameter 0:p0_ee=0.15 --row-parameter 1:connection_type=poisson"
        ),
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
        default=os.path.join("Figures", "Figure2.png"),
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


def _coerce_override_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_row_parameter_overrides(entries: Sequence[str] | None) -> Dict[int, Dict[str, Any]]:
    overrides: Dict[int, Dict[str, Any]] = {}
    if not entries:
        return overrides
    for entry in entries:
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Missing ':' separator in row override '{entry}'.")
        row_part, assignment = entry.split(":", 1)
        if "=" not in assignment:
            raise ValueError(f"Missing '=' in row override '{entry}'.")
        key_part, value_part = assignment.split("=", 1)
        row_idx = int(row_part.strip())
        key = key_part.strip()
        if not key:
            raise ValueError(f"Missing parameter key in row override '{entry}'.")
        overrides.setdefault(row_idx, {})[key] = _coerce_override_value(value_part)
    return overrides


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


def _match_branch_values(previous: Sequence[float], current: Sequence[float]) -> Dict[int, int]:
    """
    Return an assignment mapping indices of `previous` values to indices of `current`
    values, minimizing the total absolute jump and matching as many pairs as possible.
    """
    prev_vals = tuple(float(val) for val in previous)
    curr_vals = tuple(float(val) for val in current)
    n_prev = len(prev_vals)
    n_curr = len(curr_vals)
    target = min(n_prev, n_curr)
    if target == 0:
        return {}

    @lru_cache(None)
    def helper(idx: int, mask: int, matches_left: int) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
        if matches_left == 0:
            return 0.0, ()
        if idx >= n_prev or n_prev - idx < matches_left:
            return float("inf"), ()
        best_cost, best_pairs = helper(idx + 1, mask, matches_left)
        for curr_idx in range(n_curr):
            if mask & (1 << curr_idx):
                continue
            next_cost, next_pairs = helper(idx + 1, mask | (1 << curr_idx), matches_left - 1)
            total_cost = next_cost + abs(curr_vals[curr_idx] - prev_vals[idx])
            if total_cost < best_cost:
                best_cost = total_cost
                best_pairs = next_pairs + ((idx, curr_idx),)
        return best_cost, best_pairs

    _, pairs = helper(0, 0, target)
    return {prev_idx: curr_idx for prev_idx, curr_idx in pairs}


def _segment_branches(
    points: Sequence[Tuple[float, float]],
    *,
    tol: float = BRANCH_GROUP_TOL,
    max_jump: float = BRANCH_MAX_JUMP,
) -> List[List[Tuple[float, float]]]:
    if not points:
        return []
    sorted_points = sorted((float(x), float(y)) for x, y in points)
    grouped: List[Tuple[float, List[float]]] = []
    for x, y in sorted_points:
        if not grouped or abs(x - grouped[-1][0]) > tol:
            grouped.append((x, [y]))
        else:
            grouped[-1][1].append(y)

    unique_xs = [entry[0] for entry in grouped]
    typical_step = None
    if len(unique_xs) >= 2:
        diffs = [unique_xs[i + 1] - unique_xs[i] for i in range(len(unique_xs) - 1)]
        diffs = [diff for diff in diffs if diff > tol]
        if diffs:
            typical_step = float(np.median(diffs))
    gap_threshold = None
    if typical_step and typical_step > 0:
        gap_threshold = typical_step * (BRANCH_MAX_MISSING_STEPS + 1)

    branches: List[Dict[str, object]] = []

    def _start_branch(x_val: float, y_val: float) -> None:
        branches.append(
            {
                "points": [(x_val, y_val)],
                "last_x": x_val,
                "last_y": y_val,
                "active": True,
            }
        )

    first_x, first_vals = grouped[0]
    for y in sorted(first_vals):
        _start_branch(first_x, y)

    for x, y_values in grouped[1:]:
        ordered_y = sorted(y_values)
        active_indices = [idx for idx, branch in enumerate(branches) if branch["active"]]
        if not active_indices:
            for y in ordered_y:
                _start_branch(x, y)
            continue
        previous_vals = [branches[idx]["last_y"] for idx in active_indices]
        assignments = _match_branch_values(previous_vals, ordered_y)
        assigned_curr = set()
        for local_idx, branch_idx in enumerate(active_indices):
            branch = branches[branch_idx]
            curr_idx = assignments.get(local_idx)
            if curr_idx is None:
                branch["active"] = False
                continue
            x_gap = x - branch["last_x"]
            if gap_threshold is not None and x_gap > gap_threshold + tol:
                branch["active"] = False
                continue
            y_val = ordered_y[curr_idx]
            if abs(y_val - branch["last_y"]) > max_jump:
                branch["active"] = False
                continue
            branch["points"].append((x, y_val))
            branch["last_x"] = x
            branch["last_y"] = y_val
            assigned_curr.add(curr_idx)
        for idx, y_val in enumerate(ordered_y):
            if idx not in assigned_curr:
                _start_branch(x, y_val)
    return [branch["points"] for branch in branches if branch["points"]]


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
    letter: str,
    color_map: Dict[int, str],
    label_coords: Tuple[float, float],
    label_align: Tuple[str, str],
    font_cfg: FontCfg,
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
            color=FOCUS_STABLE_COLOR,
        )
    if unstable_marker:
        xs, ys = zip(*unstable_marker)
        ax.scatter(
            xs,
            ys,
            s=35,
            facecolors="none",
            edgecolors=FOCUS_UNSTABLE_COLOR,
        )
    for focus_count in line_focus_counts or []:
        points = _collect_fixpoint_points(fixpoints, focus_count)
        stable_points = [(x, y) for x, y, status in points if status == "stable"]
        if not stable_points:
            continue
        color = color_map.get(focus_count, DEFAULT_LINE_COLOR)
        segments = _segment_branches(stable_points)
        if not segments:
            xs, ys = zip(*stable_points)
            ax.plot(xs, ys, color=color, linewidth=1.5)
            continue
        for segment in segments:
            xs, ys = zip(*segment)
            ax.plot(xs, ys, color=color, linewidth=1.5)
    panel_label = ax.text(
        label_coords[0],
        label_coords[1],
        f"{letter})",
        transform=ax.transAxes,
        ha=label_align[0],
        va=label_align[1],
        fontweight="bold",
        fontsize=font_cfg.letter,
        clip_on=False,
    )
    panel_label.set_in_layout(False)


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    try:
        row_parameter_overrides = _parse_row_parameter_overrides(args.row_parameter)
    except ValueError as exc:
        raise SystemExit(f"Invalid --row-parameter value: {exc}") from exc
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
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
    margin_ratio = 0.1
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[margin_ratio] + [1.0] * n_cols,
        wspace=0.05,
        hspace=0.05,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    shared_ax = None
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            ax = fig.add_subplot(grid[r_idx, c_idx + 1], sharex=shared_ax, sharey=shared_ax)
            axes[r_idx, c_idx] = ax
            if shared_ax is None:
                shared_ax = ax
    all_focus_counts = list(args.line_focus_counts or [])
    color_map = _prepare_line_color_map(all_focus_counts, LINE_COLORS)
    letters = [chr(ord("a") + idx) for idx in range(n_rows * n_cols)]
    for r_idx, avg_conn in enumerate(row_order):
        scaled_parameter = _scale_probabilities(base_parameter, avg_conn)
        row_parameter = dict(scaled_parameter)
        overrides = row_parameter_overrides.get(r_idx)
        if overrides:
            row_parameter.update(overrides)
        row_avg_value = _mean_connectivity(row_parameter)
        actual_avg = row_avg_value if row_avg_value is not None else avg_conn
        for c_idx, kappa in enumerate(col_order):
            ax = axes[r_idx, c_idx]
            letter = letters[r_idx * n_cols + c_idx]
            parameter = dict(row_parameter)
            parameter["kappa"] = float(kappa)
            summary_path = _ensure_fixpoint_summary(args, parameter, focus_counts)
            summary = _load_summary(summary_path)
            meta_param = summary.get("metadata", {}).get("analysis_parameter", {})
            if meta_param.get("connection_type"):
                connection_type = meta_param.get("connection_type")
            if c_idx == 0:
                label_coords = PANEL_LABEL_COORDS
                label_align = PANEL_LABEL_ALIGN
            else:
                label_coords = PANEL_LABEL_ABOVE_COORDS
                label_align = PANEL_LABEL_ABOVE_ALIGN
            plot_panel(
                ax,
                summary=summary,
                marker_focus=args.marker_focus_count,
                line_focus_counts=args.line_focus_counts or [],
                letter=letter,
                color_map=color_map,
                label_coords=label_coords,
                label_align=label_align,
                font_cfg=font_cfg,
            )
            if r_idx == n_rows - 1:
                ax.set_xlabel(r"$R_{E+}$", labelpad=2)
            if c_idx == 0:
                ax.set_ylabel(r"$v_{\mathrm{out}}$")
                ax.yaxis.set_label_coords(-0.025, 0.5)
                row_label = ax.text(
                    -0.1,
                    0.5,
                    rf"$\boldsymbol{{\bar{{p}}}} \boldsymbol{{=}} {actual_avg:.2f}$",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=font_cfg.label,
                    clip_on=False,
                )
                row_label.set_in_layout(False)
            if r_idx == 0:
                ax.set_title(rf"$\kappa = {kappa:.2f}$", fontsize=font_cfg.title)
            ticks = [
                tick
                for tick in ax.get_yticks()
                if all(abs(tick - val) > 1e-6 for val in (0.25, 0.5, 0.75))
            ]
            ax.set_yticks(ticks)
            if args.x_min is not None or args.x_max is not None:
                ax.set_xlim(left=args.x_min, right=args.x_max)
            ax.set_ylim(bottom=args.y_min, top=args.y_max)
            ax.set_yticks([np.round(args.y_min), np.round(args.y_max)])
            style_axes(ax, font_cfg)
            for ax in axes[:,0]:
                ax.set_ylabel(r"$v_{\mathrm{out}}$", labelpad=+20)
    save_kwargs = {"dpi": 600}
    fig.savefig(args.output, **save_kwargs)
    if args.write_pdf:
        pdf_path = os.path.splitext(args.output)[0] + ".pdf"
        fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    print(f"Stored fixpoint grid figure at {args.output}")


if __name__ == "__main__":
    main()
