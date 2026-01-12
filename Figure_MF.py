from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from MeanField.ei_cluster_network import EIClusterNetwork
from plotting import FontCfg, style_axes, style_legend
from ei_pipeline import (
    _filter_fixpoint_candidates,
    _key_to_r_eplus,
    _taggable_configuration,
    resolve_r_eplus,
    run_analysis,
    run_simulation,
)
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg, write_yaml_config

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
COLORBAR_WIDTH_RATIO = 0.04
COLORBAR_HEIGHT_FRACTION = 0.6
LISTED_CATEGORICAL_LIMIT = 32
PANEL_LABEL_COORDS = (-0.12, 1.02)
PANEL_LABEL_ALIGN = ("right", "bottom")
PANEL_LABEL_ABOVE_COORDS = (0.0, 1.02)
PANEL_LABEL_ABOVE_ALIGN = ("center", "bottom")

PROB_KEYS = ("p0_ee", "p0_ei", "p0_ie", "p0_ii")
MARKER_STRIDE = 1
BRANCH_GROUP_TOL = 1e-6
BRANCH_MAX_JUMP = 0.3
BRANCH_MAX_MISSING_STEPS = 2
BIF_COLUMN_TOL = 1e-6
MIN_PROBABILITY = 1e-6
BIF_MARKERS = ("o", "s", "D", "^", "v", "<", ">", "P", "X")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixpoint grid augmented with a bifurcation-curve row, combining "
            "the plot_fixpoint_grid and plot_bifurcation_curves outputs."
        )
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
        "--line-colormap",
        type=str,
        default=None,
        help=(
            "Matplotlib colormap name for line focus colors; use categorical "
            "maps like 'tab10' or continuous maps like 'viridis' (default: legacy palette)."
        ),
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
    parser.add_argument(
        "--bif-focus-count",
        type=int,
        help="Override focus count used for bifurcation searches (default: config focus_count).",
    )
    parser.add_argument(
        "--bif-rj",
        type=float,
        action="append",
        help="Explicit R_j value used for bifurcation curves (may repeat). Defaults to config value.",
    )
    parser.add_argument(
        "--bif-rj-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Optional inclusive range for R_j values used in the bifurcation row.",
    )
    parser.add_argument(
        "--bif-prob-scale",
        type=float,
        action="append",
        help="Explicit probability scaling factors for bifurcation curves (may repeat).",
    )
    parser.add_argument(
        "--bif-prob-scale-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Inclusive range of probability scaling factors (default: 0.1 1.0 0.1).",
    )
    parser.add_argument(
        "--bif-avg-connectivity",
        type=float,
        action="append",
        help="Explicit average connectivity values (0-1) targeted in the bifurcation row.",
    )
    parser.add_argument(
        "--bif-avg-connectivity-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Inclusive range of average connectivity values explored in the bifurcation row.",
    )
    parser.add_argument(
        "--bif-r-eplus-min",
        type=float,
        default=1.0,
        help="Lower bracket for the bifurcation search (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-r-eplus-max",
        type=float,
        default=20.0,
        help="Upper bracket for the bifurcation search (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-bisection-tol",
        type=float,
        default=0.1,
        help="Minimum R_Eplus step resolved by the bifurcation bisection (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-max-iterations",
        type=int,
        default=24,
        help="Maximum bisection iterations per bifurcation curve (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-filter-threshold",
        type=float,
        default=1e-3,
        help="Merge threshold for nearby fixpoints during bifurcation filtering (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-fixpoint-threshold",
        type=int,
        default=3,
        help="Number of fixpoints required to count as a bifurcation (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-cache-root",
        type=str,
        default=os.path.join("data", "bifurcation_curves"),
        help="Cache directory for bifurcation sweeps (default: %(default)s).",
    )
    parser.add_argument(
        "--bif-overwrite-cache",
        action="store_true",
        help="Regenerate bifurcation entries even if cached data exist.",
    )
    parser.add_argument(
        "--bif-jobs",
        type=int,
        help="Worker processes used for bifurcation searches (default: CPU count).",
    )
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


def _cycle_palette(palette: Sequence[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if not palette:
        raise ValueError("Cannot cycle an empty palette.")
    repeats = (count + len(palette) - 1) // len(palette)
    return list(palette * repeats)[:count]


def _sample_cmap_colors(colormap: str, count: int) -> List[str]:
    if count <= 0:
        return []
    try:
        cmap = plt.get_cmap(colormap)
    except ValueError as exc:
        raise SystemExit(f"Unknown matplotlib colormap '{colormap}'.") from exc
    categorical_colors = getattr(cmap, "colors", None)
    use_categorical = (
        isinstance(cmap, mcolors.ListedColormap)
        and categorical_colors is not None
        and len(categorical_colors) <= LISTED_CATEGORICAL_LIMIT
    )
    if use_categorical:
        base_colors = list(categorical_colors)
        repeats = (count + len(base_colors) - 1) // len(base_colors)
        selected = (base_colors * repeats)[:count]
    else:
        if count == 1:
            positions = [0.5]
        else:
            positions = np.linspace(0.0, 1.0, count)
        selected = [cmap(float(pos)) for pos in positions]
    return [mcolors.to_hex(color) for color in selected]


def _prepare_line_color_map(
    focus_counts: Sequence[int],
    *,
    colormap: str | None,
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    mapping: Dict[int, str] = {}
    entries: List[Tuple[int, str]] = []
    ordered_counts = sorted(set(int(fc) for fc in focus_counts))
    if not ordered_counts:
        return mapping, entries
    if colormap:
        colors = _sample_cmap_colors(colormap, len(ordered_counts))
    else:
        colors = _cycle_palette(LINE_COLORS, len(ordered_counts))
    for focus_count, color in zip(ordered_counts, colors):
        mapping[int(focus_count)] = color
        entries.append((int(focus_count), color))
    return mapping, entries


def _focus_count_boundaries(focus_counts: Sequence[int]) -> List[float]:
    if not focus_counts:
        return []
    ordered = sorted(focus_counts)
    if len(ordered) == 1:
        fc = float(ordered[0])
        return [fc - 0.5, fc + 0.5]
    boundaries = [float(ordered[0]) - 0.5]
    for prev_val, next_val in zip(ordered[:-1], ordered[1:]):
        boundaries.append((float(prev_val) + float(next_val)) / 2.0)
    boundaries.append(float(ordered[-1]) + 0.5)
    return boundaries


def _draw_focus_count_colorbar(
    fig: plt.Figure,
    axis: plt.Axes,
    entries: Sequence[Tuple[int, str]],
    font_cfg: FontCfg,
) -> None:
    if not entries:
        axis.set_axis_off()
        return
    axis.set_axis_off()
    target_axis = axis
    if 0.0 < COLORBAR_HEIGHT_FRACTION < 1.0:
        inset_height = COLORBAR_HEIGHT_FRACTION
        inset_y = (1.0 - inset_height) / 2.0
        target_axis = axis.inset_axes([0.0, inset_y, 1.0, inset_height])
    focus_counts = [fc for fc, _ in entries]
    colors = [color for _, color in entries]
    cmap = mcolors.ListedColormap(colors)
    boundaries = _focus_count_boundaries(focus_counts)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        cax=target_axis,
        ticks=focus_counts,
        boundaries=boundaries,
    )
    colorbar.ax.tick_params(labelsize=font_cfg.tick)
    colorbar.ax.set_ylabel(
        "# active clusters",
        fontsize=font_cfg.label,
        rotation=-90,
        va="bottom",
        labelpad=10,
    )


def _cache_key(kappa: float, r_j: float, prob_scale: float, *, decimals: int = 9) -> Tuple[float, float, float]:
    return (
        round(float(kappa), decimals),
        round(float(r_j), decimals),
        round(float(prob_scale), decimals),
    )


def _normalize_float_list(values: Iterable[float], *, decimals: int = 10) -> List[float]:
    rounded = [float(round(float(v), decimals)) for v in values]
    seen = {}
    for val in rounded:
        seen.setdefault(val, True)
    return sorted(seen.keys())


def _resolve_scale_values(
    explicit: Optional[Sequence[float]],
    range_values: Optional[Sequence[float]],
    *,
    default_range: Tuple[float, float, float] = (0.1, 1.0, 0.1),
) -> List[float]:
    collected: List[float] = []
    if explicit:
        collected.extend(float(v) for v in explicit)
    source_range = range_values if range_values is not None else default_range
    if source_range and len(source_range) == 3:
        start, end, step = [float(v) for v in source_range]
        if step == 0:
            raise ValueError("Probability scaling range step must be non-zero.")
        direction = 1 if step > 0 else -1
        compare = (lambda a, b: a <= b) if direction > 0 else (lambda a, b: a >= b)
        cursor = start
        while compare(cursor, end + (1e-12 * direction)):
            collected.append(cursor)
            cursor += step
    if not collected:
        start, end, step = default_range
        direction = 1 if step > 0 else -1
        compare = (lambda a, b: a <= b) if direction > 0 else (lambda a, b: a >= b)
        cursor = start
        while compare(cursor, end + (1e-12 * direction)):
            collected.append(cursor)
            cursor += step
    return _normalize_float_list(collected)


def _apply_probability_scale(parameter: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    scaled = dict(parameter)
    factor = float(scale)
    for key in PROB_KEYS:
        value = scaled.get(key)
        if value is None:
            continue
        try:
            updated = float(value) * factor
        except (TypeError, ValueError):
            raise ValueError(f"Probability '{key}' must be float-like, got {value!r}.")
        if updated <= 0.0:
            updated = MIN_PROBABILITY
        scaled[key] = min(1.0, updated)
    return scaled


def _resolve_value_list(
    explicit: Optional[Sequence[float]],
    range_values: Optional[Sequence[float]],
    fallback: float,
) -> List[float]:
    collected: List[float] = []
    if explicit:
        collected.extend(float(v) for v in explicit)
    if range_values and len(range_values) == 3:
        start, end, step = [float(v) for v in range_values]
        if step == 0:
            raise ValueError("Range step must be non-zero.")
        direction = 1 if step > 0 else -1
        compare = (lambda a, b: a <= b) if direction > 0 else (lambda a, b: a >= b)
        cursor = start
        while compare(cursor, end + (1e-12 * direction)):
            collected.append(cursor)
            cursor += step
    if not collected:
        return [float(fallback)]
    return _normalize_float_list(collected)


def _resolve_focus_count(parameter: Mapping[str, Any], override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    existing = parameter.get("focus_count")
    if existing is not None:
        try:
            return max(1, int(existing))
        except (TypeError, ValueError):
            pass
    candidates = parameter.get("focus_counts")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
        for entry in candidates:
            try:
                return max(1, int(entry))
            except (TypeError, ValueError):
                continue
    try:
        return max(1, int(candidates))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


@dataclass
class EvaluationResult:
    value: float
    count: int
    status: str
    message: Optional[str]


def _summarize_fixpoints(fixpoints: Mapping[float, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for point, entry in sorted(fixpoints.items(), key=lambda item: float(item[0])):
        summary.append(
            {
                "value": float(point),
                "stability": entry.get("stability"),
                "residual_norm": float(entry.get("residual_norm", math.nan)),
                "included": bool(entry.get("included", True)),
            }
        )
    return summary


def evaluate_fixpoints(
    parameter: Dict[str, Any],
    sweep_kwargs: Dict[str, Any],
    filter_threshold: float,
    required_fixpoints: int,
) -> Dict[str, Any]:
    current_parameter = dict(parameter)
    result = EIClusterNetwork.generate_erf_curve(current_parameter, **sweep_kwargs)
    info: Dict[str, Any] = {
        "value": float(current_parameter["R_Eplus"]),
        "status": "ok",
        "count": 0,
        "details": {},
        "completed": bool(result.completed),
    }
    if not result.completed or not result.x_data:
        info["status"] = "erf_incomplete"
        info["message"] = "ERF sweep aborted or empty."
        return info
    sweep_entry = (result.x_data, result.y_data, result.solves, current_parameter)
    try:
        fixpoints = EIClusterNetwork.compute_fixpoints(
            sweep_entry,
            kappa=current_parameter.get("kappa"),
            connection_type=current_parameter.get("connection_type"),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        info["status"] = "fixpoint_error"
        info["message"] = str(exc)
        return info
    filtered, excluded = _filter_fixpoint_candidates(
        fixpoints,
        threshold=float(filter_threshold),
        max_fixpoints=3,
    )
    info["count"] = len(filtered)
    info["details"] = {
        "kept": _summarize_fixpoints(filtered),
        "excluded": _summarize_fixpoints({point: entry for point, entry, _ in excluded}) if excluded else [],
    }
    info["bifurcated"] = info["count"] >= required_fixpoints
    return info


def _bisection_search(
    base_parameter: Dict[str, Any],
    kappa: float,
    r_j: float,
    prob_scale: float,
    sweep_kwargs: Dict[str, Any],
    search: Dict[str, Any],
    filter_threshold: float,
    required_fixpoints: int,
) -> Dict[str, Any]:
    parameter = _apply_probability_scale(base_parameter, prob_scale)
    parameter["kappa"] = float(kappa)
    parameter["R_j"] = float(r_j)
    try:
        mean_conn = _mean_connectivity(parameter)
    except ValueError:
        mean_conn = math.nan
    r_min = float(search["min"])
    r_max = float(search["max"])
    tol = float(search["tol"])
    max_iter = int(search["max_iter"])
    cache: Dict[float, Dict[str, Any]] = {}
    history: List[EvaluationResult] = []

    print(
        f"Searching bifurcation for kappa={kappa:.4f}, R_j={r_j:.4f}, "
        f"scale={prob_scale:.3f}, mean_conn={mean_conn:.4f}"
    )

    def evaluate(value: float) -> Dict[str, Any]:
        key = float(round(float(value), 10))
        if key in cache:
            return cache[key]
        parameter["R_Eplus"] = float(value)
        info = evaluate_fixpoints(parameter, sweep_kwargs, filter_threshold, required_fixpoints)
        cache[key] = info
        history.append(
            EvaluationResult(
                value=float(value),
                count=info.get("count", 0),
                status=info["status"],
                message=info.get("message"),
            )
        )
        return info

    low_eval = evaluate(r_min)
    if low_eval["status"] != "ok":
        return {
            "kappa": float(kappa),
            "R_j": float(r_j),
            "prob_scale": float(prob_scale),
            "mean_connectivity": float(mean_conn),
            "status": f"lower_bound_failed:{low_eval['status']}",
            "bifurcation": None,
            "evaluations": [vars(entry) for entry in history],
        }
    if low_eval["count"] >= required_fixpoints:
        return {
            "kappa": float(kappa),
            "R_j": float(r_j),
            "prob_scale": float(prob_scale),
            "mean_connectivity": float(mean_conn),
            "status": "lower_bound_multistable",
            "bifurcation": r_min,
            "evaluations": [vars(entry) for entry in history],
        }
    high_eval = evaluate(r_max)
    if high_eval["status"] != "ok" or high_eval["count"] < required_fixpoints:
        return {
            "kappa": float(kappa),
            "R_j": float(r_j),
            "prob_scale": float(prob_scale),
            "mean_connectivity": float(mean_conn),
            "status": "upper_bound_missing_transition",
            "bifurcation": None,
            "evaluations": [vars(entry) for entry in history],
        }
    low = r_min
    high = r_max
    iterations = 0
    while (high - low) > tol and iterations < max_iter:
        mid = (low + high) / 2.0
        mid_eval = evaluate(mid)
        if mid_eval["status"] != "ok":
            return {
                "kappa": float(kappa),
                "R_j": float(r_j),
                "prob_scale": float(prob_scale),
                "mean_connectivity": float(mean_conn),
                "status": f"midpoint_failed:{mid_eval['status']}",
                "bifurcation": None,
                "evaluations": [vars(entry) for entry in history],
            }
        if mid_eval["count"] >= required_fixpoints:
            high = mid
            high_eval = mid_eval
        else:
            low = mid
            low_eval = mid_eval
        iterations += 1
    critical = high
    return {
        "kappa": float(kappa),
        "R_j": float(r_j),
        "prob_scale": float(prob_scale),
        "mean_connectivity": float(mean_conn),
        "status": "ok",
        "bifurcation": float(critical),
        "iterations": iterations,
        "evaluations": [vars(entry) for entry in history],
    }


def _task_entry(
    args: Tuple[float, float, float, Dict[str, Any], Dict[str, Any], Dict[str, Any], float, int]
) -> Dict[str, Any]:
    kappa, r_j, prob_scale, base_parameter, sweep_kwargs, search, filter_threshold, required_fixpoints = args
    return _bisection_search(
        base_parameter,
        kappa,
        r_j,
        prob_scale,
        sweep_kwargs,
        search,
        filter_threshold,
        required_fixpoints,
    )


def _cache_paths(root: str, connection_type: str, tag: str) -> Tuple[str, str]:
    conn_label = str(connection_type).lower().replace(" ", "_")
    folder = os.path.join(root, conn_label, tag)
    os.makedirs(folder, exist_ok=True)
    return folder, os.path.join(folder, "bifurcation_cache.pkl")


def _load_existing(cache_path: str) -> Dict[Tuple[float, float, float], Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    entries = payload.get("entries", [])
    lookup: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    for entry in entries:
        prob_scale = float(entry.get("prob_scale", 1.0))
        key = _cache_key(entry["kappa"], entry["R_j"], prob_scale)
        lookup[key] = entry
    return lookup


def _save_cache(cache_path: str, metadata: Dict[str, Any], lookup: Dict[Tuple[float, float, float], Dict[str, Any]]) -> None:
    payload = {
        "metadata": metadata,
        "entries": sorted(
            lookup.values(),
            key=lambda item: (
                float(item["kappa"]),
                float(item["R_j"]),
                float(item.get("prob_scale", 1.0)),
            ),
        ),
    }
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle)


CacheKey = Tuple[float, float, float]


def _collect_plot_data(
    entries: Dict[CacheKey, Dict[str, Any]]
) -> Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]]:
    dataset: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]] = {}
    for key, entry in entries.items():
        if entry.get("status") != "ok":
            continue
        bifurcation = entry.get("bifurcation")
        mean_conn = entry.get("mean_connectivity")
        if bifurcation is None or not math.isfinite(float(bifurcation)):
            continue
        if mean_conn is None or not math.isfinite(float(mean_conn)):
            continue
        r_j = float(entry["R_j"])
        kappa = float(entry["kappa"])
        dataset.setdefault(r_j, {}).setdefault(kappa, []).append(
            (float(mean_conn), float(bifurcation), key)
        )
    for r_j, lines in dataset.items():
        for kappa in list(lines.keys()):
            lines[kappa].sort(key=lambda item: item[0])
    return dataset


def _match_column_value(value: float, columns: Sequence[float], tol: float = BIF_COLUMN_TOL) -> float | None:
    for ref in columns:
        if abs(float(value) - float(ref)) <= tol:
            return float(ref)
    return None


def _filter_curves_for_columns(
    curves: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]],
    columns: Sequence[float],
) -> Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]]:
    if not columns:
        return {}
    filtered: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]] = {}
    for r_j, lines in curves.items():
        for kappa, entries in lines.items():
            match = _match_column_value(kappa, columns)
            if match is None:
                continue
            target = filtered.setdefault(float(r_j), {}).setdefault(match, [])
            target.extend(entries)
    for r_j, lines in filtered.items():
        for kappa in list(lines.keys()):
            lines[kappa].sort(key=lambda item: item[0])
    return filtered


def _compute_bifurcation_curves(
    args: argparse.Namespace,
    base_parameter: Dict[str, Any],
    *,
    kappa_values: Sequence[float],
    rj_values: Sequence[float],
    prob_scales: Sequence[float],
    sweep_kwargs: Mapping[str, Any],
    search: Mapping[str, Any],
) -> Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]]:
    parameter = dict(base_parameter)
    parameter["focus_count"] = _resolve_focus_count(parameter, args.bif_focus_count)
    if parameter["focus_count"] == 1:
        parameter["collapse_types"] = True
    filtered = _taggable_configuration(parameter)
    for key in ("R_j", "kappa", "R_Eplus"):
        filtered.pop(key, None)
    filtered.pop("focus_counts", None)
    filtered["focus_count"] = parameter.get("focus_count", 1)
    metadata = {
        "parameter": filtered,
        "kappa_values": [float(val) for val in kappa_values],
        "rj_values": [float(val) for val in rj_values],
        "prob_scales": [float(val) for val in prob_scales],
        "search": {key: float(val) for key, val in search.items()},
        "sweep": dict(sweep_kwargs),
    }
    tag = sim_tag_from_cfg(metadata)
    connection_type = parameter.get("connection_type", "bernoulli")
    cache_root = os.path.abspath(args.bif_cache_root)
    cache_folder, cache_path = _cache_paths(cache_root, str(connection_type), tag)
    params_path = os.path.join(cache_folder, "params.yaml")
    write_yaml_config(metadata, params_path)
    if args.bif_overwrite_cache and os.path.exists(cache_path):
        os.remove(cache_path)
    existing_entries = {} if args.bif_overwrite_cache else _load_existing(cache_path)
    lookup: Dict[CacheKey, Dict[str, Any]] = dict(existing_entries)
    tasks: List[Tuple[float, float, float, Dict[str, Any], Dict[str, Any], Dict[str, Any], float, int]] = []
    base_parameter = dict(parameter)
    for kappa in kappa_values:
        for r_j in rj_values:
            for prob_scale in prob_scales:
                key = _cache_key(kappa, r_j, prob_scale)
                if key in lookup:
                    continue
                task = (
                    float(kappa),
                    float(r_j),
                    float(prob_scale),
                    base_parameter,
                    dict(sweep_kwargs),
                    dict(search),
                    float(args.bif_filter_threshold),
                    int(max(2, args.bif_fixpoint_threshold)),
                )
                tasks.append(task)
    if tasks:
        jobs = args.bif_jobs or mp.cpu_count()
        if jobs <= 1:
            new_entries = [_task_entry(task) for task in tasks]
        else:
            with mp.Pool(processes=jobs) as pool:
                new_entries = pool.map(_task_entry, tasks)
        for entry in new_entries:
            key = _cache_key(entry["kappa"], entry["R_j"], entry.get("prob_scale", 1.0))
            lookup[key] = entry
    else:
        print("All requested bifurcation entries already cached.")
    _save_cache(cache_path, metadata, lookup)
    curves = _collect_plot_data(lookup)
    return _filter_curves_for_columns(curves, kappa_values)


def _build_rj_styles(rj_values: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    ordered = [float(val) for val in sorted({float(v) for v in rj_values})]
    if not ordered:
        return {}
    cmap = plt.get_cmap("tab10", max(len(ordered), 2))
    colors = cmap(np.linspace(0, 1, max(len(ordered), 2)))
    styles: Dict[float, Dict[str, Any]] = {}
    for idx, r_j in enumerate(ordered):
        styles[r_j] = {
            "color": mcolors.to_hex(colors[idx % len(colors)]),
            "marker": BIF_MARKERS[idx % len(BIF_MARKERS)],
            "label": rf"$R_j = {r_j:.2f}$",
        }
    return styles


def _plot_bifurcation_panel(
    ax: plt.Axes,
    *,
    data: Dict[float, List[Tuple[float, float, CacheKey]]],
    rj_order: Sequence[float],
    styles: Mapping[float, Mapping[str, Any]],
    letter: str,
    label_coords: Tuple[float, float],
    label_align: Tuple[str, str],
    font_cfg: FontCfg,
    x_limits: Tuple[float, float] | None,
    y_limits: Tuple[float, float] | None,
) -> Dict[float, plt.Line2D]:
    handles: Dict[float, plt.Line2D] = {}
    y_values: List[float] = []
    for r_j in rj_order:
        entries = data.get(r_j)
        if not entries:
            continue
        xs = [point[1] for point in entries]
        ys = [point[0] * 100.0 for point in entries]
        y_values.extend(ys)
        style = styles.get(r_j, {})
        line, = ax.plot(
            xs,
            ys,
            color=style.get("color", DEFAULT_LINE_COLOR),
            #marker=style.get("marker", "o"),
            linewidth=1.5,
            markersize=4.0,
            label=style.get("label", rf"$R_j = {r_j:.2f}$"),
        )
        handles[r_j] = line
    if x_limits:
        ax.set_xlim(*x_limits)
    if y_limits:
        ax.set_ylim(*y_limits)
    elif y_values:
        margin = 5.0
        ax.set_ylim(max(0.0, min(y_values) - margin), min(100.0, max(y_values) + margin))
    #ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    panel_label = ax.text(
        label_coords[0],
        label_coords[1],
        f"{letter}",
        transform=ax.transAxes,
        ha=label_align[0],
        va=label_align[1],
        fontweight="bold",
        fontsize=font_cfg.letter,
        clip_on=False,
    )
    panel_label.set_in_layout(False)
    return handles


def plot_bifurcation_row(
    axes: Sequence[plt.Axes],
    *,
    dataset: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]],
    col_order: Sequence[float],
    rj_values: Sequence[float],
    font_cfg: FontCfg,
    letters: Sequence[str],
    start_index: int,
    search_bounds: Tuple[float, float],
) -> None:
    if not axes:
        return
    styles = _build_rj_styles(rj_values)
    ordered_rj = [float(val) for val in sorted({float(v) for v in rj_values})]
    legend_handles: Dict[float, plt.Line2D] = {}
    all_means: List[float] = []
    for lines in dataset.values():
        for entries in lines.values():
            all_means.extend(point[0] * 100.0 for point in entries)
    if all_means:
        y_limits = (max(0.0, min(all_means) - 5.0), min(100.0, max(all_means) + 5.0))
    else:
        y_limits = (0.0, 100.0)
    x_limits = (float(search_bounds[0]), float(search_bounds[1]))
    for idx, (ax, kappa) in enumerate(zip(axes, col_order)):
        letter = letters[start_index + idx]
        column_data: Dict[float, List[Tuple[float, float, CacheKey]]] = {}
        for r_j in ordered_rj:
            entries = dataset.get(r_j, {}).get(float(kappa))
            if entries:
                column_data[r_j] = entries
        label_coords = PANEL_LABEL_ABOVE_COORDS
        label_align = PANEL_LABEL_ABOVE_ALIGN
        handles = _plot_bifurcation_panel(
            ax,
            data=column_data,
            rj_order=ordered_rj,
            styles=styles,
            letter=letter,
            label_coords=label_coords,
            label_align=label_align,
            font_cfg=font_cfg,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        for r_j, handle in handles.items():
            legend_handles.setdefault(r_j, handle)
        if idx == 0:
            ax.set_ylabel(r"$\boldsymbol{\bar{p}}$ [%]")
        ax.set_xlabel(r"$R_{E+}$")
        style_axes(ax, font_cfg)
    if legend_handles:
        last_ax = axes[-1]
        legend_entries = []
        labels = []
        for r_j in ordered_rj:
            handle = legend_handles.get(r_j)
            if handle is None:
                continue
            legend_entries.append(handle)
            labels.append(styles.get(r_j, {}).get("label", rf"$R_j = {r_j:.2f}$"))
        if legend_entries:
            last_ax.legend(legend_entries, labels, loc="upper right", frameon=False)
            style_legend(last_ax, font_cfg)


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
        f"{letter}",
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
    base_avg_conn = _mean_connectivity(base_parameter)
    if base_avg_conn is None or base_avg_conn <= 0:
        raise SystemExit("Base configuration has invalid average connectivity; cannot run Figure_MF.")
    row_order = [float(val) for val in args.rows]
    col_order = [float(val) for val in args.columns]
    if not col_order:
        raise SystemExit("At least one kappa column must be specified via --columns.")
    focus_counts = {args.marker_focus_count}
    if args.line_focus_counts:
        focus_counts.update(args.line_focus_counts)
    if args.extra_focus_counts:
        focus_counts.update(args.extra_focus_counts)
    focus_counts = sorted({max(1, int(fc)) for fc in focus_counts})
    n_rows = len(row_order)
    n_cols = len(col_order)
    total_rows = n_rows + 1
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    margin_ratio = 0.1
    height_ratios = [1.0] * n_rows + [0.9]
    grid = fig.add_gridspec(
        total_rows,
        n_cols + 2,
        width_ratios=[margin_ratio] + [1.0] * n_cols + [COLORBAR_WIDTH_RATIO],
        height_ratios=height_ratios,
        wspace=0.01,
        hspace=0.04,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    shared_ax = None
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            ax = fig.add_subplot(grid[r_idx, c_idx + 1], sharex=shared_ax, sharey=shared_ax)
            axes[r_idx, c_idx] = ax
            if shared_ax is None:
                shared_ax = ax
    bif_axes: List[plt.Axes] = []
    bif_row_idx = n_rows
    shared_bif_ax = None
    for c_idx in range(n_cols):
        ax = fig.add_subplot(
            grid[bif_row_idx, c_idx + 1],
            sharex=shared_ax,
            sharey=shared_bif_ax,
        )
        bif_axes.append(ax)
        if shared_bif_ax is None:
            shared_bif_ax = ax
    all_focus_counts = list(args.line_focus_counts or [])
    color_map, colorbar_entries = _prepare_line_color_map(
        all_focus_counts,
        colormap=args.line_colormap,
    )
    total_panels = total_rows * n_cols
    letters = [chr(ord("a") + idx) for idx in range(total_panels)]
    base_rj = float(base_parameter.get("R_j", 0.0))
    rj_values = _resolve_value_list(args.bif_rj, args.bif_rj_range, base_rj)
    if not rj_values:
        rj_values = [base_rj]
    prob_scales = _resolve_scale_values(args.bif_prob_scale, args.bif_prob_scale_range)
    if args.bif_avg_connectivity or args.bif_avg_connectivity_range:
        avg_targets = _resolve_value_list(
            args.bif_avg_connectivity,
            args.bif_avg_connectivity_range,
            base_avg_conn,
        )
        converted = [
            max(float(value), MIN_PROBABILITY) / base_avg_conn
            for value in avg_targets
            if math.isfinite(float(value)) and float(value) > 0
        ]
        if converted:
            prob_scales = sorted(
                {float(scale) for scale in list(prob_scales) + converted}
            )
    search = {
        "min": float(args.bif_r_eplus_min),
        "max": float(args.bif_r_eplus_max),
        "tol": float(args.bif_bisection_tol),
        "max_iter": int(args.bif_max_iterations),
    }
    if search["max"] <= search["min"]:
        raise SystemExit("Bifurcation search max bound must exceed the min bound.")
    sweep_kwargs = dict(
        start=float(args.v_start),
        end=float(args.v_end),
        step_number=int(args.v_steps),
        retry_step=args.retry_step,
    )
    bif_curves = _compute_bifurcation_curves(
        args,
        dict(base_parameter),
        kappa_values=col_order,
        rj_values=rj_values,
        prob_scales=prob_scales,
        sweep_kwargs=sweep_kwargs,
        search=search,
    )
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
            if c_idx == 0:
                ax.set_ylabel(r"$v_{\mathrm{out}}$")
                ax.yaxis.set_label_coords(-0.025, 0.5)
                percentage = max(0.0, actual_avg) * 100.0
                row_label = ax.text(
                    -0.18,
                    0.5,
                    rf"$\boldsymbol{{\bar{{p}}}} \boldsymbol{{=}} {percentage:.0f}\,\%$",
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
    bif_letter_offset = n_rows * n_cols
    if bif_axes:
        plot_bifurcation_row(
            bif_axes,
            dataset=bif_curves,
            col_order=col_order,
            rj_values=rj_values,
            font_cfg=font_cfg,
            letters=letters,
            start_index=bif_letter_offset,
            search_bounds=(search["min"], search["max"]),
        )
        ticks = list(bif_axes[0].get_yticks())
        if len(ticks) > 1:
            trimmed = ticks[:-1]
            for ax in bif_axes:
                ax.set_yticks(trimmed)
    colorbar_rows = slice(0, max(1, n_rows))
    colorbar_ax = fig.add_subplot(grid[colorbar_rows, -1])
    _draw_focus_count_colorbar(fig, colorbar_ax, colorbar_entries, font_cfg)
    save_kwargs = {"dpi": 600}
    fig.savefig(args.output, **save_kwargs)
    if args.write_pdf:
        pdf_path = os.path.splitext(args.output)[0] + ".pdf"
        fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    print(f"Stored Figure_MF figure at {args.output}")


if __name__ == "__main__":
    main()
