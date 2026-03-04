from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
import numpy as np

from MeanField.ei_cluster_network import EIClusterNetwork
from ei_pipeline import _filter_fixpoint_candidates, _taggable_configuration
from plotting import FontCfg, style_axes, style_colorbar, style_legend
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg, write_yaml_config


PROB_KEYS = ("p0_ee", "p0_ei", "p0_ie", "p0_ii")
MIN_PROBABILITY = 1e-6


def _cache_key(kappa: float, r_j: float, prob_scale: float, *, decimals: int = 9) -> Tuple[float, float, float]:
    return (
        round(float(kappa), decimals),
        round(float(r_j), decimals),
        round(float(prob_scale), decimals),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate ERF bifurcation points (transition from one to three fixpoints) "
            "for grids of kappa and R_j values."
        )
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--focus-count",
        type=int,
        help="Number of focus populations (default: config value or 1).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        action="append",
        help="Explicit kappa value (may be repeated). Defaults to the config value.",
    )
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Optional inclusive range for kappa values.",
    )
    parser.add_argument(
        "--rj",
        type=float,
        action="append",
        help="Explicit R_j value (may be repeated). Defaults to the config value.",
    )
    parser.add_argument(
        "--rj-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Optional inclusive range for R_j values.",
    )
    parser.add_argument(
        "--prob-scale",
        type=float,
        action="append",
        help="Explicit scaling factor for p0_ee/p0_ei/p0_ie/p0_ii (may repeat).",
    )
    parser.add_argument(
        "--prob-scale-range",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="Inclusive range for probability scaling factors (default: 0.1 1 0.1).",
    )
    parser.add_argument(
        "--r-eplus-min",
        type=float,
        default=1.0,
        help="Lower bound of the R_Eplus bisection bracket (default: %(default)s).",
    )
    parser.add_argument(
        "--r-eplus-max",
        type=float,
        default=20.0,
        help="Upper bound of the R_Eplus bisection bracket (default: %(default)s).",
    )
    parser.add_argument(
        "--bisection-tol",
        type=float,
        default=0.1,
        help="Tolerance (minimum step size) for the bisection search (default: %(default)s).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=24,
        help="Maximum number of bisection iterations per curve (default: %(default)s).",
    )
    parser.add_argument("--v-start", type=float, default=0.0, help="ERF sweep start value.")
    parser.add_argument("--v-end", type=float, default=1.0, help="ERF sweep end value.")
    parser.add_argument("--v-steps", type=int, default=500, help="Number of ERF samples.")
    parser.add_argument("--retry-step", type=float, default=None, help="Retry increment for solver restarts.")
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=1e-3,
        help="Merge threshold for nearby fixpoints during filtering (default: %(default)s).",
    )
    parser.add_argument(
        "--fixpoint-threshold",
        type=int,
        default=3,
        help="Number of fixpoints required to count as a bifurcation (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Number of worker processes (default: CPU count).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("Figures", "Figure3.png"),
        help="Custom output base path for the generated figure (PNG/PDF will be saved).",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=os.path.join("data", "bifurcation_curves"),
        help="Directory used for cached bifurcation results.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Regenerate all bifurcation entries even if cached data exists.",
    )
    parser.add_argument(
        "--debug-jumps",
        action="store_true",
        help="Detect large R_Eplus jumps and dump ERF/fixpoint diagnostics.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=1.0,
        help="Minimum R_Eplus gap that triggers a debug figure when --debug-jumps is set.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
        help="Maximum number of jump segments to visualize when debugging.",
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default=os.path.join("plots", "bifurcation_debug"),
        help="Folder for ERF/fixpoint debug figures.",
    )
    parser.add_argument(
        "--projection-step",
        type=float,
        default=0.1,
        help="Delta added to the lower-branch R_Eplus when plotting projected ERFs.",
    )
    parser.add_argument(
        "--correction-sweep",
        action="store_true",
        help="After the main sweep, re-run entries with large jumps near the lower branch.",
    )
    parser.add_argument(
        "--correction-window",
        type=float,
        default=1.0,
        help="Maximum R_Eplus window above the lower branch explored during corrections.",
    )
    return parser.parse_args()


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


def _mean_connectivity(parameter: Mapping[str, Any]) -> float:
    try:
        N_E = float(parameter["N_E"])
        N_I = float(parameter["N_I"])
        p_ee = float(parameter.get("p0_ee", 0.0))
        p_ei = float(parameter.get("p0_ei", 0.0))
        p_ie = float(parameter.get("p0_ie", 0.0))
        p_ii = float(parameter.get("p0_ii", 0.0))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Missing or invalid connectivity parameters.") from exc
    denom = (N_E + N_I) ** 2
    if denom <= 0:
        return 0.0
    numerator = (N_E ** 2) * p_ee + (N_E * N_I) * (p_ei + p_ie) + (N_I ** 2) * p_ii
    return numerator / denom


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
        history.append(EvaluationResult(value=float(value), count=info.get("count", 0), status=info["status"], message=info.get("message")))
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


def _task_entry(args: Tuple[float, float, float, Dict[str, Any], Dict[str, Any], Dict[str, Any], float, int]) -> Dict[str, Any]:
    kappa, r_j, prob_scale, base_parameter, sweep_kwargs, search, filter_threshold, required_fixpoints = args
    return _bisection_search(base_parameter, kappa, r_j, prob_scale, sweep_kwargs, search, filter_threshold, required_fixpoints)


def _default_output_path(connection_type: str, tag: str) -> str:
    os.makedirs("Figures", exist_ok=True)
    return os.path.join("Figures", "Figure3.png")


def _cache_paths(root: str, connection_type: str, tag: str) -> Tuple[str, str]:
    conn_label = str(connection_type).lower().replace(" ", "_")
    folder = os.path.join(root, conn_label, tag)
    os.makedirs(folder, exist_ok=True)
    return folder, os.path.join(folder, "bifurcation_cache.pkl")


def _prepare_output_paths(path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(path)
    root = base if ext else path
    png_path = root + ".png"
    pdf_path = root + ".pdf"
    for dest in (png_path, pdf_path):
        folder = os.path.dirname(dest)
        if folder:
            os.makedirs(folder, exist_ok=True)
    return png_path, pdf_path


def _categorical_boundaries(values: Sequence[float]) -> List[float]:
    ordered = sorted({float(v) for v in values})
    if not ordered:
        return []
    if len(ordered) == 1:
        return [ordered[0] - 0.5, ordered[0] + 0.5]
    boundaries = [ordered[0] - (ordered[1] - ordered[0]) / 2.0]
    for prev_val, next_val in zip(ordered[:-1], ordered[1:]):
        boundaries.append((prev_val + next_val) / 2.0)
    boundaries.append(ordered[-1] + (ordered[-1] - ordered[-2]) / 2.0)
    return boundaries


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
            key=lambda item: (float(item["kappa"]), float(item["R_j"]), float(item.get("prob_scale", 1.0))),
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


def _detect_jump_segments(
    dataset: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]],
    *,
    threshold: float,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for r_j, lines in dataset.items():
        for kappa, entries in lines.items():
            ordered = sorted(entries, key=lambda item: item[0])
            for left, right in zip(ordered, ordered[1:]):
                gap = abs(float(right[1]) - float(left[1]))
                if gap >= threshold:
                    issues.append(
                        {
                            "r_j": float(r_j),
                            "kappa": float(kappa),
                            "left": left,
                            "right": right,
                            "gap": gap,
                        }
                    )
    return issues


def generate_plot(
    curves: Dict[float, Dict[float, List[Tuple[float, float, CacheKey]]]],
    *,
    output_path: str,
    connection_type: str,
    tag: str,
) -> None:
    if not curves:
        print("No valid bifurcation points available for plotting.")
        return
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    rj_values = sorted(curves.keys())
    kappa_values = sorted({kappa for lines in curves.values() for kappa in lines})
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
    if kappa_values:
        cmap = plt.cm.get_cmap("tab10", max(len(kappa_values), 2))
        colors = cmap(np.linspace(0, 1, max(len(kappa_values), 2)))
    else:
        colors = []
    kappa_colors = {
        kappa: colors[idx % len(colors)] if colors else "#1f77b4"
        for idx, kappa in enumerate(kappa_values)
    }
    scatter_handles = []
    for r_idx, r_j in enumerate(rj_values):
        marker = markers[r_idx % len(markers)]
        lines = curves[r_j]
        for k_idx, kappa in enumerate(kappa_values):
            data = lines.get(kappa)
            if not data:
                continue
            xs = [item[0] for item in data]
            ys = [item[1] for item in data]
            color = kappa_colors.get(kappa, "#1f77b4")
            ax.plot(ys, xs, marker=marker, color=color, label=None)
    for r_idx, r_j in enumerate(rj_values):
        marker = markers[r_idx % len(markers)]
        handle = plt.Line2D([], [], color="black", marker=marker, linestyle="None", label=f"R_j={r_j:.2f}")
        scatter_handles.append(handle)
    ax.set_ylabel(r"$\boldsymbol{\bar{p}}$")
    ax.set_xlabel(r"$R_{E+}$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if scatter_handles:
        ax.legend(scatter_handles, [handle.get_label() for handle in scatter_handles], loc="best")
        style_legend(ax, font_cfg)
    style_axes(ax, font_cfg)
    if kappa_values:
        ordered_kappa = [float(val) for val in kappa_values]
        cmap = mcolors.ListedColormap([kappa_colors[val] for val in ordered_kappa])
        boundaries = _categorical_boundaries(ordered_kappa)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.025, ticks=ordered_kappa)
        cbar.ax.set_ylabel(r"$\kappa$")
        style_colorbar(cbar, font_cfg)
    png_path, pdf_path = _prepare_output_paths(output_path)
    save_kwargs = {"dpi": 600}
    fig.savefig(png_path, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    print(f"Stored figures at {png_path} and {pdf_path}")


def _prepare_parameter_for_entry(base_parameter: Mapping[str, Any], entry: Mapping[str, Any]) -> Dict[str, Any]:
    parameter = _apply_probability_scale(base_parameter, entry.get("prob_scale", 1.0))
    parameter["kappa"] = float(entry.get("kappa", parameter.get("kappa", 0.0)))
    parameter["R_j"] = float(entry.get("R_j", parameter.get("R_j", 0.0)))
    focus_count = base_parameter.get("focus_count", 1)
    parameter["focus_count"] = int(focus_count)
    parameter.pop("focus_counts", None)
    if parameter["focus_count"] == 1:
        parameter["collapse_types"] = True
    return parameter


def _transition_bounds(entry: Mapping[str, Any]) -> Tuple[float | None, float | None]:
    evaluations = entry.get("evaluations") or []
    valid = [evt for evt in evaluations if evt and evt.get("status") == "ok"]
    if not valid:
        return None, None
    valid.sort(key=lambda item: float(item.get("value", float("inf"))))
    lower_candidates = [float(evt["value"]) for evt in valid if evt.get("count", 0) < 3]
    upper_candidates = [float(evt["value"]) for evt in valid if evt.get("count", 0) >= 3]
    lower = max(lower_candidates) if lower_candidates else None
    upper = min(upper_candidates) if upper_candidates else None
    return lower, upper


def _collect_debug_samples(
    current_entry: Mapping[str, Any],
    previous_entry: Mapping[str, Any] | None,
    *,
    projection_entry: Mapping[str, Any] | None = None,
    projection_target: float | None = None,
    projection_step: float = 0.1,
) -> List[Tuple[float, Mapping[str, Any], str]]:
    samples: List[Tuple[float, Mapping[str, Any], str]] = []
    if previous_entry and previous_entry.get("bifurcation") is not None:
        samples.append((float(previous_entry["bifurcation"]), previous_entry, "previous_bifurcation"))
    lower, upper = _transition_bounds(current_entry)
    if lower is not None:
        samples.append((float(lower), current_entry, "lower_bound"))
    if upper is not None:
        samples.append((float(upper), current_entry, "upper_bound"))
    if current_entry.get("bifurcation") is not None:
        samples.append((float(current_entry["bifurcation"]), current_entry, "current_bifurcation"))
    if (
        projection_entry
        and projection_entry.get("status") == "ok"
        and projection_entry.get("bifurcation") is not None
        and projection_target is not None
        and math.isfinite(float(projection_target))
    ):
        left_value = float(projection_entry.get("bifurcation"))
        delta = max(float(projection_step), 1e-3)
        projected = float(projection_target) - 1e-3
        projected = min(projected, left_value + delta)
        if projected > left_value + 1e-4:
            samples.append((projected, projection_entry, "projected_lower_branch"))
    seen = set()
    unique_samples: List[Tuple[float, Mapping[str, Any], str]] = []
    for value, source, label in samples:
        key = (round(value, 6), id(source), label)
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append((value, source, label))
    return unique_samples


def _compute_erf_snapshot(
    parameter: Mapping[str, Any],
    r_eplus: float,
    sweep_kwargs: Mapping[str, Any],
    filter_threshold: float,
) -> Dict[str, Any] | None:
    param = dict(parameter)
    param["R_Eplus"] = float(r_eplus)
    result = EIClusterNetwork.generate_erf_curve(param, **sweep_kwargs)
    if not result.completed or not result.x_data:
        return None
    sweep_entry = (result.x_data, result.y_data, result.solves, param)
    fixpoints = EIClusterNetwork.compute_fixpoints(
        sweep_entry,
        kappa=param.get("kappa"),
        connection_type=param.get("connection_type"),
    )
    filtered, _ = _filter_fixpoint_candidates(
        fixpoints,
        threshold=filter_threshold,
        max_fixpoints=3,
    )
    fix_list = [
        {"value": float(point), "stability": details.get("stability", "unstable")}
        for point, details in sorted(filtered.items(), key=lambda item: float(item[0]))
    ]
    return {
        "value": float(r_eplus),
        "x": list(result.x_data),
        "y": list(result.y_data),
        "fixpoints": fix_list,
    }


def _plot_debug_snapshots(
    *,
    segment: Dict[str, Any],
    snapshots: List[Dict[str, Any]],
    output_path: str,
    connection_type: str,
) -> None:
    cols = len(snapshots)
    fig, axes = plt.subplots(1, cols, figsize=(4.5 * cols, 4.5), sharey=True)
    axis_list = [axes] if cols == 1 else list(axes)
    for ax, snapshot in zip(axis_list, snapshots):
        ax.plot(snapshot["x"], snapshot["y"], color="#1f77b4", label="ERF")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="Identity")
        for fix in snapshot["fixpoints"]:
            stable = fix.get("stability") == "stable"
            edge = "k" if stable else "#d62728"
            face = "k" if stable else "none"
            ax.scatter(
                [fix["value"]],
                [fix["value"]],
                facecolors=face,
                edgecolors=edge,
                marker="o",
                s=40,
            )
        scale = snapshot.get("prob_scale")
        conn = snapshot.get("mean_connectivity")
        src = snapshot.get("source_label", "")
        annotation = f"R_Eplus = {snapshot['value']:.3f}"
        ax.set_title(annotation)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        detail_lines = []
        if scale is not None and math.isfinite(scale):
            detail_lines.append(f"scale={scale:.3f}")
        if conn is not None and math.isfinite(conn):
            detail_lines.append(f"mean_conn={conn:.3f}")
        if src:
            detail_lines.append(f"source={src}")
        if detail_lines:
            ax.text(
                0.02,
                0.95,
                "\n".join(detail_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
    for idx, snapshot in enumerate(snapshots):
        if snapshot.get("is_bifurcation"):
            axis_list[idx].set_title(f"{axis_list[idx].get_title()} (bifurcation)")
    axis_list[0].set_ylabel("v_out")
    axis_list[-1].set_xlabel("v_in")
    gap = segment.get("gap", float("nan"))
    kappa = segment.get("kappa", float("nan"))
    r_j = segment.get("r_j", float("nan"))
    fig.suptitle(f"Debug ERFs ({connection_type}) kappa={kappa:.3f}, R_j={r_j:.3f}, gap={gap:.3f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def _generate_debug_reports(
    segments: List[Dict[str, Any]],
    lookup: Dict[CacheKey, Dict[str, Any]],
    base_parameter: Mapping[str, Any],
    sweep_kwargs: Mapping[str, Any],
    filter_threshold: float,
    *,
    output_folder: str,
    limit: int,
    connection_type: str,
    projection_step: float,
) -> None:
    if not segments:
        return
    os.makedirs(output_folder, exist_ok=True)
    produced = 0
    for index, segment in enumerate(sorted(segments, key=lambda item: -item.get("gap", 0.0))):
        if produced >= limit:
            break
        curr_key = segment["right"][2]
        prev_key = segment["left"][2]
        current_entry = lookup.get(curr_key)
        previous_entry = lookup.get(prev_key)
        if current_entry is None:
            continue
        samples = _collect_debug_samples(
            current_entry,
            previous_entry,
            projection_entry=previous_entry,
            projection_target=float(segment["right"][1]),
            projection_step=projection_step,
        )
        snapshots: List[Dict[str, Any]] = []
        for value, source_entry, label in samples:
            parameter = _prepare_parameter_for_entry(base_parameter, source_entry)
            snapshot = _compute_erf_snapshot(parameter, value, sweep_kwargs, filter_threshold)
            if not snapshot:
                continue
            snapshot["is_bifurcation"] = math.isclose(
                value,
                float(source_entry.get("bifurcation", float("nan"))),
                rel_tol=1e-3,
                abs_tol=1e-3,
            )
            snapshot["prob_scale"] = float(source_entry.get("prob_scale", float("nan")))
            snapshot["mean_connectivity"] = float(source_entry.get("mean_connectivity", float("nan")))
            snapshot["source_label"] = label
            snapshots.append(snapshot)
        if not snapshots:
            continue
        encoded_kappa = f"{segment['kappa']:.2f}".replace(".", "_")
        encoded_rj = f"{segment['r_j']:.2f}".replace(".", "_")
        filename = os.path.join(
            output_folder,
            f"debug_kappa{encoded_kappa}_Rj{encoded_rj}_idx{index}.png",
        )
        _plot_debug_snapshots(
            segment=segment,
            snapshots=snapshots,
            output_path=filename,
            connection_type=connection_type,
        )
        produced += 1


def _apply_corrections(
    segments: List[Dict[str, Any]],
    lookup: Dict[CacheKey, Dict[str, Any]],
    base_parameter: Mapping[str, Any],
    sweep_kwargs: Mapping[str, Any],
    search: Mapping[str, Any],
    filter_threshold: float,
    *,
    window: float,
    required_fixpoints: int,
) -> bool:
    if window <= 0:
        return False
    corrections_applied = False
    MIN_IMPROVEMENT = 1e-3
    for segment in sorted(segments, key=lambda item: -item.get("gap", 0.0)):
        right_key: CacheKey = segment["right"][2]
        left_value = float(segment["left"][1])
        updated = False
        while True:
            entry = lookup.get(right_key)
            if not entry or entry.get("status") != "ok":
                break
            current_bif = float(entry.get("bifurcation", float("inf")))
            if not math.isfinite(current_bif):
                break
            branch_min = max(search["min"], left_value - window)
            branch_max = min(search["max"], min(current_bif - MIN_IMPROVEMENT, left_value + window))
            if branch_max - branch_min <= MIN_IMPROVEMENT:
                break
            kappa = float(entry.get("kappa", 0.0))
            r_j = float(entry.get("R_j", 0.0))
            prob_scale = float(entry.get("prob_scale", 1.0))
            correction_search = dict(search)
            correction_search["min"] = branch_min
            correction_search["max"] = branch_max
            correction = _bisection_search(
                base_parameter,
                kappa,
                r_j,
                prob_scale,
                sweep_kwargs,
                correction_search,
                filter_threshold,
                required_fixpoints,
            )
            if correction.get("status") != "ok":
                break
            new_bif = float(correction.get("bifurcation", float("inf")))
            if not math.isfinite(new_bif) or new_bif >= current_bif - MIN_IMPROVEMENT:
                break
            lookup[right_key] = correction
            corrections_applied = True
            updated = True
            print(
                f"Correction sweep replaced bifurcation for kappa={kappa:.3f}, "
                f"R_j={r_j:.3f}, scale={prob_scale:.3f} "
                f"from {current_bif:.3f} to {new_bif:.3f}."
            )
            if abs(new_bif - left_value) <= MIN_IMPROVEMENT:
                break
        if updated:
            gap_after = abs(float(lookup[right_key].get("bifurcation", float("inf"))) - left_value)
            if gap_after <= MIN_IMPROVEMENT:
                print(
                    f"Branch merged near left segment for kappa={lookup[right_key]['kappa']:.3f}, "
                    f"R_j={lookup[right_key]['R_j']:.3f}."
                )
    return corrections_applied


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    parameter["focus_count"] = _resolve_focus_count(parameter, args.focus_count)
    if parameter["focus_count"] == 1:
        parameter["collapse_types"] = True
    base_kappa = float(parameter.get("kappa", 0.0))
    base_rj = float(parameter.get("R_j", 0.0))
    kappa_values = _resolve_value_list(args.kappa, args.kappa_range, base_kappa)
    rj_values = _resolve_value_list(args.rj, args.rj_range, base_rj)
    prob_scales = _resolve_scale_values(args.prob_scale, args.prob_scale_range)
    search = {
        "min": float(args.r_eplus_min),
        "max": float(args.r_eplus_max),
        "tol": float(args.bisection_tol),
        "max_iter": int(args.max_iterations),
    }
    if search["max"] <= search["min"]:
        raise ValueError("R_Eplus max must be larger than the min bound.")
    sweep_kwargs = dict(
        start=float(args.v_start),
        end=float(args.v_end),
        step_number=int(args.v_steps),
        retry_step=args.retry_step,
    )
    filtered = _taggable_configuration(parameter)
    filtered.pop("R_j", None)
    filtered.pop("kappa", None)
    filtered.pop("R_Eplus", None)
    filtered.pop("focus_counts", None)
    filtered["focus_count"] = parameter.get("focus_count", 1)
    metadata = {
        "parameter": filtered,
        "kappa_values": kappa_values,
        "rj_values": rj_values,
        "prob_scales": prob_scales,
        "search": search,
        "sweep": sweep_kwargs,
    }
    tag = sim_tag_from_cfg(metadata)
    connection_type = parameter.get("connection_type", "bernoulli")
    cache_root = os.path.abspath(args.cache_root)
    cache_folder, cache_path = _cache_paths(cache_root, connection_type, tag)
    params_path = os.path.join(cache_folder, "params.yaml")
    if args.overwrite_cache and os.path.exists(cache_path):
        os.remove(cache_path)
    existing_entries = {} if args.overwrite_cache else _load_existing(cache_path)
    lookup: Dict[Tuple[float, float, float], Dict[str, Any]] = dict(existing_entries)
    summary_cfg = dict(metadata)
    write_yaml_config(summary_cfg, params_path)
    tasks: List[Tuple[float, float, float, Dict[str, Any], Dict[str, Any], Dict[str, Any], float, int]] = []
    base_parameter = dict(parameter)
    for kappa in kappa_values:
        for r_j in rj_values:
            for prob_scale in prob_scales:
                key = _cache_key(kappa, r_j, prob_scale)
                if key in lookup:
                    continue
                tasks.append(
                    (
                        float(kappa),
                        float(r_j),
                        float(prob_scale),
                        base_parameter,
                        sweep_kwargs,
                        search,
                        args.filter_threshold,
                        int(max(2, args.fixpoint_threshold)),
                    )
                )
    if tasks:
        jobs = args.jobs or mp.cpu_count()
        if jobs <= 1:
            new_entries = [_task_entry(task) for task in tasks]
        else:
            with mp.Pool(processes=jobs) as pool:
                new_entries = pool.map(_task_entry, tasks)
        for entry in new_entries:
            key = _cache_key(entry["kappa"], entry["R_j"], entry.get("prob_scale", 1.0))
            lookup[key] = entry
    else:
        print("All requested (kappa, R_j) pairs already cached.")
    metadata["cache_folder"] = cache_folder
    curves = _collect_plot_data(lookup)
    segments = _detect_jump_segments(curves, threshold=float(args.jump_threshold))
    if args.correction_sweep and segments:
        corrected = _apply_corrections(
            segments,
            lookup,
            base_parameter,
            sweep_kwargs,
            search,
            args.filter_threshold,
            window=float(args.correction_window),
            required_fixpoints=int(max(2, args.fixpoint_threshold)),
        )
        if corrected:
            print("Recomputing curves after correction sweep.")
            curves = _collect_plot_data(lookup)
            segments = _detect_jump_segments(curves, threshold=float(args.jump_threshold))
        else:
            print("Correction sweep requested but no entries were updated.")
    _save_cache(cache_path, metadata, lookup)
    output_path = args.output or _default_output_path(connection_type, tag)
    generate_plot(curves, output_path=output_path, connection_type=str(connection_type), tag=tag)
    if args.debug_jumps:
        if segments:
            _generate_debug_reports(
                segments,
                lookup,
                base_parameter,
                sweep_kwargs,
                args.filter_threshold,
                output_folder=args.debug_output,
                limit=int(args.debug_limit),
                connection_type=str(connection_type),
                projection_step=float(max(args.projection_step, 0.01)),
            )
        else:
            print("Debugging enabled but no jump segments exceeded the threshold.")


if __name__ == "__main__":
    main()
