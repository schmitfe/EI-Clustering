from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pickle

from MeanField.ei_cluster_network import EIClusterNetwork
from MeanField.rate_system import (
    ERFResult,
    aggregate_data,
    ensure_output_folder,
    serialize_erf,
)
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg, write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate and analyze EI-cluster networks.")
    parser.add_argument("--simulation-only", action="store_true", help="Run only the simulation stage.")
    parser.add_argument("--analysis-only", action="store_true", help="Run only the analysis stage.")
    parser.add_argument("--folder", type=str, help="Existing data folder for analysis-only runs.")
    parser.add_argument("--v-start", type=float, default=0.0, help="ERF sweep start value.")
    parser.add_argument("--v-end", type=float, default=1.0, help="ERF sweep end value.")
    parser.add_argument("--v-steps", type=int, default=1000, help="Number of ERF samples.")
    parser.add_argument("--retry-step", type=float, default=None, help="Optional retry increment for solver restarts.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot an aggregated ERF figure during analysis instead of per-simulation plots.",
    )
    parser.add_argument("--jobs", type=int, help="Number of parallel workers (default: CPU count).")
    parser.add_argument("--r-eplus", type=float, action="append", help="Explicit R_Eplus values (can repeat).")
    parser.add_argument("--r-eplus-start", type=float, help="Start of the R_Eplus sweep.")
    parser.add_argument("--r-eplus-end", type=float, help="End of the R_Eplus sweep.")
    parser.add_argument("--r-eplus-step", type=float, help="Step size for the R_Eplus sweep.")
    parser.add_argument("--overwrite-simulation", action="store_true", help="Regenerate ERFs even if files exist.")
    parser.add_argument("--full-focus-system", action="store_true", help="Disable type grouping so the solver runs on all non-focus populations (2Q-1 system).")
    add_override_arguments(parser)
    parser.add_argument(
        "--focus-count",
        type=int,
        nargs="+",
        help="One or more focus population counts (starting from index 0).",
    )
    return parser.parse_args()


def _flatten_focus_entries(source) -> List[int]:
    if source is None:
        return []
    if isinstance(source, str):
        cleaned = (
            source.replace(";", ",")
            .replace("[", "")
            .replace("]", "")
            .split(",")
        )
        items = []
        for chunk in cleaned:
            text = chunk.strip()
            if not text:
                continue
            try:
                items.append(int(float(text)))
            except (TypeError, ValueError):
                continue
        return items
    if isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray)):
        values: List[int] = []
        for entry in source:
            values.extend(_flatten_focus_entries(entry))
        return values
    try:
        return [int(source)]
    except (TypeError, ValueError):
        return []


def _normalize_focus_counts(values: Sequence[int]) -> List[int]:
    normalized = sorted({max(1, int(entry)) for entry in values if entry is not None})
    return normalized


def resolve_focus_counts(args: argparse.Namespace, parameter: Dict[str, Any]) -> List[int]:
    if args.focus_count is not None:
        values = _flatten_focus_entries(args.focus_count)
    else:
        raw = parameter.get("focus_counts")
        if raw is None:
            raw = parameter.get("focus_count")
        values = _flatten_focus_entries(raw)
    normalized = _normalize_focus_counts(values)
    if normalized:
        return normalized
    fallback_raw = parameter.get("focus_count") or 1
    fallback = _normalize_focus_counts(_flatten_focus_entries(fallback_raw))
    return fallback if fallback else [1]


def _taggable_configuration(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {k: v for k, v in parameter.items()}
    filtered.pop("R_Eplus", None)
    filtered.pop("focus_count", None)
    filtered.pop("focus_counts", None)
    return filtered


def _entry_parameter(entry: Sequence[Any]) -> Dict[str, Any]:
    if isinstance(entry, Sequence) and len(entry) > 3 and isinstance(entry[3], dict):
        return entry[3]
    return {}


def _entry_focus_count(entry: Sequence[Any]) -> int:
    parameter = _entry_parameter(entry)
    focus_value = parameter.get("focus_count")
    if focus_value is None:
        return 1
    try:
        return max(1, int(focus_value))
    except (TypeError, ValueError):
        return 1


def _entry_r_eplus(entry: Sequence[Any], fallback: Any = None) -> float:
    parameter = _entry_parameter(entry)
    value = parameter.get("R_Eplus", fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _group_erf_data(data: Dict[str, Sequence[Any]]) -> Dict[int, Dict[str, Sequence[Any]]]:
    grouped: Dict[int, Dict[str, Sequence[Any]]] = {}
    for key, entry in data.items():
        focus_count = _entry_focus_count(entry)
        grouped.setdefault(focus_count, {})[key] = entry
    return grouped


def resolve_r_eplus(args: argparse.Namespace, parameter: Dict) -> List[float]:
    if args.r_eplus:
        return [float(val) for val in args.r_eplus]
    if args.r_eplus_start is not None and args.r_eplus_end is not None and args.r_eplus_step is not None:
        vals = np.arange(args.r_eplus_start, args.r_eplus_end + 1e-12, args.r_eplus_step)
        return [float(v) for v in vals]
    base_value = parameter.get("R_Eplus")
    if base_value is not None:
        return [float(base_value)]
    return [float(parameter.get("Q", 1.0))]


def _erf_filename(value: float, *, focus_count: int) -> str:
    encoded = f"{value:.2f}".replace(".", "_")
    return f"R_Eplus{encoded}_focus{focus_count}.pkl"


def _key_to_r_eplus(key: Any) -> float:
    text = str(key)
    token = text.split("_focus", 1)[0]
    try:
        return float(token)
    except (TypeError, ValueError):
        return float("nan")


def _plot_erf_collection(
    grouped_data: Dict[int, Dict[str, Sequence]],
    focus_counts: Sequence[int],
    *,
    parameter: Dict,
    kappa: float,
    connection_type: str,
    tag: str | None = None,
) -> None:
    focus_order = [int(fc) for fc in focus_counts if fc in grouped_data and grouped_data[fc]]
    if not focus_order:
        print("Skipping ERF plotting: no valid curves for the requested focus counts.")
        return
    focus_curves: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = {}
    rep_values: List[float] = []
    for focus_count in focus_order:
        curves: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for key, entry in grouped_data.get(focus_count, {}).items():
            if not isinstance(entry, Sequence) or len(entry) < 2:
                continue
            x_data = np.asarray(entry[0], dtype=float)
            y_data = np.asarray(entry[1], dtype=float)
            if x_data.size == 0 or y_data.size == 0:
                continue
            rep_value = _entry_r_eplus(entry, fallback=_key_to_r_eplus(key))
            if not np.isfinite(rep_value):
                continue
            curves.append((float(rep_value), x_data, y_data))
        if curves:
            curves.sort(key=lambda item: item[0])
            focus_curves[focus_count] = curves
            rep_values.extend([item[0] for item in curves])
    plot_order = [fc for fc in focus_order if fc in focus_curves]
    if not plot_order:
        print("Skipping ERF plotting: no valid curves found.")
        return
    rep_array = np.array(rep_values, dtype=float)
    vmin = float(np.nanmin(rep_array))
    vmax = float(np.nanmax(rep_array))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print("Skipping ERF plotting: invalid R_Eplus values.")
        return
    if np.isclose(vmin, vmax):
        spread = 0.5 if vmin == 0 else abs(vmin) * 0.1
        vmin -= spread
        vmax += spread
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rows = len(plot_order)
    fig, axes = plt.subplots(rows, 1, sharex=True, sharey=True, figsize=(6, 3.5 * rows))
    axis_list = np.atleast_1d(axes)
    for idx, focus_count in enumerate(plot_order):
        ax = axis_list[idx]
        curves = focus_curves[focus_count]
        for rep_value, x_data, y_data in curves:
            ax.plot(x_data, y_data, color=cmap(norm(rep_value)), alpha=0.85, linewidth=1.5)
        ax.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=1.0)
        ax.set_ylabel("v_out")
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlim(-0.025, 1.025)
        ax.set_title(f"focus_count = {focus_count}")
    axis_list[-1].set_xlabel("v_in")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axis_list.tolist(), label="R_Eplus")
    tag_label = f", tag={tag}" if tag else ""
    fig.suptitle(f"ERF curves (R_j = {parameter['R_j']}{tag_label})")
    conn_label = str(connection_type).lower().replace(" ", "_")
    encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
    os.makedirs("plots", exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    output = os.path.join("plots", f"erfs_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}{tag_suffix}.png")
    #fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output)
    plt.close(fig)
    print(f"Stored ERF overview plot at {output}")


def _plot_fixpoint_collection(
    scatter_points: Dict[int, Dict[str, List[Tuple[float, float]]]],
    focus_counts: Sequence[int],
    *,
    parameter: Dict,
    kappa: float,
    connection_type: str,
    tag: str | None = None,
) -> None:
    plot_order = [
        fc
        for fc in focus_counts
        if scatter_points.get(fc, {}).get("stable") or scatter_points.get(fc, {}).get("unstable")
    ]
    if not plot_order:
        print("Skipping fixpoint plotting: no included fixpoints for the requested focus counts.")
        return
    rows = len(plot_order)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(6, 3.2 * rows))
    axis_list = np.atleast_1d(axes)
    all_x_values: List[float] = []
    for focus_count in plot_order:
        entries = scatter_points.get(focus_count, {})
        for key in ("stable", "unstable"):
            all_x_values.extend([point[0] for point in entries.get(key, [])])
    default_xlim = (0.0, 1.0)
    if all_x_values:
        xmin = min(all_x_values)
        xmax = max(all_x_values)
        margin = 0.05 * max(1.0, abs(xmax - xmin) or 1.0)
        default_xlim = (xmin - margin, xmax + margin)
    for idx, focus_count in enumerate(plot_order):
        ax = axis_list[idx]
        stable_points = scatter_points.get(focus_count, {}).get("stable", [])
        unstable_points = scatter_points.get(focus_count, {}).get("unstable", [])
        if stable_points:
            xs, ys = zip(*stable_points)
            ax.scatter(xs, ys, color="k", label="stable")
        if unstable_points:
            xs, ys = zip(*unstable_points)
            ax.scatter(xs, ys, facecolors="none", edgecolors="k", marker="o", label="unstable")
        if not stable_points and not unstable_points:
            ax.text(0.5, 0.5, "No fixpoints", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("v_out")
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlim(*default_xlim)
        ax.set_title(f"focus_count = {focus_count}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    axis_list[-1].set_xlabel("P_E+")
    tag_label = f", tag={tag}" if tag else ""
    fig.suptitle(f"R_j = {parameter['R_j']}{tag_label}")
    conn_label = str(connection_type).lower().replace(" ", "_")
    encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
    os.makedirs("plots", exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    output = os.path.join(
        "plots",
        f"fixpoints_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}{tag_suffix}.png",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output)
    plt.close(fig)
    print(f"Stored fixpoint plot at {output}")


def _residual_score(entry: Dict[str, Any]) -> float:
    score = entry.get("residual_norm")
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


def _candidate_rank(point: float, entry: Dict[str, Any]) -> Tuple[int, float, float, float]:
    stability_rank = 0 if entry.get("stability") == "stable" else 1
    residual_rank = _residual_score(entry)
    magnitude_rank = -abs(float(point))
    value_rank = -float(point)
    return stability_rank, residual_rank, magnitude_rank, value_rank


def _select_best_from_group(group: List[Tuple[float, Dict[str, Any]]]) -> Tuple[Tuple[float, Dict[str, Any]] | None, List[Tuple[float, Dict[str, Any]]]]:
    if not group:
        return None, []
    ordered = sorted(group, key=lambda item: _candidate_rank(item[0], item[1]))
    keep = ordered[0]
    extras = ordered[1:]
    return keep, extras


def _min_distance_to_selected(value: float, selected: List[Tuple[float, Dict[str, Any]]]) -> float:
    if not selected:
        return float("inf")
    distance = min(abs(float(value) - float(item[0])) for item in selected)
    return float(distance)


def _select_diverse_subset(
    candidates: List[Tuple[float, Dict[str, Any]]],
    max_fixpoints: int,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], List[Tuple[float, Dict[str, Any]]]]:
    if len(candidates) <= max_fixpoints:
        return list(candidates), []
    pool = sorted(candidates, key=lambda item: _candidate_rank(item[0], item[1]))
    selected: List[Tuple[float, Dict[str, Any]]] = []
    if pool:
        selected.append(pool.pop(0))
    while len(selected) < max_fixpoints and pool:
        pool.sort(
            key=lambda item: (
                -_min_distance_to_selected(item[0], selected),
                _candidate_rank(item[0], item[1]),
            )
        )
        selected.append(pool.pop(0))
    dropped = pool
    return selected, dropped


def _is_trivial_fixpoint(point: float, entry: Dict[str, Any], *, threshold: float = 1e-6) -> bool:
    rates = entry.get("rates")
    if rates is None:
        return False
    arr = np.asarray(rates, dtype=float).ravel()
    if arr.size == 0:
        return False
    return (abs(float(point)) <= threshold) and np.all(np.abs(arr) <= threshold)


def _describe_fixpoints(
    entries,
    *,
    include_reason: bool = False,
) -> str:
    if not entries:
        return "none"
    chunks = []
    for item in entries:
        if include_reason:
            point, entry, reason = item
        else:
            point, entry = item
            reason = None
        stability = entry.get("stability", "?")
        text = f"{float(point):.5f} [{stability}]"
        if include_reason and reason:
            text = f"{text} ({reason})"
        chunks.append(text)
    return ", ".join(chunks)


def _filter_fixpoint_candidates(
    fixpoints: Dict[float, Dict[str, Any]],
    *,
    threshold: float = 1e-3,
    max_fixpoints: int = 3,
) -> Tuple[Dict[float, Dict[str, Any]], List[Tuple[float, Dict[str, Any], str]]]:
    if not fixpoints:
        return {}, []
    sorted_points = sorted(fixpoints.items(), key=lambda item: float(item[0]))
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    excluded: List[Tuple[float, Dict[str, Any], str]] = []
    current_group: List[Tuple[float, Dict[str, Any]]] = []
    last_value: float | None = None
    for key, entry in sorted_points:
        value = float(key)
        if _is_trivial_fixpoint(value, entry):
            excluded.append((key, entry, "trivial_zero"))
            continue
        if last_value is None or abs(value - last_value) <= threshold:
            current_group.append((key, entry))
        else:
            keep, extras = _select_best_from_group(current_group)
            if keep is not None:
                candidates.append(keep)
            excluded.extend((val, item, "duplicate_threshold") for val, item in extras)
            current_group = [(key, entry)]
        last_value = value
    keep, extras = _select_best_from_group(current_group)
    if keep is not None:
        candidates.append(keep)
    excluded.extend((val, item, "duplicate_threshold") for val, item in extras)
    if len(candidates) <= max_fixpoints:
        return dict(candidates), excluded
    kept, dropped = _select_diverse_subset(candidates, max_fixpoints)
    for drop in dropped:
        excluded.append((drop[0], drop[1], "max_fixpoints"))
    return dict(kept), excluded


def _simulate_erf_task(task: Tuple[float, int, Dict, Dict]) -> Tuple[float, int, Dict, ERFResult]:
    value, focus_count, parameter, sweep_kwargs = task
    current_param = dict(parameter)
    current_param["R_Eplus"] = float(value)
    current_param["focus_count"] = int(focus_count)
    print("-----------------------------")
    print(f"Simulate Network for R_Eplus = {value} (focus_count={focus_count})")
    result: ERFResult = EIClusterNetwork.generate_erf_curve(current_param, **sweep_kwargs)
    return float(value), int(focus_count), current_param, result


def run_simulation(
    args: argparse.Namespace,
    parameter: Dict,
    r_eplus_values: Sequence[float],
    focus_counts: Sequence[int],
) -> str | None:
    filtered = _taggable_configuration(parameter)
    tag = sim_tag_from_cfg(filtered)
    folder = ensure_output_folder(parameter, tag=tag)
    params_path = os.path.join(folder, "params.yaml")
    if args.overwrite_simulation or not os.path.exists(params_path):
        summary = dict(filtered)
        summary["focus_counts"] = list(focus_counts)
        write_yaml_config(summary, params_path)
    produced_any = False
    existing_detected = False
    sweep_kwargs = dict(
        start=args.v_start,
        end=args.v_end,
        step_number=args.v_steps,
        retry_step=args.retry_step,
    )
    tasks: List[Tuple[float, int, Dict, Dict]] = []
    for focus_count in focus_counts:
        for value in r_eplus_values:
            file_path = os.path.join(folder, _erf_filename(value, focus_count=focus_count))
            if not args.overwrite_simulation and os.path.exists(file_path):
                print(f"Skipping R_Eplus = {value} (focus_count={focus_count}): existing data at {file_path}")
                existing_detected = True
                continue
            task_param = dict(parameter)
            task_param["focus_count"] = int(focus_count)
            tasks.append((float(value), int(focus_count), task_param, sweep_kwargs))
    results: List[Tuple[float, int, Dict, ERFResult]] = []
    jobs = args.jobs if args.jobs and args.jobs > 0 else mp.cpu_count()
    if tasks:
        if jobs > 1 and len(tasks) > 1:
            with mp.Pool(jobs) as pool:
                results = pool.map(_simulate_erf_task, tasks)
        else:
            results = [_simulate_erf_task(task) for task in tasks]
    for value, focus_count, current_param, result in results:
        if not result.completed:
            print(f"Skipping serialization for R_Eplus = {value}: solver did not converge for the entire sweep.")
            continue
        produced_any = True
        file_path = os.path.join(folder, _erf_filename(value, focus_count=focus_count))
        path = serialize_erf(file_path, current_param, result, focus_count=focus_count)
        if path:
            print(f"Stored ERF data at {path}")
    if produced_any or existing_detected:
        return folder
    return None


def run_analysis(folder: str, parameter: Dict, focus_counts: Sequence[int], *, plot_erfs: bool = False) -> None:
    print(f"Analyzing data in {folder}")
    bundle_path = aggregate_data(folder)
    with open(bundle_path, "rb") as handle:
        data = pickle.load(handle)
    grouped_data = _group_erf_data(data)
    available_focus = sorted(grouped_data.keys())
    selected_focus_counts = [fc for fc in focus_counts if fc in grouped_data]
    if not selected_focus_counts:
        print(
            f"No ERF data found for requested focus counts {list(focus_counts)}. "
            f"Available focus counts in {folder}: {available_focus}."
        )
        return
    missing_focus = [fc for fc in focus_counts if fc not in grouped_data]
    if missing_focus:
        print(
            f"Skipping focus counts without data: {missing_focus}. "
            f"Available focus counts: {available_focus}."
        )
    kappa = parameter.get("kappa", 0.0)
    connection_type = parameter.get("connection_type", "bernoulli")
    filtered = _taggable_configuration(parameter)
    analysis_tag = sim_tag_from_cfg(filtered)
    if plot_erfs:
        _plot_erf_collection(
            grouped_data,
            selected_focus_counts,
            parameter=parameter,
            kappa=kappa,
            connection_type=connection_type,
            tag=analysis_tag,
        )
    filtered_threshold = 1e-3
    focus_fixpoints: Dict[int, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    scatter_points: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    for focus_count in selected_focus_counts:
        scatter_points[focus_count] = {"stable": [], "unstable": []}
        entries = grouped_data.get(focus_count, {})
        ordered_entries = sorted(
            entries.items(),
            key=lambda item: _entry_r_eplus(item[1], fallback=_key_to_r_eplus(item[0])),
        )
        focus_fixpoints[focus_count] = {}
        for key, sweep_entry in ordered_entries:
            r_value = _entry_r_eplus(sweep_entry, fallback=_key_to_r_eplus(key))
            print(f"P_Eplus: {r_value:.5f} (focus_count={focus_count})")
            fixpoint = EIClusterNetwork.compute_fixpoints(sweep_entry, kappa=kappa, connection_type=connection_type)
            filtered_points, excluded = _filter_fixpoint_candidates(fixpoint, threshold=filtered_threshold, max_fixpoints=3)
            for entry in fixpoint.values():
                entry["included"] = False
                entry["filter_reason"] = None
            for point, entry in filtered_points.items():
                entry["included"] = True
                entry["filter_reason"] = None
            for point, entry, reason in excluded:
                entry["included"] = False
                entry["filter_reason"] = reason
            kept_entries = sorted(filtered_points.items(), key=lambda item: float(item[0]))
            excluded_entries = sorted(excluded, key=lambda item: float(item[0]))
            if fixpoint:
                kept_str = _describe_fixpoints(kept_entries)
                excluded_str = _describe_fixpoints(excluded_entries, include_reason=True)
                print(
                    f"Fixpoints for P_Eplus {r_value:.5f} (focus_count={focus_count}): kept [{kept_str}], "
                    f"excluded [{excluded_str}] (threshold={filtered_threshold})."
                )
            focus_fixpoints[focus_count][key] = fixpoint
            for point, entry in fixpoint.items():
                if not entry.get("included"):
                    continue
                status = entry.get("stability", "unstable")
                target = "stable" if status == "stable" else "unstable"
                scatter_points[focus_count][target].append((float(r_value), float(point)))
    _plot_fixpoint_collection(
        scatter_points,
        selected_focus_counts,
        parameter=parameter,
        kappa=kappa,
        connection_type=connection_type,
        tag=analysis_tag,
    )
    os.makedirs("data", exist_ok=True)
    conn_label = str(connection_type).lower().replace(" ", "_")
    encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
    output_path = os.path.join(
        "data",
        f"all_fixpoints_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}_{analysis_tag}.pkl",
    )
    params_path = os.path.join(folder, "params.yaml")
    params_content = None
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as params_file:
            params_content = params_file.read()
    summary_payload = {
        "metadata": {
            "source_folder": os.path.abspath(folder),
            "params_path": os.path.abspath(params_path) if os.path.exists(params_path) else None,
            "params_yaml": params_content,
            "analysis_parameter": dict(parameter),
            "analysis_focus_counts": list(selected_focus_counts),
            "filtered_threshold": filtered_threshold,
        },
        "fixpoints": focus_fixpoints,
    }
    with open(output_path, "wb") as file:
        pickle.dump(summary_payload, file)
    print(f"Stored fixpoint summary at {output_path}")


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    focus_counts = resolve_focus_counts(args, parameter)
    parameter["focus_counts"] = focus_counts
    if args.full_focus_system:
        parameter["collapse_types"] = False
    else:
        parameter.setdefault("collapse_types", True)
    r_eplus_values = resolve_r_eplus(args, parameter)
    folder: str | None = args.folder
    if not args.analysis_only:
        folder = run_simulation(args, parameter, r_eplus_values, focus_counts)
    if not args.simulation_only:
        if folder is None:
            if args.folder:
                folder = args.folder
            else:
                print("Skipping analysis: no ERF data folder was provided or generated.")
                return
        run_analysis(folder, parameter, focus_counts, plot_erfs=args.plot)


if __name__ == "__main__":
    main()
