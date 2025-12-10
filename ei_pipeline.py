from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pickle

from ei_cluster_network import EIClusterNetwork
from rate_system import (
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
    parser.add_argument("--focus-count", type=int, help="Number of focus populations (starting from index 0).")
    return parser.parse_args()


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


def _erf_filename(value: float) -> str:
    encoded = f"{value:.2f}".replace(".", "_")
    return f"R_Eplus{encoded}.pkl"


def _plot_erf_collection(
    data: Dict[str, Sequence],
    *,
    parameter: Dict,
    kappa: float,
    connection_type: str,
) -> None:
    curves: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for key, entry in data.items():
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        x_data = np.asarray(entry[0], dtype=float)
        y_data = np.asarray(entry[1], dtype=float)
        if x_data.size == 0 or y_data.size == 0:
            continue
        entry_param = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
        rep_value = entry_param.get("R_Eplus")
        if rep_value is None:
            try:
                rep_value = float(key)
            except (TypeError, ValueError):
                continue
        curves.append((float(rep_value), x_data, y_data))
    if not curves:
        print("Skipping ERF plotting: no valid curves found.")
        return
    curves.sort(key=lambda item: item[0])
    rep_values = np.array([item[0] for item in curves], dtype=float)
    vmin = float(np.nanmin(rep_values))
    vmax = float(np.nanmax(rep_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print("Skipping ERF plotting: invalid R_Eplus values.")
        return
    if np.isclose(vmin, vmax):
        spread = 0.5 if vmin == 0 else abs(vmin) * 0.1
        vmin -= spread
        vmax += spread
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    plt.figure()
    for rep_value, x_data, y_data in curves:
        plt.plot(x_data, y_data, color=cmap(norm(rep_value)), alpha=0.85, linewidth=1.5)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("v_in")
    plt.ylabel("v_out")
    plt.ylim(-0.025, 1.025)
    plt.xlim(-0.025, 1.025)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("R_Eplus")
    plt.title(f"ERF curves (R_j = {parameter['R_j']})")
    conn_label = str(connection_type).lower().replace(" ", "_")
    encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
    os.makedirs("plots", exist_ok=True)
    output = os.path.join("plots", f"erfs_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}.png")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Stored ERF overview plot at {output}")


def _residual_score(entry: Dict[str, Any]) -> float:
    score = entry.get("residual_norm")
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


def _select_best_from_group(group: List[Tuple[float, Dict[str, Any]]]) -> Tuple[Tuple[float, Dict[str, Any]] | None, List[Tuple[float, Dict[str, Any]]]]:
    if not group:
        return None, []
    ordered = sorted(group, key=lambda item: (_residual_score(item[1]), float(item[0])))
    keep = ordered[0]
    extras = ordered[1:]
    return keep, extras


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
    ordered_candidates = sorted(candidates, key=lambda item: (_residual_score(item[1]), float(item[0])))
    kept = ordered_candidates[:max_fixpoints]
    for dropped in ordered_candidates[max_fixpoints:]:
        excluded.append((dropped[0], dropped[1], "max_fixpoints"))
    return dict(kept), excluded


def _simulate_erf_task(task: Tuple[float, Dict, Dict]) -> Tuple[float, Dict, ERFResult]:
    value, parameter, sweep_kwargs = task
    current_param = dict(parameter)
    current_param["R_Eplus"] = float(value)
    print("-----------------------------")
    print(f"Simulate Network for R_Eplus = {value}")
    result: ERFResult = EIClusterNetwork.generate_erf_curve(current_param, **sweep_kwargs)
    return float(value), current_param, result


def run_simulation(
    args: argparse.Namespace,
    parameter: Dict,
    r_eplus_values: Sequence[float],
) -> str | None:
    filtered = {k: v for k, v in parameter.items()}
    filtered.pop("R_Eplus", None)
    tag = sim_tag_from_cfg(filtered)
    folder = ensure_output_folder(parameter, tag=tag)
    params_path = os.path.join(folder, "params.yaml")
    if args.overwrite_simulation or not os.path.exists(params_path):
        summary = dict(filtered)
        write_yaml_config(summary, params_path)
    produced_any = False
    existing_detected = False
    sweep_kwargs = dict(
        start=args.v_start,
        end=args.v_end,
        step_number=args.v_steps,
        retry_step=args.retry_step,
    )
    tasks: List[Tuple[float, Dict, Dict]] = []
    for value in r_eplus_values:
        file_path = os.path.join(folder, _erf_filename(value))
        if not args.overwrite_simulation and os.path.exists(file_path):
            print(f"Skipping R_Eplus = {value}: existing data at {file_path}")
            existing_detected = True
            continue
        tasks.append((float(value), dict(parameter), sweep_kwargs))
    results: List[Tuple[float, Dict, ERFResult]] = []
    jobs = args.jobs if args.jobs and args.jobs > 0 else mp.cpu_count()
    if tasks:
        if jobs > 1 and len(tasks) > 1:
            with mp.Pool(jobs) as pool:
                results = pool.map(_simulate_erf_task, tasks)
        else:
            results = [_simulate_erf_task(task) for task in tasks]
    for value, current_param, result in results:
        if not result.completed:
            print(f"Skipping serialization for R_Eplus = {value}: solver did not converge for the entire sweep.")
            continue
        produced_any = True
        file_path = os.path.join(folder, _erf_filename(value))
        path = serialize_erf(file_path, current_param, result)
        if path:
            print(f"Stored ERF data at {path}")
    if produced_any or existing_detected:
        return folder
    return None


def run_analysis(folder: str, parameter: Dict, *, plot_erfs: bool = False) -> None:
    print(f"Analyzing data in {folder}")
    bundle_path = aggregate_data(folder)
    with open(bundle_path, "rb") as handle:
        data = pickle.load(handle)
    kappa = parameter.get("kappa", 0.0)
    connection_type = parameter.get("connection_type", "bernoulli")
    if plot_erfs:
        _plot_erf_collection(data, parameter=parameter, kappa=kappa, connection_type=connection_type)
    all_fixpoints: Dict[str, Dict[float, Dict[str, Any]]] = {}
    filtered_threshold = 1e-3
    for key, value in data.items():
        print(f"P_Eplus: {key}")
        fixpoint = EIClusterNetwork.compute_fixpoints(value, kappa=kappa, connection_type=connection_type)
        filtered, excluded = _filter_fixpoint_candidates(fixpoint, threshold=filtered_threshold, max_fixpoints=3)
        kept_values = sorted(float(point) for point in filtered)
        for entry in fixpoint.values():
            entry["included"] = False
            entry["filter_reason"] = None
        for point, entry in filtered.items():
            entry["included"] = True
            entry["filter_reason"] = None
        for point, entry, reason in excluded:
            entry["included"] = False
            entry["filter_reason"] = reason
        if excluded:
            dropped_str = ", ".join(f"{float(val):.5f} ({reason})" for val, _, reason in excluded)
            kept_str = ", ".join(f"{val:.5f}" for val in kept_values) if kept_values else "none"
            print(
                f"Filtered fixpoints for P_Eplus {key}: kept [{kept_str}], "
                f"excluded [{dropped_str}] (threshold={filtered_threshold})."
            )
        all_fixpoints[key] = fixpoint
    x_stable: List[float] = []
    y_stable: List[float] = []
    x_unstable: List[float] = []
    y_unstable: List[float] = []
    for r_value, result in all_fixpoints.items():
        for point, entry in result.items():
            if not entry.get("included"):
                continue
            status = entry.get("stability", "unstable")
            point_value = float(point)
            if status == "stable":
                x_stable.append(float(r_value))
                y_stable.append(point_value)
            else:
                x_unstable.append(float(r_value))
                y_unstable.append(point_value)
    if x_stable or x_unstable:
        plt.figure()
        if x_stable:
            plt.scatter(x_stable, y_stable, color="k", label="stable")
        if x_unstable:
            plt.scatter(
                x_unstable,
                y_unstable,
                facecolors="none",
                edgecolors="k",
                marker="o",
                label="unstable",
            )
        plt.xlabel("P_E+")
        plt.ylabel("v_out")
        plt.ylim(-0.025, 1.025)
        x_values = x_stable + x_unstable
        if x_values:
            plt.xlim(0.995, max(x_values) + 0.05)
        plt.legend()
        plt.title(f"R_j = {parameter['R_j']}")
        conn_label = str(connection_type).lower().replace(" ", "_")
        encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", f"fixpoints_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}.png"))
        plt.close()
    os.makedirs("data", exist_ok=True)
    conn_label = str(connection_type).lower().replace(" ", "_")
    encoded_kappa = f"{float(kappa):.2f}".replace(".", "_")
    output_path = os.path.join("data", f"all_fixpoints_{conn_label}_kappa{encoded_kappa}_Rj{parameter['R_j']}.pkl")
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
        },
        "fixpoints": all_fixpoints,
    }
    with open(output_path, "wb") as file:
        pickle.dump(summary_payload, file)
    print(f"Stored fixpoint summary at {output_path}")


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    if args.focus_count is not None:
        parameter["focus_count"] = max(1, int(args.focus_count))
    if args.full_focus_system:
        parameter["collapse_types"] = False
        parameter["focus_count"] = 1
    else:
        parameter.setdefault("collapse_types", True)
    r_eplus_values = resolve_r_eplus(args, parameter)
    folder: str | None = args.folder
    if not args.analysis_only:
        folder = run_simulation(args, parameter, r_eplus_values)
    if not args.simulation_only:
        if folder is None:
            if args.folder:
                folder = args.folder
            else:
                print("Skipping analysis: no ERF data folder was provided or generated.")
                return
        run_analysis(folder, parameter, plot_erfs=args.plot)


if __name__ == "__main__":
    main()
