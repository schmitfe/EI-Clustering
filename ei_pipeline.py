from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Dict, List, Sequence, Tuple

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
    parser.add_argument("--plot", action="store_true", help="Plot ERF curves during simulation.")
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


def maybe_plot_curve(x: Sequence[float], y: Sequence[float], label: str, folder: str) -> None:
    plt.figure()
    plt.plot(x, y, label=label)
    plt.legend()
    plt.plot([0, 1], [0, 1], "black")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"erf_{label}.png"))
    plt.close()


def _erf_filename(value: float) -> str:
    encoded = f"{value:.2f}".replace(".", "_")
    return f"R_Eplus{encoded}.pkl"


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
        if args.plot:
            maybe_plot_curve(result.x_data, result.y_data, label=f"R_Eplus_{value:.2f}", folder="plots")
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


def run_analysis(folder: str, parameter: Dict) -> None:
    print(f"Analyzing data in {folder}")
    bundle_path = aggregate_data(folder)
    with open(bundle_path, "rb") as handle:
        data = pickle.load(handle)
    kappa = parameter.get("kappa", 0.0)
    connection_type = parameter.get("connection_type", "bernoulli")
    all_fixpoints = {}
    for key, value in data.items():
        print(f"P_Eplus: {key}")
        fixpoint = EIClusterNetwork.compute_fixpoints(value, kappa=kappa, connection_type=connection_type)
        all_fixpoints[key] = fixpoint
    x_stable: List[float] = []
    y_stable: List[float] = []
    x_unstable: List[float] = []
    y_unstable: List[float] = []
    for r_value, result in all_fixpoints.items():
        for point, state in result.items():
            if state == "stable":
                x_stable.append(float(r_value))
                y_stable.append(point)
            else:
                x_unstable.append(float(r_value))
                y_unstable.append(point)
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
    with open(output_path, "wb") as file:
        pickle.dump(all_fixpoints, file)
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
        run_analysis(folder, parameter)


if __name__ == "__main__":
    main()
