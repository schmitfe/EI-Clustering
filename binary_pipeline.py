from __future__ import annotations

import argparse
import gc
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from BinaryNetwork.ClusteredEI_network import ClusteredEI_network

from MeanField.rate_system import ensure_output_folder
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg, write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate clustered binary EI networks using YAML configs.")
    add_override_arguments(parser)
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps from the config.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps from the config.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval from the config.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size from the config.")
    parser.add_argument("--seed", type=int, help="Random seed for numpy.")
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for saved traces (defaults to binary.output_name or 'activity_trace').",
    )
    parser.add_argument(
        "--plot-activity",
        action="store_true",
        help="Render a heatmap from activity_trace.npz and store it next to the binary traces.",
    )
    parser.add_argument(
        "--fixpoints-file",
        type=str,
        help="Path to an all_fixpoints_*.pkl summary to seed binary simulations from fixed points.",
    )
    parser.add_argument(
        "--fixpoint-reps",
        type=float,
        nargs="+",
        help="R_Eplus values (rep grid) to simulate when --fixpoints-file is provided.",
    )
    return parser.parse_args()


def _resolve_binary_config(parameter: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(parameter.get("binary", {}))
    cfg["warmup_steps"] = args.warmup_steps if args.warmup_steps is not None else cfg.get("warmup_steps", 5000)
    cfg["simulation_steps"] = (
        args.simulation_steps if args.simulation_steps is not None else cfg.get("simulation_steps", 20000)
    )
    cfg["sample_interval"] = (
        args.sample_interval if args.sample_interval is not None else cfg.get("sample_interval", 10)
    )
    cfg["batch_size"] = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)
    cfg["seed"] = args.seed if args.seed is not None else cfg.get("seed")
    cfg["output_name"] = args.output_name or cfg.get("output_name", "activity_trace")
    cfg["plot_activity"] = bool(args.plot_activity or cfg.get("plot_activity", False))
    return cfg


def _sample_populations(network: ClusteredEI_network) -> Tuple[List[str], np.ndarray]:
    pops = network.E_pops + network.I_pops
    names = [pop.name for pop in pops]
    values = np.array(
        [float(network.state[pop.view[0]:pop.view[1]].mean()) for pop in pops],
        dtype=float,
    )
    return names, values


def _save_activity_plot(states: np.ndarray, interval: int, parameter: Dict, path: str) -> None:
    if states.size == 0:
        raise RuntimeError("Cannot plot activity: no neuron states were recorded.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to save activity plots. Install it via 'pip install matplotlib'.") from exc
    steps = states.shape[0]
    neuron_count = states.shape[1]
    fig_width = max(steps / 80.0, 6.0)
    fig_height = max(neuron_count / 400.0, 4.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.imshow(states.T, interpolation="none", aspect="auto", origin="lower", cmap="binary")
    ax.set_xlabel(f"Sample index (interval={interval})")
    ax.set_ylabel("Neuron index")
    title = f"Binary activity (R_Eplus={parameter.get('R_Eplus')}, R_j={parameter.get('R_j')})"
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("State (0/1)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_activity_onset_plot(states: np.ndarray, interval: int, parameter: Dict, path: str) -> None:
    if states.size == 0:
        raise RuntimeError("Cannot plot activity: no neuron states were recorded.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to save activity plots. Install it via 'pip install matplotlib'.") from exc
    steps, neuron_count = states.shape
    state_int = states.astype(np.int16, copy=False)
    event_times: List[np.ndarray] = []
    event_neurons: List[np.ndarray] = []
    if steps > 1:
        transitions = np.argwhere(np.diff(state_int, axis=0) == 1)
        if transitions.size:
            event_times.append(transitions[:, 0] + 1)
            event_neurons.append(transitions[:, 1])
    if event_times:
        onset_times = np.concatenate(event_times)
        onset_neurons = np.concatenate(event_neurons)
    else:
        onset_times = np.zeros(0, dtype=int)
        onset_neurons = np.zeros(0, dtype=int)
    excitatory_count = int(parameter.get("N_E", neuron_count) or 0)
    excitatory_count = max(0, min(excitatory_count, neuron_count))
    excit_mask = onset_neurons < excitatory_count
    inhib_mask = onset_neurons >= excitatory_count
    excit_times = onset_times[excit_mask]
    excit_neurons = onset_neurons[excit_mask]
    inhib_times = onset_times[inhib_mask]
    inhib_neurons = onset_neurons[inhib_mask]
    #fig_width = max(steps / 80.0, 6.0)
    #fig_height = max(neuron_count / 400.0, 4.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    if excit_times.size:
        ax.scatter(
            excit_times * interval,
            excit_neurons,
            s=18,
            marker=".",
            color="black",
            linewidths=0.8,
            label="Excitatory",
        )
    if inhib_times.size:
        ax.scatter(
            inhib_times * interval,
            inhib_neurons,
            s=18,
            marker=".",
            color="#8B0000",
            linewidths=0.8,
            label="Inhibitory",
        )
    ax.set_ylim(-0.5, neuron_count - 0.5)
    ax.set_xlabel(f"Sample index (interval={interval})")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"Binary activity onsets (R_Eplus={parameter.get('R_Eplus')}, R_j={parameter.get('R_j')})")
    if excit_times.size and inhib_times.size:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_binary_simulation(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
    output_name: str | None = None,
) -> Dict[str, Any]:
    seed = binary_cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
    network = ClusteredEI_network(parameter)
    network.initialize()
    warmup_steps = int(binary_cfg["warmup_steps"])
    batch_size = int(binary_cfg["batch_size"])
    if warmup_steps > 0:
        network.run(warmup_steps, batch_size=batch_size)
    interval = int(binary_cfg["sample_interval"])
    if interval <= 0:
        raise ValueError("sample_interval must be positive.")
    total_steps = int(binary_cfg["simulation_steps"])
    samples = max(total_steps // interval, 0)
    trace: List[np.ndarray] = []
    state_trace: List[np.ndarray] = []
    names: List[str] = []
    pop_count = len(network.E_pops) + len(network.I_pops)
    for _ in range(samples):
        network.run(interval, batch_size=batch_size)
        names, values = _sample_populations(network)
        trace.append(values)
        state_trace.append(network.state.astype(np.uint8).copy())
    if not names:
        names = [pop.name for pop in network.E_pops + network.I_pops]
    rates = np.vstack(trace) if trace else np.zeros((0, pop_count))
    mean_values = rates.mean(axis=0) if rates.size else np.zeros(pop_count)
    mean_rates = {name: float(value) for name, value in zip(names, mean_values)}
    states = np.vstack(state_trace) if state_trace else np.zeros((0, network.N), dtype=np.uint8)

    filtered = dict(parameter)
    filtered.pop("R_Eplus", None)
    tag = sim_tag_from_cfg(filtered)
    folder = ensure_output_folder(parameter, tag=tag)
    params_path = os.path.join(folder, "params.yaml")
    if not os.path.exists(params_path):
        write_yaml_config(filtered, params_path)
    binary_folder = os.path.join(folder, "binary")
    os.makedirs(binary_folder, exist_ok=True)
    resolved_output_name = output_name or str(binary_cfg["output_name"])
    np.savez_compressed(
        os.path.join(binary_folder, f"{resolved_output_name}.npz"),
        rates=rates,
        times=np.arange(rates.shape[0]) * interval,
        names=np.array(names),
        neuron_states=states,
        warmup_steps=warmup_steps,
        sample_interval=interval,
        batch_size=batch_size,
        seed=seed,
    )
    plot_path = None
    onset_plot_path = None
    if binary_cfg.get("plot_activity"):
        plot_path = os.path.join(binary_folder, f"{resolved_output_name}_activity.png")
        _save_activity_plot(states, interval, parameter, plot_path)
        onset_plot_path = os.path.join(binary_folder, f"{resolved_output_name}_activity_onsets.png")
        _save_activity_onset_plot(states, interval, parameter, onset_plot_path)
    summary = {
        "warmup_steps": warmup_steps,
        "simulation_steps": total_steps,
        "sample_interval": interval,
        "batch_size": batch_size,
        "seed": seed,
        "mean_rates": mean_rates,
        "samples": rates.shape[0],
        "neurons": states.shape[1] if states.size else network.N,
        "activity_plot": {
            "enabled": bool(binary_cfg.get("plot_activity")),
            "file": os.path.basename(plot_path) if plot_path else None,
            "onsets_file": os.path.basename(onset_plot_path) if onset_plot_path else None,
        },
    }
    summary_path = os.path.join(binary_folder, f"{resolved_output_name}_summary.yaml")
    write_yaml_config(summary, summary_path)
    if names:
        print("Average population activities:")
        for name in names:
            value = mean_rates.get(name, 0.0)
            print(f"  {name}: {value:.4f}")
    else:
        print("No samples recorded. Increase simulation_steps or reduce sample_interval.")
    result = {
        "binary_folder": binary_folder,
        "output_name": resolved_output_name,
        "names": names,
        "mean_rates": mean_rates,
        "trace_path": os.path.join(binary_folder, f"{resolved_output_name}.npz"),
        "summary_path": summary_path,
    }
    del trace
    del state_trace
    del rates
    del states
    del mean_values
    del network
    gc.collect()
    return result


def load_fixpoint_summary(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Fixpoint file {path} did not contain a dictionary payload.")
    if "metadata" not in payload or "fixpoints" not in payload:
        raise ValueError(f"Fixpoint file {path} is missing required keys.")
    if not isinstance(payload["metadata"], dict):
        raise ValueError(f"Fixpoint file {path} contains malformed metadata.")
    return payload


def available_rep_values(fixpoints: Dict[int, Dict[str, Any]]) -> List[float]:
    reps: List[float] = []
    seen = set()
    for entries in fixpoints.values():
        for key in entries.keys():
            rep = _parse_rep_from_key(key)
            if rep is None or rep in seen:
                continue
            seen.add(rep)
            reps.append(rep)
    return sorted(reps)


def focus_rep_grid(fixpoints: Dict[int, Dict[str, Any]]) -> Dict[int, List[float]]:
    grid: Dict[int, List[float]] = {}
    for focus_count, entries in fixpoints.items():
        values = sorted(
            rep
            for rep in (_parse_rep_from_key(key) for key in entries.keys())
            if rep is not None
        )
        grid[int(focus_count)] = values
    return grid


def simulate_fixpoint_reps(
    fixpoint_path: str,
    reps: Sequence[float],
    overrides: argparse.Namespace,
) -> None:
    if not reps:
        raise ValueError("No rep values were provided for the fixpoint batch mode.")
    summary = load_fixpoint_summary(fixpoint_path)
    metadata = summary.get("metadata") or {}
    analysis_parameter = metadata.get("analysis_parameter")
    if not isinstance(analysis_parameter, dict):
        raise ValueError(f"Fixpoint file {fixpoint_path} does not define 'analysis_parameter'.")
    base_parameter: Dict[str, Any] = deepcopy(analysis_parameter)
    fixpoints = summary.get("fixpoints")
    if not isinstance(fixpoints, dict) or not fixpoints:
        raise ValueError(f"Fixpoint file {fixpoint_path} does not list any fixpoints.")
    binary_cfg = _resolve_binary_config(base_parameter, overrides)
    base_output_name = overrides.output_name or binary_cfg.get("output_name", "activity_trace")
    binary_cfg = dict(binary_cfg)
    binary_cfg["output_name"] = base_output_name
    available = available_rep_values(fixpoints)
    missing = [
        rep
        for rep in reps
        if not any(abs(rep - candidate) <= 1e-9 for candidate in available)
    ]
    if missing:
        raise ValueError(
            f"Rep values {missing} are not available in {os.path.basename(fixpoint_path)}. "
            f"Available reps: {available}."
        )
    for rep in reps:
        rep_parameter = deepcopy(base_parameter)
        rep_parameter["R_Eplus"] = float(rep)
        rep_label = _format_rep_label(rep)
        output_name = f"{base_output_name}_rep{rep_label}"
        focus_entries = _extract_fixpoints_for_rep(fixpoints, rep)
        if not focus_entries:
            print(f"Warning: no fixpoints recorded for rep {rep:g} in {fixpoint_path}.")
        print(f"Simulating binary network for rep {rep:g} (output '{output_name}').")
        result = run_binary_simulation(rep_parameter, binary_cfg, output_name=output_name)
        _store_fixpoint_reference(
            result["binary_folder"],
            output_name,
            fixpoint_path,
            rep,
            focus_entries,
            result["mean_rates"],
            result["trace_path"],
            result["summary_path"],
        )


def _parse_rep_from_key(key: str) -> float | None:
    if "_focus" not in key:
        return None
    prefix, _, _ = key.partition("_focus")
    try:
        return float(prefix)
    except ValueError:
        return None


def _extract_fixpoints_for_rep(
    fixpoints: Dict[int, Dict[str, Any]],
    rep: float,
) -> Dict[int, Dict[str, Any]]:
    extracted: Dict[int, Dict[str, Any]] = {}
    for focus_count, entries in fixpoints.items():
        match_key = None
        match_entry = None
        for key, value in entries.items():
            rep_value = _parse_rep_from_key(str(key))
            if rep_value is None:
                continue
            if abs(rep_value - rep) <= 1e-9:
                match_key = key
                match_entry = value
                break
        if match_entry is None:
            continue
        extracted[int(focus_count)] = {
            "key": match_key,
            "fixpoints": _sanitize_fixpoint_entries(match_entry),
        }
    return extracted


def _sanitize_fixpoint_entries(entries: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for point, payload in sorted(entries.items(), key=lambda item: float(item[0])):
        clean_entry = {"value": float(point)}
        for key, value in payload.items():
            clean_entry[str(key)] = _clean_value(value)
        sanitized.append(clean_entry)
    return sanitized


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    return value


def _format_rep_label(rep: float) -> str:
    formatted = f"{rep:.6f}".rstrip("0").rstrip(".")
    if not formatted:
        formatted = "0"
    return formatted.replace("-", "m").replace(".", "_")


def _store_fixpoint_reference(
    binary_folder: str,
    output_name: str,
    fixpoint_path: str,
    rep_value: float,
    focus_entries: Dict[int, Dict[str, Any]],
    mean_rates: Dict[str, float],
    trace_path: str,
    summary_path: str,
) -> None:
    payload = {
        "fixpoints_file": os.path.abspath(fixpoint_path),
        "rep_value": float(rep_value),
        "rep_label": _format_rep_label(rep_value),
        "trace_file": os.path.abspath(trace_path),
        "summary_file": os.path.abspath(summary_path),
        "population_rates": dict(mean_rates),
        "focus_fixpoints": focus_entries,
    }
    output_path = os.path.join(binary_folder, f"{output_name}_fixpoints.yaml")
    write_yaml_config(payload, output_path)
    print(f"Stored fixpoint reference at {output_path}")


def main() -> None:
    args = parse_args()
    if args.fixpoints_file:
        if not args.fixpoint_reps:
            raise ValueError("Provide at least one --fixpoint-reps value when using --fixpoints-file.")
        simulate_fixpoint_reps(args.fixpoints_file, args.fixpoint_reps, args)
        return
    parameter = load_from_args(args)
    binary_cfg = _resolve_binary_config(parameter, args)
    run_binary_simulation(parameter, binary_cfg)


if __name__ == "__main__":
    main()
