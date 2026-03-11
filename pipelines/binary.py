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
        "--state-chunk-size",
        type=int,
        help="Number of samples per neuron-state chunk written to disk (default: keep all in memory).",
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
    parser.add_argument(
        "--population-rate-init",
        type=float,
        help="Set a uniform population firing-rate initializer (default: 0.1).",
    )
    return parser.parse_args()


DEFAULT_LOG_DECIMATE_FACTOR = 10
DEFAULT_QUEUE_CHUNK_SIZE = 5000
DEFAULT_INHIBITORY_REPEAT = 2


def ensure_binary_behavior_defaults(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Ensure logging/queue defaults match the legacy binary simulation behavior."""
    normalized = dict(cfg or {})
    if "log_step_states" not in normalized:
        normalized["log_step_states"] = True
    else:
        normalized["log_step_states"] = bool(normalized["log_step_states"])
    raw_factor = normalized["log_decimate_factor"] if "log_decimate_factor" in normalized else DEFAULT_LOG_DECIMATE_FACTOR
    if raw_factor in (None, "", 0):
        normalized["log_decimate_factor"] = None
    else:
        normalized["log_decimate_factor"] = max(1, int(raw_factor))
    normalized["log_decimate_zero_phase"] = bool(normalized.get("log_decimate_zero_phase", True))
    normalized["log_decimate_ftype"] = str(normalized.get("log_decimate_ftype", "iir"))
    queue_cfg = dict(normalized.get("update_queue") or {})
    if "enabled" not in queue_cfg:
        queue_cfg["enabled"] = True
    else:
        queue_cfg["enabled"] = bool(queue_cfg["enabled"])
    raw_chunk = queue_cfg.get("chunk_size", DEFAULT_QUEUE_CHUNK_SIZE)
    if raw_chunk in (None, "", 0):
        queue_cfg["chunk_size"] = DEFAULT_QUEUE_CHUNK_SIZE
    else:
        queue_cfg["chunk_size"] = max(1, int(raw_chunk))
    queue_cfg["excitatory_repeat"] = max(1, int(queue_cfg.get("excitatory_repeat", 1) or 1))
    queue_cfg["inhibitory_repeat"] = max(
        1,
        int(queue_cfg.get("inhibitory_repeat", DEFAULT_INHIBITORY_REPEAT) or 1),
    )
    normalized["update_queue"] = queue_cfg
    if "population_rate_init" in normalized:
        entry = normalized["population_rate_init"]
        if entry is None:
            normalized["population_rate_init"] = None
        elif isinstance(entry, (list, tuple, np.ndarray)):
            normalized["population_rate_init"] = np.asarray(entry, dtype=float).tolist()
        else:
            normalized["population_rate_init"] = float(entry)
    else:
        normalized["population_rate_init"] = 0.1
    return normalized


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
    if args.state_chunk_size is not None:
        cfg["state_chunk_size"] = max(0, int(args.state_chunk_size))
    else:
        cfg["state_chunk_size"] = int(cfg.get("state_chunk_size", 0) or 0)
    if args.population_rate_init is not None:
        cfg["population_rate_init"] = float(args.population_rate_init)
    else:
        cfg["population_rate_init"] = cfg.get("population_rate_init", 0.1)
    queue_cfg = cfg.get("update_queue")
    if queue_cfg is not None and not isinstance(queue_cfg, dict):
        raise ValueError("binary.update_queue must be a mapping when provided.")
    return ensure_binary_behavior_defaults(cfg)


def _sample_populations(network: ClusteredEI_network) -> Tuple[List[str], np.ndarray]:
    pops = network.E_pops + network.I_pops
    names = [pop.name for pop in pops]
    values = np.array(
        [float(network.state[pop.view[0]:pop.view[1]].mean()) for pop in pops],
        dtype=float,
    )
    return names, values


def _population_metadata(pops: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_count = len(pops)
    starts = np.zeros(pop_count, dtype=np.int64)
    ends = np.zeros(pop_count, dtype=np.int64)
    sizes = np.zeros(pop_count, dtype=np.float32)
    for idx, pop in enumerate(pops):
        start = int(pop.view[0])
        end = int(pop.view[1])
        starts[idx] = start
        ends[idx] = end
        sizes[idx] = max(1, end - start)
    return starts, ends, sizes


def _population_rates_from_diff_logs(
    initial_state: np.ndarray,
    updates: np.ndarray,
    deltas: np.ndarray,
    pops: Sequence,
    *,
    sample_interval: int,
) -> np.ndarray:
    if not pops:
        return np.zeros((0, 0), dtype=np.float32)
    update_arr = np.asarray(updates, dtype=np.int64)
    delta_arr = np.asarray(deltas, dtype=np.int8)
    if update_arr.ndim == 1:
        update_arr = update_arr[None, :]
    if delta_arr.ndim == 1:
        delta_arr = delta_arr[None, :]
    if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
        return np.zeros((0, len(pops)), dtype=np.float32)
    starts, ends, sizes = _population_metadata(pops)
    state = np.asarray(initial_state, dtype=np.int8).ravel().copy()
    pop_count = len(pops)
    cluster_of = np.empty(state.size, dtype=np.int32)
    cluster_sums = np.zeros(pop_count, dtype=np.int64)
    for idx, (start, end) in enumerate(zip(starts, ends)):
        cluster_of[start:end] = idx
        cluster_sums[idx] = int(state[start:end].sum())
    steps = update_arr.shape[1]
    stride = max(1, int(sample_interval))
    samples: List[np.ndarray] = []
    for step in range(steps):
        units = update_arr[:, step]
        delta_step = delta_arr[:, step]
        for unit, delta in zip(units, delta_step):
            unit_idx = int(unit)
            delta_val = int(delta)
            if delta_val == 0:
                continue
            cluster_idx = int(cluster_of[unit_idx])
            cluster_sums[cluster_idx] += delta_val
            state_value = int(state[unit_idx]) + delta_val
            state[unit_idx] = 1 if state_value > 0 else 0
        if step % stride == 0:
            samples.append((cluster_sums.astype(np.float32, copy=False) / sizes).copy())
    if not samples:
        return np.zeros((0, pop_count), dtype=np.float32)
    return np.stack(samples, axis=0)


def _reconstruct_states_from_diff_logs(
    initial_state: np.ndarray,
    updates: np.ndarray,
    deltas: np.ndarray,
    *,
    sample_interval: int,
) -> np.ndarray:
    update_arr = np.asarray(updates, dtype=np.int64)
    delta_arr = np.asarray(deltas, dtype=np.int8)
    if update_arr.ndim == 1:
        update_arr = update_arr[None, :]
    if delta_arr.ndim == 1:
        delta_arr = delta_arr[None, :]
    if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
        return np.zeros((0, np.asarray(initial_state).size), dtype=np.uint8)
    state = np.asarray(initial_state, dtype=np.int8).ravel().copy()
    stride = max(1, int(sample_interval))
    states: List[np.ndarray] = []
    for step in range(update_arr.shape[1]):
        units = update_arr[:, step]
        delta_step = delta_arr[:, step]
        for unit, delta in zip(units, delta_step):
            delta_val = int(delta)
            if delta_val == 0:
                continue
            unit_idx = int(unit)
            state_value = int(state[unit_idx]) + delta_val
            state[unit_idx] = 1 if state_value > 0 else 0
        if step % stride == 0:
            states.append(state.astype(np.uint8, copy=True))
    if not states:
        return np.zeros((0, state.size), dtype=np.uint8)
    return np.stack(states, axis=0)


def _extract_spike_events(updates: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    update_arr = np.asarray(updates, dtype=np.int64)
    delta_arr = np.asarray(deltas, dtype=np.int8)
    if update_arr.ndim == 1:
        update_arr = update_arr[None, :]
    if delta_arr.ndim == 1:
        delta_arr = delta_arr[None, :]
    if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    mask = delta_arr > 0
    if not mask.any():
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    per_step, steps = update_arr.shape
    times = np.repeat(np.arange(steps, dtype=np.int64), per_step)
    flat_updates = update_arr.reshape(-1, order="F")
    flat_mask = mask.reshape(-1, order="F")
    return times[flat_mask].astype(np.float64, copy=False), flat_updates[flat_mask].astype(np.int64, copy=False)


def _apply_population_rate_initialization(
    network: ClusteredEI_network,
    rates: Sequence[float],
) -> None:
    """
    Set the initial network state by drawing neuron activities according to
    per-population firing probabilities.
    """
    pops = network.E_pops + network.I_pops
    values = np.asarray(rates, dtype=float).ravel()
    if values.size != len(pops):
        raise ValueError(
            f"population_rate_inits must list {len(pops)} entries (one per population), "
            f"got {values.size}."
        )
    for pop, prob in zip(pops, values):
        start, end = int(pop.view[0]), int(pop.view[1])
        prob = float(np.clip(prob, 0.0, 1.0))
        draws = (np.random.random(pop.N) < prob).astype(np.uint8)
        network.state[start:end] = draws
    network._recompute_field()


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


def _erf_parameter_for_tag(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    for key in ("R_Eplus", "focus_count", "focus_counts"):
        filtered.pop(key, None)
    return filtered


def _taggable_binary_config(parameter: Dict[str, Any], binary_cfg: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "parameter": dict(parameter),
        "binary": dict(binary_cfg),
    }
    return payload


def run_binary_simulation(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
    output_name: str | None = None,
    population_rate_inits: Sequence[float] | None = None,
) -> Dict[str, Any]:
    binary_cfg = ensure_binary_behavior_defaults(binary_cfg)
    seed = binary_cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
    network = ClusteredEI_network(parameter)
    network.initialize()
    default_rate = binary_cfg.get("population_rate_init")
    if population_rate_inits is None and default_rate is not None:
        pops = network.E_pops + network.I_pops
        if isinstance(default_rate, (list, tuple, np.ndarray)):
            inferred = np.asarray(default_rate, dtype=float).ravel()
            if inferred.size != len(pops):
                raise ValueError(
                    f"binary.population_rate_init must supply {len(pops)} entries, got {inferred.size}."
                )
        else:
            inferred = np.full(len(pops), float(default_rate), dtype=float)
        population_rate_inits = inferred
    queue_cfg = dict(binary_cfg.get("update_queue") or {})
    if queue_cfg.get("enabled"):
        excit_repeat = int(queue_cfg.get("excitatory_repeat", 1) or 1)
        inhib_repeat = int(queue_cfg.get("inhibitory_repeat", 1) or 1)
        chunk_size = queue_cfg.get("chunk_size")
        chunk_cast = int(chunk_size) if chunk_size not in (None, "") else None
        network.configure_cell_type_queue(
            excitatory_repeat=max(1, excit_repeat),
            inhibitory_repeat=max(1, inhib_repeat),
            chunk_size=chunk_cast,
        )
    else:
        network.configure_update_queue(None)
    if population_rate_inits is not None:
        _apply_population_rate_initialization(network, population_rate_inits)
    warmup_steps = int(binary_cfg["warmup_steps"])
    batch_size = int(binary_cfg["batch_size"])
    if warmup_steps > 0:
        network.run(warmup_steps, batch_size=batch_size)
    interval = int(binary_cfg["sample_interval"])
    if interval <= 0:
        raise ValueError("sample_interval must be positive.")
    total_steps = int(binary_cfg["simulation_steps"])
    if total_steps < 0:
        raise ValueError("simulation_steps must be non-negative.")
    filtered = _erf_parameter_for_tag(parameter)
    tag = sim_tag_from_cfg(filtered)
    folder = ensure_output_folder(parameter, tag=tag)
    params_path = os.path.join(folder, "params.yaml")
    if not os.path.exists(params_path):
        write_yaml_config(filtered, params_path)
    binary_tag = sim_tag_from_cfg(_taggable_binary_config(parameter, binary_cfg))
    binary_folder = os.path.join(folder, "binary", binary_tag)
    os.makedirs(binary_folder, exist_ok=True)
    binary_params_path = os.path.join(binary_folder, "params.yaml")
    if not os.path.exists(binary_params_path):
        write_yaml_config({"parameter": parameter, "binary": binary_cfg}, binary_params_path)
    resolved_output_name = output_name or str(binary_cfg["output_name"])
    if int(binary_cfg.get("state_chunk_size", 0) or 0) > 0:
        print("Ignoring binary.state_chunk_size: binary traces now use legacy-style diff logs only.")
    names: List[str] = []
    pops = network.E_pops + network.I_pops
    pop_count = len(pops)
    names = [pop.name for pop in pops]
    initial_state = network.state.astype(np.uint8, copy=True)
    network.enable_diff_logging(total_steps)
    if total_steps > 0:
        network.run(total_steps, batch_size=batch_size)
    state_updates, state_deltas = network.consume_diff_log()
    if state_updates.shape[1] != total_steps or state_deltas.shape[1] != total_steps:
        raise RuntimeError("Diff logging did not capture the expected number of simulation steps.")
    rates = _population_rates_from_diff_logs(
        initial_state,
        state_updates,
        state_deltas,
        pops,
        sample_interval=interval,
    )
    times = np.arange(rates.shape[0], dtype=np.int64) * interval
    spike_times, spike_ids = _extract_spike_events(state_updates, state_deltas)
    if interval > 1 and state_updates.shape[1] > 0:
        state_updates = state_updates[:, ::interval]
        state_deltas = state_deltas[:, ::interval]
    mean_values = rates.mean(axis=0) if rates.size else np.zeros(pop_count)
    mean_rates = {name: float(value) for name, value in zip(names, mean_values)}
    time_axis = times
    np.savez_compressed(
        os.path.join(binary_folder, f"{resolved_output_name}.npz"),
        rates=rates,
        times=time_axis,
        names=np.array(names),
        state_updates=state_updates,
        state_deltas=state_deltas,
        initial_state=initial_state,
        warmup_steps=warmup_steps,
        simulation_steps=total_steps,
        sample_interval=interval,
        neuron_state_interval=interval,
        batch_size=batch_size,
        seed=seed,
        spike_times=spike_times,
        spike_ids=spike_ids,
    )
    plot_path = None
    onset_plot_path = None
    if binary_cfg.get("plot_activity"):
        states = _reconstruct_states_from_diff_logs(
            initial_state,
            state_updates,
            state_deltas,
            sample_interval=1,
        )
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
        "neurons": network.N,
        "activity_plot": {
            "enabled": bool(binary_cfg.get("plot_activity")),
            "file": os.path.basename(plot_path) if plot_path else None,
            "onsets_file": os.path.basename(onset_plot_path) if onset_plot_path else None,
        },
        "state_chunks": {
            "enabled": False,
            "chunk_size": 0,
            "files": [],
        },
        "step_logging": {
            "enabled": True,
            "format": "diff_log",
            "logged_steps": int(total_steps),
            "decimate_factor": int(interval),
            "decimate_ftype": None,
            "decimate_zero_phase": None,
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
    del rates
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
