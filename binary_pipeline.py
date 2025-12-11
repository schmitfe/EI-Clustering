from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from BinaryNetwork.ClusteredEI_network import ClusteredEI_network

from rate_system import ensure_output_folder
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    binary_cfg = _resolve_binary_config(parameter, args)
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
    means = rates.mean(axis=0) if rates.size else np.zeros(pop_count)
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
    output_name = str(binary_cfg["output_name"])
    np.savez_compressed(
        os.path.join(binary_folder, f"{output_name}.npz"),
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
    if binary_cfg.get("plot_activity"):
        plot_path = os.path.join(binary_folder, f"{output_name}_activity.png")
        _save_activity_plot(states, interval, parameter, plot_path)
    summary = {
        "warmup_steps": warmup_steps,
        "simulation_steps": total_steps,
        "sample_interval": interval,
        "batch_size": batch_size,
        "seed": seed,
        "mean_rates": {name: float(value) for name, value in zip(names, means)},
        "samples": rates.shape[0],
        "neurons": states.shape[1] if states.size else network.N,
        "activity_plot": {
            "enabled": bool(binary_cfg.get("plot_activity")),
            "file": os.path.basename(plot_path) if plot_path else None,
        },
    }
    write_yaml_config(summary, os.path.join(binary_folder, f"{output_name}_summary.yaml"))
    if names:
        print("Average population activities:")
        for name, value in zip(names, means):
            print(f"  {name}: {value:.4f}")
    else:
        print("No samples recorded. Increase simulation_steps or reduce sample_interval.")


if __name__ == "__main__":
    main()
