#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

from BinaryNetwork.ClusteredEI_network import ClusteredEI_network  # noqa: E402


def _progress(iterable: Iterable[int], *, desc: str) -> Iterable[int]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def _build_parameter(
    *,
    base_probability: float,
    r_eplus: float,
    r_j: float,
    kappa: float,
    q_value: int,
    connection_type: str,
) -> dict[str, float | int | str]:
    return {
        "N_E": 8000,
        "N_I": 2000,
        "Q": q_value,
        "V_th": 1.0,
        "g": 1.2,
        "p0_ee": base_probability,
        "p0_ei": base_probability,
        "p0_ie": base_probability,
        "p0_ii": base_probability,
        "tau_e": 10.0,
        "tau_i": 5.0,
        "R_Eplus": r_eplus,
        "R_j": r_j,
        "kappa": kappa,
        "connection_type": connection_type,
        "m_X": 0.03,
    }


def _simulate_recording(
    parameter: dict[str, float | int | str],
    *,
    warmup_steps: int,
    samples: int,
    sample_stride: int,
    batch_size: int,
) -> np.ndarray:
    network = ClusteredEI_network(parameter)
    network.initialize(weight_mode="dense")
    network.configure_cell_type_queue(excitatory_repeat=1, inhibitory_repeat=2)

    if warmup_steps > 0:
        for _ in _progress(range(warmup_steps), desc="Warmup"):
            network.run(sample_stride, batch_size=batch_size)

    recording = np.zeros((network.N, samples), dtype=np.int8)
    for idx in _progress(range(samples), desc="Recording"):
        network.run(sample_stride, batch_size=batch_size)
        recording[:, idx] = network.state
    return recording


def _load_spiking_payload(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _plot_binary_panel(ax: plt.Axes, recording: np.ndarray, title: str, ylabel: str) -> None:
    neuron_ids, sample_ids = np.nonzero(recording > 0)
    if sample_ids.size:
        ax.scatter(sample_ids, neuron_ids, s=1.0, c="black", marker=".", linewidths=0)
        ax.set_xlim(0, recording.shape[1])
        ax.set_ylim(-0.5, recording.shape[0] - 0.5)
    else:
        ax.text(0.5, 0.5, "No active states recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_yticks([])


def _plot_spike_panel(ax: plt.Axes, payload: dict | None, title: str, *, show_xlabel: bool) -> None:
    ax.set_title(title)
    if payload is None or "spiketimes" not in payload:
        ax.text(0.5, 0.5, "No spiking data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    spiketimes = np.asarray(payload["spiketimes"])
    if spiketimes.ndim != 2 or spiketimes.shape[0] < 2:
        ax.text(0.5, 0.5, "Invalid spiking payload", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.plot(spiketimes[0] / 1000.0, spiketimes[1], "k.", markersize=0.5)
    ax.set_yticks([])
    if show_xlabel:
        ax.set_xlabel("Time [s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simulation comparison figure using the ClusteredEI_network API."
    )
    parser.add_argument("--samples", type=int, default=12000, help="Recorded binary snapshots per panel.")
    parser.add_argument("--sample-stride", type=int, default=50, help="Update steps between snapshots.")
    parser.add_argument("--warmup-iterations", type=int, default=500, help="Warmup iterations before recording.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size passed to network.run().")
    parser.add_argument("--q", type=int, default=20, help="Number of E/I clusters.")
    parser.add_argument("--r-j", type=float, default=0.8, help="Inhibitory clustering ratio.")
    parser.add_argument("--weight-r-eplus", type=float, default=8.0, help="R_Eplus for the weight-mixed binary panel.")
    parser.add_argument(
        "--prob-r-eplus",
        type=float,
        default=8.,
        help="R_Eplus for the probability-mixed binary panel.",
    )
    parser.add_argument(
        "--connection-type",
        choices=("bernoulli", "poisson", "fixed_indegree"),
        default="poisson",
        help="Connectivity sampler for both binary panels.",
    )
    parser.add_argument(
        "--weight-spiking-data",
        type=Path,
        default=Path("Data/Data_weight.pkl"),
        help="Optional pickle file with a 'spiketimes' entry for the weight-mixed panel.",
    )
    parser.add_argument(
        "--prob-spiking-data",
        type=Path,
        default=Path("Data/Data_prob.pkl"),
        help="Optional pickle file with a 'spiketimes' entry for the probability-mixed panel.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("fig_poster_simulations"),
        help="Output prefix for the saved figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weight_parameter = _build_parameter(
        base_probability=0.3,
        r_eplus=args.weight_r_eplus,
        r_j=args.r_j,
        kappa=1.0,
        q_value=args.q,
        connection_type=args.connection_type,
    )
    prob_parameter = _build_parameter(
        base_probability=0.3,
        r_eplus=args.prob_r_eplus,
        r_j=args.r_j,
        kappa=0.0,
        q_value=args.q,
        connection_type=args.connection_type,
    )

    weight_recording = _simulate_recording(
        weight_parameter,
        warmup_steps=args.warmup_iterations,
        samples=args.samples,
        sample_stride=args.sample_stride,
        batch_size=args.batch_size,
    )
    prob_recording = _simulate_recording(
        prob_parameter,
        warmup_steps=args.warmup_iterations,
        samples=args.samples,
        sample_stride=args.sample_stride,
        batch_size=args.batch_size,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)

    _plot_binary_panel(axes[0, 0], weight_recording, "Binary: weight-mixed (kappa=1)", "weight")
    _plot_binary_panel(axes[1, 0], prob_recording, "Binary: probability-mixed (kappa=0)", "prob.")
    axes[1, 0].set_xlabel("Sample index")

    _plot_spike_panel(
        axes[0, 1],
        _load_spiking_payload(args.weight_spiking_data),
        "LIF reference: weight-mixed",
        show_xlabel=False,
    )
    _plot_spike_panel(
        axes[1, 1],
        _load_spiking_payload(args.prob_spiking_data),
        "LIF reference: probability-mixed",
        show_xlabel=True,
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
