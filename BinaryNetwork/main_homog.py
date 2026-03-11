#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

from BinaryNetwork.ClusteredEI_network import ClusteredEI_network


def _progress(iterable: Iterable[int], *, desc: str) -> Iterable[int]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def _build_parameter(
    *,
    probability_mode: bool,
    q_value: int,
    r_j: float,
    connection_type: str,
) -> dict[str, float | int | str]:
    return {
        "N_E": 8000,
        "N_I": 2000,
        "Q": q_value,
        "V_th": 1.0,
        "g": 1.2,
        "p0_ee": 0.3,
        "p0_ei": 0.3,
        "p0_ie": 0.3,
        "p0_ii": 0.3,
        "tau_e": 10.0,
        "tau_i": 5.0,
        "R_Eplus": 8. if probability_mode else 8.,
        "R_j": r_j,
        "kappa": 0.0 if probability_mode else 1.0,
        "connection_type": connection_type,
        "m_X": 0.03,
    }


def _cluster_views(network: ClusteredEI_network) -> list[tuple[int, int]]:
    pops = network.E_pops + network.I_pops
    return [(int(pop.view[0]), int(pop.view[1])) for pop in pops]


def _homogenize_blocks(network: ClusteredEI_network, *, diagonal_only: bool) -> None:
    if network.weight_mode != "dense" or network.weights_dense is None:
        raise RuntimeError("Homogenization requires a dense weight matrix.")
    views = _cluster_views(network)
    for post_idx, (post_start, post_end) in enumerate(views):
        for pre_idx, (pre_start, pre_end) in enumerate(views):
            same_cluster = post_idx == pre_idx
            if diagonal_only != same_cluster:
                continue
            block = network.weights_dense[post_start:post_end, pre_start:pre_end]
            if block.size == 0:
                continue
            mean_value = float(block.mean())
            block[:] = mean_value
    if not diagonal_only:
        np.fill_diagonal(network.weights_dense, 0.0)
    network._recompute_field()


def _simulate_recording(
    network: ClusteredEI_network,
    *,
    warmup_iterations: int,
    samples: int,
    sample_stride: int,
    batch_size: int,
) -> np.ndarray:
    network.configure_cell_type_queue(excitatory_repeat=1, inhibitory_repeat=2)
    for _ in _progress(range(warmup_iterations), desc="Warmup"):
        network.run(sample_stride, batch_size=batch_size)
    recording = np.zeros((network.N, samples), dtype=np.int8)
    for idx in _progress(range(samples), desc="Recording"):
        network.run(sample_stride, batch_size=batch_size)
        recording[:, idx] = network.state
    return recording


def _plot_active_state_raster(ax: plt.Axes, recording: np.ndarray, title: str) -> None:
    neuron_ids, sample_ids = np.nonzero(recording > 0)
    if sample_ids.size:
        ax.scatter(sample_ids, neuron_ids, s=1.0, c="black", marker=".", linewidths=0)
        ax.set_xlim(0, recording.shape[1])
        ax.set_ylim(-0.5, recording.shape[0] - 0.5)
    else:
        ax.text(0.5, 0.5, "No active states recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_yticks([])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Homogenize clustered weights and simulate activity using the ClusteredEI_network API."
    )
    parser.add_argument(
        "--mode",
        choices=("prob", "weight"),
        default="prob",
        help="Use the probability-mixed (kappa=0) or weight-mixed (kappa=1) setup.",
    )
    parser.add_argument("--q", type=int, default=20, help="Number of E/I clusters.")
    parser.add_argument("--r-j", type=float, default=0.8, help="Inhibitory clustering ratio.")
    parser.add_argument(
        "--connection-type",
        choices=("bernoulli", "poisson", "fixed_indegree"),
        default="poisson",
        help="Connectivity sampler for the network.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=500, help="Warmup iterations before recording.")
    parser.add_argument("--samples", type=int, default=2000, help="Recorded binary snapshots.")
    parser.add_argument("--sample-stride", type=int, default=100, help="Update steps between snapshots.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size passed to network.run().")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("BinaryNetwork/main_homog"),
        help="Output prefix for the saved figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probability_mode = args.mode == "prob"
    parameter = _build_parameter(
        probability_mode=probability_mode,
        q_value=args.q,
        r_j=args.r_j,
        connection_type=args.connection_type,
    )

    network = ClusteredEI_network(parameter)
    network.initialize(weight_mode="dense")

    weights_initial = network.weights_dense.copy()
    _homogenize_blocks(network, diagonal_only=True)
    weights_within = network.weights_dense.copy()
    _homogenize_blocks(network, diagonal_only=False)
    weights_between = network.weights_dense.copy()

    recording = _simulate_recording(
        network,
        warmup_iterations=args.warmup_iterations,
        samples=args.samples,
        sample_stride=args.sample_stride,
        batch_size=args.batch_size,
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    cmap = "coolwarm"
    axes[0].imshow(weights_initial, aspect="auto", interpolation="none", cmap=cmap)
    axes[0].set_title("Initial weights")
    axes[1].imshow(weights_within, aspect="auto", interpolation="none", cmap=cmap)
    axes[1].set_title("Within-cluster homogenized")
    axes[2].imshow(weights_between, aspect="auto", interpolation="none", cmap=cmap)
    axes[2].set_title("Fully block-homogenized")
    _plot_active_state_raster(axes[3], recording, f"Active-state raster ({args.mode})")

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
