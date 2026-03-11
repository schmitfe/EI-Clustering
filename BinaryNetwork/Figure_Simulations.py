#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

from ClusteredEI_network import ClusteredEI_network


def _progress(iterable: Iterable[int], *, desc: str) -> Iterable[int]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def _build_parameter(
    *,
    q_value: int,
    r_eplus: float,
    r_j: float,
    kappa: float,
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
        "R_Eplus": r_eplus,
        "R_j": r_j,
        "kappa": kappa,
        "connection_type": connection_type,
        "m_X": 0.03,
    }


def _simulate_recording(
    parameter: dict[str, float | int | str],
    *,
    warmup_iterations: int,
    samples: int,
    sample_stride: int,
    batch_size: int,
) -> np.ndarray:
    network = ClusteredEI_network(parameter)
    network.initialize(weight_mode="dense")
    for _ in _progress(range(max(0, warmup_iterations)), desc="Warmup"):
        network.run(sample_stride, batch_size=batch_size)
    recording = np.zeros((network.N, samples), dtype=np.int8)
    for idx in _progress(range(samples), desc="Recording"):
        network.run(sample_stride, batch_size=batch_size)
        recording[:, idx] = network.state
    return recording


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
            block[:] = float(block.mean())
    if not diagonal_only:
        np.fill_diagonal(network.weights_dense, 0.0)
    network._recompute_field()


def _simulate_homogenized_recording(
    parameter: dict[str, float | int | str],
    *,
    warmup_iterations: int,
    samples: int,
    sample_stride: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    network = ClusteredEI_network(parameter)
    network.initialize(weight_mode="dense")
    weights_initial = network.weights_dense.copy()
    _homogenize_blocks(network, diagonal_only=True)
    weights_within = network.weights_dense.copy()
    _homogenize_blocks(network, diagonal_only=False)
    weights_between = network.weights_dense.copy()
    for _ in _progress(range(max(0, warmup_iterations)), desc="Warmup"):
        network.run(sample_stride, batch_size=batch_size)
    recording = np.zeros((network.N, samples), dtype=np.int8)
    for idx in _progress(range(samples), desc="Recording"):
        network.run(sample_stride, batch_size=batch_size)
        recording[:, idx] = network.state
    return weights_initial, weights_within, weights_between, recording


def _plot_active_state_raster(ax: plt.Axes, recording: np.ndarray, title: str) -> None:
    neuron_ids, sample_ids = np.nonzero(recording > 0)
    if sample_ids.size:
        ax.scatter(sample_ids, neuron_ids, s=1.0, c="black", marker=".", linewidths=0)
        ax.set_xlim(0, recording.shape[1])
        ax.set_ylim(-0.5, recording.shape[0] - 0.5)
    else:
        ax.text(0.5, 0.5, "No active states recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_yticks([])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BinaryNetwork test driver.")
    parser.add_argument(
        "--mode",
        choices=("compare", "homog"),
        default="compare",
        help="Run the kappa comparison raster test or the homogenized-weight test.",
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
        "--weight-r-eplus",
        type=float,
        default=8.0,
        help="R_Eplus for the weight-mixed comparison panel.",
    )
    parser.add_argument(
        "--prob-r-eplus",
        type=float,
        default=8.0,
        help="R_Eplus for the probability-mixed comparison panel.",
    )
    parser.add_argument(
        "--homog-mode",
        choices=("prob", "weight"),
        default="prob",
        help="Parameter set used for the homogenization test.",
    )
    parser.add_argument(
        "--homog-r-eplus",
        type=float,
        default=8.0,
        help="R_Eplus for the homogenization test.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("BinaryNetwork/module_test"),
        help="Output prefix for the saved figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        weight_parameter = _build_parameter(
            q_value=args.q,
            r_eplus=args.weight_r_eplus,
            r_j=args.r_j,
            kappa=1.0,
            connection_type=args.connection_type,
        )
        prob_parameter = _build_parameter(
            q_value=args.q,
            r_eplus=args.prob_r_eplus,
            r_j=args.r_j,
            kappa=0.0,
            connection_type=args.connection_type,
        )
        weight_recording = _simulate_recording(
            weight_parameter,
            warmup_iterations=args.warmup_iterations,
            samples=args.samples,
            sample_stride=args.sample_stride,
            batch_size=args.batch_size,
        )
        prob_recording = _simulate_recording(
            prob_parameter,
            warmup_iterations=args.warmup_iterations,
            samples=args.samples,
            sample_stride=args.sample_stride,
            batch_size=args.batch_size,
        )
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        _plot_active_state_raster(axes[0], weight_recording, "Weight-mixed (kappa=1)")
        _plot_active_state_raster(axes[1], prob_recording, "Probability-mixed (kappa=0)")
        axes[1].set_xlabel("Sample index")
    else:
        probability_mode = args.homog_mode == "prob"
        parameter = _build_parameter(
            q_value=args.q,
            r_eplus=args.homog_r_eplus,
            r_j=args.r_j,
            kappa=0.0 if probability_mode else 1.0,
            connection_type=args.connection_type,
        )
        weights_initial, weights_within, weights_between, recording = _simulate_homogenized_recording(
            parameter,
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
        _plot_active_state_raster(axes[3], recording, f"Active-state raster ({args.homog_mode})")
        axes[3].set_xlabel("Sample index")
        for ax in axes[:3]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
