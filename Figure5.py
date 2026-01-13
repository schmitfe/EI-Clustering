#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from plotting import (
    FontCfg,
    RasterLabels,
    add_panel_label,
    plot_spike_raster,
    style_axes,
)
from sim_config import add_override_arguments, load_from_args, parse_overrides, deep_update
from spiketools.rate import gaussian_kernel, kernel_rate
from spiking_pipeline import run_spiking_simulation


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
DEFAULT_OUTPUT = FIGURES_DIR / "Figure5"
KERNEL_SIGMA_MS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 5 with spiking-network rasters and rates.")
    add_override_arguments(parser)
    parser.add_argument(
        "--kappa-values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="List of kappa values to simulate (default: %(default)s).",
    )
    parser.add_argument(
        "--neuron-stride",
        type=int,
        default=5,
        help="Plot every Nth neuron in the raster (default: %(default)s).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path prefix for the saved figure (default: %(default)s.{png,pdf}).",
    )
    parser.add_argument(
        "--column-override",
        action="append",
        default=[],
        metavar="index:path=value",
        help="Apply overrides to a specific column (0-based index).",
    )
    return parser.parse_args()


def _cluster_size(neurons: int, clusters: int, label: str) -> int:
    if clusters <= 0:
        raise ValueError(f"Number of {label} clusters must be positive (got {clusters}).")
    if neurons % clusters != 0:
        raise ValueError(f"{label.title()} neuron count {neurons} must be divisible by {clusters} clusters.")
    return neurons // clusters


def _ensure_trial_entries(spikes: np.ndarray, total_trials: int) -> np.ndarray:
    if total_trials <= 0:
        return spikes
    if spikes.size == 0:
        missing = np.arange(total_trials, dtype=float)
    else:
        present = np.unique(spikes[1].astype(int))
        missing = np.setdiff1d(np.arange(total_trials, dtype=int), present)
    if missing.size == 0:
        return spikes
    extra = np.vstack([np.full(missing.size, np.nan), missing.astype(float)])
    return np.concatenate([spikes, extra], axis=1) if spikes.size else extra


def _exc_cluster_spikes(spiketimes: np.ndarray, net_cfg: dict[str, object]) -> tuple[np.ndarray, int, int]:
    n_exc = int(net_cfg.get("N_E", 0))
    n_clusters = int(net_cfg.get("n_clusters", 0))
    if n_exc <= 0 or n_clusters <= 0:
        return np.empty((2, 0), dtype=float), max(n_exc // max(n_clusters, 1), 1), n_clusters
    cluster_size = _cluster_size(n_exc, n_clusters, "excitatory")
    spikes = np.asarray(spiketimes, dtype=float)
    if spikes.ndim != 2 or spikes.shape[0] != 2:
        raise ValueError("spiketimes must be a 2xN array.")
    ids = spikes[1].astype(int)
    mask = ids < n_exc
    if not np.any(mask):
        return np.empty((2, 0), dtype=float), cluster_size, n_clusters
    cluster_ids = (ids[mask] // cluster_size).astype(float)
    exc_spikes = np.vstack([spikes[0, mask], cluster_ids])
    return exc_spikes, cluster_size, n_clusters


def compute_cluster_rates(
    spiketimes: np.ndarray,
    net_cfg: dict[str, object],
    sim_cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    exc_spikes, cluster_size, cluster_count = _exc_cluster_spikes(spiketimes, net_cfg)
    exc_spikes = _ensure_trial_entries(exc_spikes, cluster_count)
    if cluster_count <= 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    dt = float(sim_cfg.get("dt", 0.1))
    simtime = float(sim_cfg.get("simtime", exc_spikes[0].max() if exc_spikes.size else dt))
    kernel = gaussian_kernel(KERNEL_SIGMA_MS, dt=dt)
    pad_steps = len(kernel) // 2
    pad_time = pad_steps * dt
    rates, rate_time = kernel_rate(
        exc_spikes,
        kernel,
        tlim=(-pad_time, simtime + pad_time),
        dt=dt,
        pool=False,
    )
    if rates.size == 0:
        return rate_time, rates
    mask = (rate_time >= 0.0) & (rate_time <= simtime)
    if not np.any(mask):
        return np.zeros(0, dtype=float), np.zeros((rates.shape[0], 0), dtype=float)
    rate_time = rate_time[mask]
    rates = rates[:, mask]
    return rate_time, rates / float(cluster_size)


def plot_raster(
    ax: plt.Axes,
    spiketimes: np.ndarray,
    net_cfg: dict[str, object],
    sim_cfg: dict[str, object],
    font_cfg: FontCfg,
    neuron_stride: int,
) -> None:
    times = np.asarray(spiketimes[0], dtype=float)
    ids = np.asarray(spiketimes[1], dtype=int)
    simtime = float(sim_cfg.get("simtime", times.max() if times.size else 0.0))
    n_exc = int(net_cfg.get("N_E", 0))
    n_inh = int(net_cfg.get("N_I", 0))
    if times.size == 0 or ids.size == 0:
        ax.text(0.5, 0.5, "No spikes recorded", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    labels = RasterLabels(
        show=True,
        excitatory="Exc.",
        inhibitory="Inh.",
        location="left",
        kwargs={
            "fontsize": font_cfg.tick,
            "rotation": 90,
            "ha": "right",
            "va": "center",
        },
    )
    plot_spike_raster(
        ax,
        times,
        ids,
        n_exc=n_exc,
        n_inh=n_inh,
        stride=max(1, int(neuron_stride)),
        t_start=0.0,
        t_end=simtime,
        labels=labels,
        marker_size=1.5
    )
    ax.set_xlim(0.0, simtime)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for text in ax.texts:
        content = text.get_text().strip().lower()
        if content.startswith("exc"):
            text.set_color("black")
        elif content.startswith("inh"):
            text.set_color("#8B0000")


def plot_rate_traces(
    ax: plt.Axes,
    rate_time: np.ndarray,
    rates: np.ndarray,
    sim_cfg: dict[str, object],
    ylabel: str | None,
) -> None:
    if rates.size == 0 or rate_time.size == 0:
        ax.text(0.5, 0.5, "No excitatory spikes", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    simtime = float(sim_cfg.get("simtime", rate_time.max()))
    cmap = plt.get_cmap("Greys")
    if rates.shape[0] <= 1:
        shades = [0.6]
    else:
        shades = np.linspace(0.25, 0.85, rates.shape[0])
    for trace, shade in zip(rates, shades):
        ax.plot(rate_time, trace, color=cmap(shade), linewidth=1.2)
    ax.set_xlim(0.0, simtime)
    ax.set_ylim(bottom=0.0)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Time [ms]")


def _panel_labels(count: int) -> Sequence[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels = []
    for idx in range(count):
        letter = alphabet[idx % len(alphabet)]
        labels.append(f"{letter}")
    return labels


def _parse_column_overrides(entries: Sequence[str], column_count: int) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Column override '{raw}' must use 'index:path=value' format.")
        prefix, payload = raw.split(":", 1)
        try:
            column_index = int(prefix.strip())
        except ValueError as exc:
            raise ValueError(f"Column override '{raw}' has an invalid numeric index.") from exc
        if column_index < 0 or column_index >= column_count:
            raise ValueError(f"Column index {column_index} is outside the valid range [0, {column_count - 1}].")
        mapping.setdefault(column_index, []).append(payload.strip())
    return mapping


def simulate_runs(
    base_parameter: dict[str, object],
    kappas: Sequence[float],
    column_overrides: dict[int, list[str]],
) -> list[dict[str, object]]:
    runs = []
    for idx, value in enumerate(kappas):
        parameter = deepcopy(base_parameter)
        if column_overrides.get(idx):
            override_dict = parse_overrides(column_overrides[idx])
            parameter = deep_update(parameter, override_dict)
        parameter["kappa"] = float(value)
        result = run_spiking_simulation(parameter)
        entry = dict(result)
        entry["kappa"] = float(value)
        entry["spiketimes"] = np.asarray(result["spiketimes"], dtype=float)
        runs.append(entry)
    return runs


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    kappas = [float(value) for value in args.kappa_values]
    if len(kappas) != 3:
        raise ValueError("Figure 5 expects exactly three kappa values to populate the 3-column layout.")
    column_overrides = (
        _parse_column_overrides(args.column_override, len(kappas)) if args.column_override else {}
    )
    runs = simulate_runs(base_parameter, kappas, column_overrides)
    font_cfg = FontCfg(base=12.0, scale=1.3).resolve()
    fig = plt.figure(figsize=(13.0, 6.0))
    grid = fig.add_gridspec(
        3,
        len(runs),
        height_ratios=[0.65, 0.02, 0.33],
        hspace=0.05,
        wspace=0.15,
        left=0.055,
        right=0.98,
        top=0.94,
        bottom=0.12,
    )
    raster_axes = []
    rate_axes = []
    column_labels = _panel_labels(len(runs))
    for col_idx, run in enumerate(runs):
        raster_ax = fig.add_subplot(grid[0, col_idx])
        rate_ax = fig.add_subplot(grid[2, col_idx], sharex=raster_ax)
        raster_axes.append(raster_ax)
        rate_axes.append(rate_ax)
        plot_raster(
            raster_ax,
            run["spiketimes"],
            run["net_dict"],
            run["sim_dict"],
            font_cfg,
            neuron_stride=args.neuron_stride,
        )
        rate_time, cluster_rates = compute_cluster_rates(run["spiketimes"], run["net_dict"], run["sim_dict"])
        ylabel = r"$\bar{\lambda}_C$ [Spikes/s]" if col_idx == 0 else None
        plot_rate_traces(rate_ax, rate_time, cluster_rates, run["sim_dict"], ylabel)
        style_axes(raster_ax, font_cfg, set_xlabel=False, set_ylabel=False)
        style_axes(rate_ax, font_cfg)
        raster_ax.set_title(fr"$\kappa={run['kappa']:.2g}$", fontsize=font_cfg.title)
        add_panel_label(raster_ax, f"{column_labels[col_idx]}1", font_cfg, x=-0.11, y=1.03)
        add_panel_label(rate_ax, f"{column_labels[col_idx]}2", font_cfg, x=-0.11, y=1.05)
    if rate_axes:
        global_min = min(ax.get_ylim()[0] for ax in rate_axes)
        global_max = max(ax.get_ylim()[1] for ax in rate_axes)
        for ax in rate_axes:
            ax.set_ylim(global_min, global_max)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", dpi=600)
    fig.savefig(f"{output_prefix}.pdf", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
