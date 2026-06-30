#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from spiketools.rate import gaussian_kernel, kernel_rate


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
DEFAULT_OUTPUT = FIGURES_DIR / "Figure6"
DEFAULT_INPUT_DIR = REPO_ROOT / "BrainScaleS2" / "bs2_cluster_runs"
EXPECTED_PICKLES = (
    "bs2_cluster_kappa_0p00.pkl",
    "bs2_cluster_kappa_0p50.pkl",
    "bs2_cluster_kappa_1p00.pkl",
)
MISSING_RUNS_ERROR = "No saved network runs from BrainScaleS2 found."
RATE_DT_MS = 0.001
KERNEL_SIGMA_MS = 0.1
DISPLAY_DURATION_MS = 6.0


class _MissingDependencyPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _SafeUnpickler(pickle.Unpickler):
    """Load saved run dictionaries without requiring hardware-only classes."""

    def find_class(self, module: str, name: str):
        try:
            __import__(module)
            return getattr(sys.modules[module], name)
        except Exception:
            return _MissingDependencyPlaceholder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 6 from saved BrainScaleS2 cluster runs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing bs2_cluster_kappa_*.pkl files (default: %(default)s).",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=DISPLAY_DURATION_MS,
        help="Time window to show from the start of each run in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--kernel-sigma-ms",
        type=float,
        default=KERNEL_SIGMA_MS,
        help="Gaussian rate-kernel width in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--rate-dt-ms",
        type=float,
        default=RATE_DT_MS,
        help="Time step of the rate-estimation grid in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--neuron-stride",
        type=int,
        default=1,
        help="Plot every Nth neuron in the raster (default: %(default)s).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path prefix for the saved figure (default: %(default)s.{png,pdf}).",
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


def _valid_spikes(spiketimes: np.ndarray) -> np.ndarray:
    spikes = np.asarray(spiketimes, dtype=float)
    if spikes.ndim != 2 or spikes.shape[0] != 2:
        raise ValueError("spiketimes must be a 2xN array.")
    if spikes.shape[1] == 0:
        return spikes
    valid = np.isfinite(spikes[0]) & np.isfinite(spikes[1]) & (spikes[1] >= 0)
    return spikes[:, valid]


def _exc_cluster_spikes(spiketimes: np.ndarray, net_cfg: dict[str, object]) -> tuple[np.ndarray, int, int]:
    n_exc = int(net_cfg.get("N_E", 0))
    n_clusters = int(net_cfg.get("n_clusters", 0))
    if n_exc <= 0 or n_clusters <= 0:
        return np.empty((2, 0), dtype=float), max(n_exc // max(n_clusters, 1), 1), n_clusters
    cluster_size = _cluster_size(n_exc, n_clusters, "excitatory")
    spikes = _valid_spikes(spiketimes)
    if spikes.size == 0:
        return np.empty((2, 0), dtype=float), cluster_size, n_clusters
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
    *,
    duration_ms: float,
    kernel_sigma_ms: float,
    rate_dt_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    exc_spikes, cluster_size, cluster_count = _exc_cluster_spikes(spiketimes, net_cfg)
    exc_spikes = _ensure_trial_entries(exc_spikes, cluster_count)
    if cluster_count <= 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    dt = float(rate_dt_ms)
    if dt <= 0.0:
        raise ValueError(f"Rate-estimation dt must be positive (got {dt} ms).")
    if kernel_sigma_ms <= 0.0:
        raise ValueError(f"Kernel sigma must be positive (got {kernel_sigma_ms} ms).")
    kernel = gaussian_kernel(float(kernel_sigma_ms), dt=dt)
    pad_steps = len(kernel) // 2
    pad_time = pad_steps * dt
    rates, rate_time = kernel_rate(
        exc_spikes,
        kernel,
        tlim=(-pad_time, float(duration_ms) + pad_time),
        dt=dt,
        pool=False,
    )
    if rates.size == 0:
        return rate_time, rates
    mask = (rate_time >= 0.0) & (rate_time <= float(duration_ms))
    if not np.any(mask):
        return np.zeros(0, dtype=float), np.zeros((rates.shape[0], 0), dtype=float)
    rate_time = rate_time[mask]
    rates = rates[:, mask]
    return rate_time, rates / (1000.0 * float(cluster_size))


def plot_raster(
    ax: plt.Axes,
    spiketimes: np.ndarray,
    net_cfg: dict[str, object],
    font_cfg: FontCfg,
    *,
    duration_ms: float,
    neuron_stride: int,
) -> None:
    spikes = _valid_spikes(spiketimes)
    n_exc = int(net_cfg.get("N_E", 0))
    n_inh = int(net_cfg.get("N_I", 0))
    if spikes.size == 0:
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
        spikes[0],
        spikes[1].astype(int),
        n_exc=n_exc,
        n_inh=n_inh,
        stride=max(1, int(neuron_stride)),
        t_start=0.0,
        t_end=float(duration_ms),
        labels=labels,
        marker_size=1.5,
    )
    ax.set_xlim(0.0, float(duration_ms))
    ax.set_ylim(-1, n_exc + n_inh)
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
    *,
    duration_ms: float,
    ylabel: str | None,
) -> None:
    if rates.size == 0 or rate_time.size == 0:
        ax.text(0.5, 0.5, "No excitatory spikes", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    cmap = plt.get_cmap("Greys")
    shades = [0.6] if rates.shape[0] <= 1 else np.linspace(0.25, 0.85, rates.shape[0])
    for trace, shade in zip(rates, shades):
        ax.plot(rate_time, trace, color=cmap(shade), linewidth=1.2)
    ax.set_xlim(0.0, float(duration_ms))
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(ylabel if ylabel else "")
    ax.set_xlabel("Time [ms]")


def _panel_labels(count: int) -> Sequence[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [alphabet[idx % len(alphabet)] for idx in range(count)]


def _load_pickle(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        data = _SafeUnpickler(handle).load()
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a run dictionary.")
    return data


def load_runs(input_dir: Path) -> list[dict[str, object]]:
    missing = [name for name in EXPECTED_PICKLES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(MISSING_RUNS_ERROR)
    runs = []
    for name in EXPECTED_PICKLES:
        path = input_dir / name
        data = _load_pickle(path)
        params = dict(data.get("params") or data.get("net_dict") or {})
        net_cfg = dict(data.get("net_dict") or params)
        sim_cfg = dict(data.get("sim_dict") or data.get("params") or {})
        entry = {
            "path": path,
            "kappa": float(data["kappa"]),
            "spiketimes": np.asarray(data["spiketimes"], dtype=float),
            "net_dict": net_cfg,
            "sim_dict": sim_cfg,
        }
        runs.append(entry)
    return sorted(runs, key=lambda run: float(run["kappa"]))


def main() -> None:
    args = parse_args()
    runs = load_runs(Path(args.input_dir))
    if len(runs) != 3:
        raise ValueError("Figure 6 expects exactly three BrainScaleS2 kappa runs.")

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

    rate_axes = []
    column_labels = _panel_labels(len(runs))
    for col_idx, run in enumerate(runs):
        raster_ax = fig.add_subplot(grid[0, col_idx])
        rate_ax = fig.add_subplot(grid[2, col_idx], sharex=raster_ax)
        rate_axes.append(rate_ax)

        plot_raster(
            raster_ax,
            run["spiketimes"],
            run["net_dict"],
            font_cfg,
            duration_ms=args.duration_ms,
            neuron_stride=args.neuron_stride,
        )
        rate_time, cluster_rates = compute_cluster_rates(
            run["spiketimes"],
            run["net_dict"],
            duration_ms=args.duration_ms,
            kernel_sigma_ms=args.kernel_sigma_ms,
            rate_dt_ms=args.rate_dt_ms,
        )
        ylabel = r"$\lambda_C$ [Spikes/ms]" if col_idx == 0 else None
        plot_rate_traces(rate_ax, rate_time, cluster_rates, duration_ms=args.duration_ms, ylabel=ylabel)

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
