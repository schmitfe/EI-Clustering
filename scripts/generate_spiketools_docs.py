#!/usr/bin/env python3
"""Generate `spiketools` assets and documentation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PDOC_PYTHON = Path("/home/fschmitt/anaconda3/bin/python")
SPIKETOOLS_ASSETS = ROOT / "docs" / "spiketools_assets"
SPIKETOOLS_RASTER = SPIKETOOLS_ASSETS / "shared_example_raster.png"
GAUSSIAN_KERNEL_FIGURE = SPIKETOOLS_ASSETS / "gaussian_kernel_example.png"
TRIANGULAR_KERNEL_FIGURE = SPIKETOOLS_ASSETS / "triangular_kernel_example.png"
KERNEL_RATE_FIGURE = SPIKETOOLS_ASSETS / "kernel_rate_example.png"
SLIDING_COUNTS_FIGURE = SPIKETOOLS_ASSETS / "sliding_counts_example.png"
RATE_INTEGRAL_FIGURE = SPIKETOOLS_ASSETS / "rate_integral_example.png"
TIME_RESOLVED_CV2_FIGURE = SPIKETOOLS_ASSETS / "time_resolved_cv2_example.png"
COOKBOOK_RATE_TOOLS_FIGURE = SPIKETOOLS_ASSETS / "cookbook_rate_tools_matched_support.png"


def _shared_example_spiketimes():
    sys.path.insert(0, str(ROOT))
    from spiketools.surrogates import gamma_spikes

    np.random.seed(0)
    rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
    orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
    spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)
    return spiketimes, rates, orders


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_spike_raster(ax, spiketimes) -> None:
    valid = np.isfinite(spiketimes[0]) & np.isfinite(spiketimes[1])
    ax.scatter(
        spiketimes[0, valid],
        spiketimes[1, valid],
        marker="|",
        s=90.0,
        linewidths=0.8,
        color="black",
    )
    ax.set_xlim(0.0, 5000.0)
    n_trials = int(np.nanmax(spiketimes[1])) + 1 if spiketimes.shape[1] else 0
    ax.set_ylim(-0.5, n_trials - 0.5)
    tick_step = 1 if n_trials <= 12 else 2
    ax.set_yticks(np.arange(0, n_trials, tick_step))
    ax.set_ylabel("Trial")
    _style_axis(ax)


def generate_shared_spike_raster(output_path: Path) -> None:
    spiketimes, _, _ = _shared_example_spiketimes()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    _plot_spike_raster(ax, spiketimes)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Shared example: 20 gamma-process trials")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _shared_rate_examples():
    sys.path.insert(0, str(ROOT))

    from spiketools.rate import gaussian_kernel, kernel_rate, rate_integral, sliding_counts, triangular_kernel

    spiketimes, _, _ = _shared_example_spiketimes()
    dt = 5.0
    gauss = gaussian_kernel(25.0, dt=dt, nstd=2.0)
    tri = triangular_kernel((25.0 * 2.0) / (np.sqrt(6.0) * dt), dt=dt)
    pooled_rate, rate_time = kernel_rate(spiketimes, gauss, tlim=[0.0, 5000.0], dt=dt, pool=True)
    counts, count_time = sliding_counts(spiketimes, window=250.0, dt=dt, tlim=[0.0, 5000.0])
    integrated = rate_integral(pooled_rate[0], dt=dt)
    return spiketimes, dt, gauss, tri, pooled_rate[0], rate_time, counts.mean(axis=0), count_time, integrated


def _shared_windowing_example():
    sys.path.insert(0, str(ROOT))

    from spiketools.variability import cv2
    from spiketools.windowing import time_resolved

    spiketimes, _, _ = _shared_example_spiketimes()
    values, window_time = time_resolved(
        spiketimes,
        window=1000.0,
        func=cv2,
        kwargs={"pool": True},
        tlim=[0.0, 5000.0],
        tstep=250.0,
    )
    return spiketimes, values, window_time


def generate_kernel_figure(output_path: Path, kernel: np.ndarray, dt: float, title: str, color: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    support = (np.arange(kernel.size) - kernel.size // 2) * dt
    ax.plot(support, kernel, color=color, linewidth=1.8)
    ax.fill_between(support, 0.0, kernel, color=color, alpha=0.15)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_raster_and_trace_figure(
    output_path: Path,
    spiketimes: np.ndarray,
    time: np.ndarray,
    values: np.ndarray,
    trace_title: str,
    ylabel: str,
    color: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )
    _plot_spike_raster(axes[0], spiketimes)
    axes[0].set_title("Input spike train")
    axes[0].tick_params(labelbottom=False)

    axes[1].plot(time, values, color=color, linewidth=1.5)
    axes[1].set_xlim(0.0, 5000.0)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(trace_title)
    _style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_cookbook_rate_figure(
    output_path: Path,
    dt: float,
    gauss: np.ndarray,
    tri: np.ndarray,
    rate_time: np.ndarray,
    pooled_rate: np.ndarray,
    count_time: np.ndarray,
    mean_counts: np.ndarray,
    integrated: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.4), gridspec_kw={"height_ratios": [1.0, 1.2, 1.0, 1.0]})

    support_gauss = (np.arange(gauss.size) - gauss.size // 2) * dt
    support_tri = (np.arange(tri.size) - tri.size // 2) * dt
    axes[0].plot(support_gauss, gauss, color="black", linewidth=1.5, label="Gaussian")
    axes[0].plot(support_tri, tri, color="#8B0000", linewidth=1.5, label="Triangular")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Matched-support smoothing kernels")
    axes[0].legend(frameon=False, loc="upper left")
    _style_axis(axes[0])

    axes[1].plot(rate_time, pooled_rate, color="black", linewidth=1.5)
    axes[1].set_ylabel("Rate (spikes/s)")
    axes[1].set_title("Kernel rate")
    _style_axis(axes[1])

    axes[2].plot(count_time, mean_counts, color="#8B0000", linewidth=1.5)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Mean sliding counts")
    _style_axis(axes[2])

    axes[3].plot(rate_time, integrated, color="#4C7A2A", linewidth=1.5)
    axes[3].set_xlim(0.0, 5000.0)
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_ylabel("Expected spikes")
    axes[3].set_title("Rate integral")
    _style_axis(axes[3])

    for ax in axes[1:]:
        ax.set_xlim(0.0, 5000.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_assets() -> None:
    generate_shared_spike_raster(SPIKETOOLS_RASTER)
    spiketimes, dt, gauss, tri, pooled_rate, rate_time, mean_counts, count_time, integrated = _shared_rate_examples()
    generate_kernel_figure(GAUSSIAN_KERNEL_FIGURE, gauss, dt, "Gaussian kernel", "black")
    generate_kernel_figure(TRIANGULAR_KERNEL_FIGURE, tri, dt, "Triangular kernel", "#8B0000")
    generate_raster_and_trace_figure(
        KERNEL_RATE_FIGURE,
        spiketimes,
        rate_time,
        pooled_rate,
        "Kernel rate estimate",
        "Rate (spikes/s)",
        "black",
    )
    generate_raster_and_trace_figure(
        SLIDING_COUNTS_FIGURE,
        spiketimes,
        count_time,
        mean_counts,
        "Mean sliding counts across trials",
        "Count",
        "#8B0000",
    )
    generate_raster_and_trace_figure(
        RATE_INTEGRAL_FIGURE,
        spiketimes,
        rate_time,
        integrated,
        "Integrated population rate",
        "Expected spikes",
        "#4C7A2A",
    )
    generate_cookbook_rate_figure(
        COOKBOOK_RATE_TOOLS_FIGURE,
        dt,
        gauss,
        tri,
        rate_time,
        pooled_rate,
        count_time,
        mean_counts,
        integrated,
    )
    spiketimes, values, window_time = _shared_windowing_example()
    generate_raster_and_trace_figure(
        TIME_RESOLVED_CV2_FIGURE,
        spiketimes,
        window_time,
        values,
        "Time-resolved CV2",
        "CV2",
        "#355C7D",
    )


def _build_pdoc() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        str(PDOC_PYTHON),
        "-m",
        "pdoc",
        "-d",
        "numpy",
        "--math",
        "-o",
        str(ROOT / "docs"),
        "spiketools",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"Wrote {ROOT / 'docs'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true", help="Generate documentation assets without running pdoc.")
    args = parser.parse_args()

    generate_assets()
    if not args.assets_only:
        _build_pdoc()


if __name__ == "__main__":
    main()
