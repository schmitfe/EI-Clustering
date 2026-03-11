#!/usr/bin/env python3
"""Generate example assets for the plotting package documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _synthetic_spike_trains() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    spike_times: list[np.ndarray] = []
    spike_ids: list[np.ndarray] = []
    group_specs = [
        (0, 6, 5, (0.0, 180.0)),
        (6, 12, 4, (10.0, 180.0)),
        (12, 16, 3, (20.0, 180.0)),
    ]
    for start_id, stop_id, mean_count, window in group_specs:
        for neuron_id in range(start_id, stop_id):
            count = int(rng.poisson(mean_count))
            times = np.sort(rng.uniform(window[0], window[1], size=count))
            spike_times.append(times)
            spike_ids.append(np.full(times.size, neuron_id, dtype=int))
    times = np.concatenate(spike_times) if spike_times else np.zeros(0, dtype=float)
    ids = np.concatenate(spike_ids) if spike_ids else np.zeros(0, dtype=int)
    order = np.argsort(times, kind="mergesort")
    return times[order], ids[order]


def generate_spike_raster_example(output_path: Path) -> None:
    from plotting import FontCfg, RasterLabels, plot_spike_raster, style_axes

    times, ids = _synthetic_spike_trains()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    plot_spike_raster(
        ax,
        times,
        ids,
        n_exc=12,
        n_inh=4,
        marker_size=2.2,
        labels=RasterLabels(
            excitatory="Excitatory",
            inhibitory="Inhibitory",
            kwargs={"fontsize": 10},
        ),
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Neuron index")
    ax.set_title("Spike raster example")
    style_axes(ax, FontCfg(base=10.0, scale=1.1).resolve())
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_grouped_spike_raster_example(output_path: Path) -> None:
    from plotting import FontCfg, RasterGroup, RasterLabels, plot_spike_raster, style_axes

    times, ids = _synthetic_spike_trains()
    groups = [
        RasterGroup("exc_a", ids=range(0, 6), color="#1f77b4", size=3.0, label="Exc A"),
        RasterGroup("exc_b", ids=range(6, 12), color="#2ca02c", size=3.0, label="Exc B"),
        RasterGroup("inh", ids=range(12, 16), color="#8B0000", marker="s", size=4.0, label="Inh"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    plot_spike_raster(
        ax,
        times,
        ids,
        groups=groups,
        labels=RasterLabels(
            mapping={"exc_a": "Exc A", "exc_b": "Exc B", "inh": "Inh"},
            location="right",
            kwargs={"fontsize": 9},
        ),
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Neuron index")
    ax.set_title("Grouped spike raster with RasterGroup / RasterLabels")
    style_axes(ax, FontCfg(base=10.0, scale=1.1).resolve())
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_plotting_showcase(output_path: Path, image_example_path: Path) -> None:
    from plotting import (
        LINE_COLORS,
        BinaryStateSource,
        FontCfg,
        RasterGroup,
        RasterLabels,
        add_corner_tag,
        add_image_ax,
        add_panel_label,
        draw_listed_colorbar,
        plot_binary_raster,
        plot_spike_raster,
        style_axes,
        style_colorbar,
        style_legend,
    )

    fc = FontCfg(base=10.0, scale=1.05).resolve()
    times, ids = _synthetic_spike_trains()
    groups = [
        RasterGroup("exc_a", ids=range(0, 6), color=LINE_COLORS[0], size=3.0, label="Exc A"),
        RasterGroup("exc_b", ids=range(6, 12), color=LINE_COLORS[1], size=3.0, label="Exc B"),
        RasterGroup("inh", ids=range(12, 16), color="#8B0000", marker="s", size=4.0, label="Inh"),
    ]

    states = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    source = BinaryStateSource.from_array(states)

    time = np.linspace(0.0, 1.0, 150)
    traces = [
        0.12 + 0.55 / (1.0 + np.exp(-10.0 * (time - 0.32))),
        0.08 + 0.48 / (1.0 + np.exp(-8.0 * (time - 0.55))),
        0.05 + 0.30 / (1.0 + np.exp(-7.0 * (time - 0.68))),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.2))

    plot_spike_raster(
        axs[0, 0],
        times,
        ids,
        groups=groups,
        labels=RasterLabels(location="right", kwargs={"fontsize": 8}),
    )
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Neuron index")
    axs[0, 0].set_title("Grouped spike raster")
    add_panel_label(axs[0, 0], "A", fc)
    style_axes(axs[0, 0], fc)

    event_times, event_ids = plot_binary_raster(
        axs[0, 1],
        state_source=source,
        sample_interval=10,
        n_exc=4,
        n_inh=2,
        labels=RasterLabels(
            mapping={"exc": "Exc", "inh": "Inh"},
            location="right",
            kwargs={"fontsize": 8},
        ),
    )
    axs[0, 1].set_xlabel("Time [ms]")
    axs[0, 1].set_ylabel("Neuron index")
    axs[0, 1].set_title("Binary onset raster")
    add_corner_tag(axs[0, 1], f"{event_times.size} events", "#333333", fc)
    add_panel_label(axs[0, 1], "B", fc)
    style_axes(axs[0, 1], fc)

    for idx, trace in enumerate(traces, start=1):
        axs[1, 0].plot(time * 180.0, trace, linewidth=2.2, color=LINE_COLORS[idx - 1], label=f"{idx} focus")
    axs[1, 0].set_xlabel("Driven input [a.u.]")
    axs[1, 0].set_ylabel("Mean rate")
    axs[1, 0].set_title("Lines with discrete colorbar")
    axs[1, 0].set_ylim(0.0, 0.8)
    style_axes(axs[1, 0], fc)
    axs[1, 0].legend(loc="upper left", frameon=False)
    style_legend(axs[1, 0], fc)
    cax = axs[1, 0].inset_axes([0.78, 0.18, 0.18, 0.64])
    cbar = draw_listed_colorbar(
        fig,
        cax,
        entries=[(1.0, LINE_COLORS[0]), (2.0, LINE_COLORS[1]), (3.0, LINE_COLORS[2])],
        font_cfg=fc,
        label="Focus",
    )
    style_colorbar(cbar, fc)
    add_panel_label(axs[1, 0], "C", fc)

    add_image_ax(axs[1, 1], str(image_example_path), fc=fc)
    axs[1, 1].set_title("Embedded image via add_image_ax")
    add_panel_label(axs[1, 1], "D", fc)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    assets_dir = ROOT / "docs" / "plotting_assets"
    spike_raster = assets_dir / "spike_raster_example.png"
    grouped_raster = assets_dir / "grouped_spike_raster_example.png"
    showcase = assets_dir / "plotting_showcase.png"

    generate_spike_raster_example(spike_raster)
    generate_grouped_spike_raster_example(grouped_raster)
    generate_plotting_showcase(showcase, grouped_raster)
    print(f"Wrote {assets_dir}")


if __name__ == "__main__":
    main()
