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


def _save_example_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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


def _documented_grouped_spike_example():
    from plotting import RasterGroup

    spike_times = np.array([5, 8, 11, 13, 21, 23, 29], dtype=float)
    spike_ids = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    groups = [
        RasterGroup("exc_a", ids=range(0, 3), color="#1f77b4", label="Exc A"),
        RasterGroup("exc_b", ids=range(3, 5), color="#2ca02c", label="Exc B"),
        RasterGroup("inh", ids=range(5, 7), color="#8B0000", label="Inh"),
    ]
    return spike_times, spike_ids, groups


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
    fig.subplots_adjust(left=0.12, right=0.84, bottom=0.18, top=0.88)
    _save_example_figure(fig, output_path)


def generate_grouped_spike_raster_example(output_path: Path) -> None:
    from plotting import FontCfg, RasterGroup, RasterLabels, plot_spike_raster, style_axes

    times, ids, groups = _documented_grouped_spike_example()
    groups = [
        RasterGroup(group.name, ids=group.ids, color=group.color, marker=group.marker, size=6.0, label=group.label)
        for group in groups
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    plot_spike_raster(
        ax,
        times,
        ids,
        groups=groups,
        labels=RasterLabels(
            mapping={"exc_a": "Exc A", "exc_b": "Exc B", "inh": "Inh"},
            location="right",
            kwargs={"fontsize": 8},
        ),
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Neuron index")
    ax.set_title("Grouped spike raster example")
    style_axes(ax, FontCfg(base=10.0, scale=1.1).resolve())
    fig.subplots_adjust(left=0.14, right=0.78, bottom=0.22, top=0.82)
    _save_example_figure(fig, output_path)


def generate_binary_raster_example(output_path: Path) -> None:
    from plotting import BinaryStateSource, FontCfg, RasterLabels, plot_binary_raster, style_axes

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
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    plot_binary_raster(
        ax,
        state_source=BinaryStateSource.from_array(states),
        sample_interval=10,
        n_exc=4,
        n_inh=2,
        labels=RasterLabels(mapping={"exc": "Exc", "inh": "Inh"}, kwargs={"fontsize": 9}),
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Neuron index")
    ax.set_title("Binary onset raster example")
    style_axes(ax, FontCfg(base=10.0, scale=1.1).resolve())
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.18, top=0.88)
    _save_example_figure(fig, output_path)


def generate_add_image_ax_example(output_path: Path, image_example_path: Path) -> None:
    from plotting import FontCfg, add_image_ax

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    add_image_ax(ax, str(image_example_path), label="A", fc=FontCfg(base=10.0, scale=1.1).resolve())
    ax.set_title("Image embedding example")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.86)
    _save_example_figure(fig, output_path)


def generate_corner_tag_example(output_path: Path) -> None:
    from plotting import FontCfg, add_corner_tag, style_axes

    fc = FontCfg(base=10.0, scale=1.1).resolve()
    x = np.linspace(0.0, 10.0, 200)
    y = 0.4 + 0.25 * np.sin(x / 1.4)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(x, y, linewidth=2.2, color="#1f77b4")
    ax.set_xlabel("Time [a.u.]")
    ax.set_ylabel("Signal")
    ax.set_title("Corner tag example")
    style_axes(ax, fc)
    add_corner_tag(ax, "Demo", "#333333", fc)
    fig.tight_layout()
    _save_example_figure(fig, output_path)


def generate_panel_label_example(output_path: Path) -> None:
    from plotting import FontCfg, add_panel_label, style_axes

    fc = FontCfg(base=10.0, scale=1.1).resolve()
    x = np.linspace(0.0, 8.0, 160)
    y = 0.2 + 0.6 / (1.0 + np.exp(-1.3 * (x - 3.5)))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(x, y, linewidth=2.2, color="#2ca02c")
    ax.set_xlabel("Driven input [a.u.]")
    ax.set_ylabel("Mean rate")
    ax.set_title("Panel label example")
    style_axes(ax, fc)
    add_panel_label(ax, "A", fc)
    fig.tight_layout()
    _save_example_figure(fig, output_path)


def generate_panel_label_column_example(output_path: Path) -> None:
    from plotting import FontCfg, add_panel_labels_column_left_of_ylabel, style_axes

    fc = FontCfg(base=10.0, scale=1.1).resolve()
    x = np.linspace(0.0, 6.0, 120)
    fig, axs = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True)
    axs[0].plot(x, np.sin(x), color="#1f77b4", linewidth=2.0)
    axs[1].plot(x, np.cos(x), color="#d84ab3", linewidth=2.0)
    axs[0].set_ylabel("Trace A")
    axs[1].set_ylabel("Trace B")
    axs[1].set_xlabel("Time [a.u.]")
    axs[0].set_title("Shared panel-label column")
    for ax in axs:
        style_axes(ax, fc)
    add_panel_labels_column_left_of_ylabel(list(axs), ["A", "B"], fc, y_axes=1.01)
    fig.subplots_adjust(left=0.26, right=0.97, bottom=0.14, top=0.92, hspace=0.14)
    _save_example_figure(fig, output_path)


def generate_listed_colorbar_example(output_path: Path) -> None:
    from plotting import FontCfg, LINE_COLORS, draw_listed_colorbar, style_axes, style_colorbar

    fc = FontCfg(base=10.0, scale=1.1).resolve()
    x = np.linspace(0.0, 12.0, 240)
    fig = plt.figure(figsize=(5.2, 2.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.18], wspace=0.2)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    ax.plot(x, 0.2 + 0.45 / (1.0 + np.exp(-(x - 5.0))), color=LINE_COLORS[0], linewidth=2.2)
    ax.set_xlabel("Driven input [a.u.]")
    ax.set_ylabel("Mean rate")
    ax.set_title("Discrete colorbar example")
    style_axes(ax, fc)
    cbar = draw_listed_colorbar(
        fig,
        cax,
        entries=[(1.0, LINE_COLORS[0]), (2.0, LINE_COLORS[1]), (3.0, LINE_COLORS[2])],
        font_cfg=fc,
        label="Focus",
    )
    style_colorbar(cbar, fc)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.18, top=0.88, wspace=0.2)
    _save_example_figure(fig, output_path)


def generate_style_axes_comparison(output_path: Path) -> None:
    from plotting import FontCfg, style_axes

    fc = FontCfg(base=10.0, scale=1.4).resolve()
    x = np.linspace(0.0, 8.0, 160)
    y = 0.3 + 0.35 * np.sin(x / 1.5)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.8), sharey=True)
    for ax in axs:
        ax.plot(x, y, color="#1f77b4", linewidth=2.2)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Rate")
    axs[0].set_title("Before style_axes")
    axs[1].set_title("After style_axes")
    style_axes(axs[1], fc)
    fig.tight_layout()
    _save_example_figure(fig, output_path)


def generate_style_colorbar_comparison(output_path: Path) -> None:
    from plotting import FontCfg, LINE_COLORS, draw_listed_colorbar, style_colorbar

    fc = FontCfg(base=10.0, scale=1.4).resolve()
    fig = plt.figure(figsize=(8.2, 2.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)
    for idx in range(2):
        grid = outer[0, idx].subgridspec(1, 2, width_ratios=[1.0, 0.18], wspace=0.18)
        ax = fig.add_subplot(grid[0, 0])
        cax = fig.add_subplot(grid[0, 1])
        x = np.linspace(0.0, 10.0, 120)
        ax.plot(x, 0.15 + 0.5 / (1.0 + np.exp(-(x - 4.5))), color=LINE_COLORS[0], linewidth=2.0)
        ax.set_title("Before style_colorbar" if idx == 0 else "After style_colorbar")
        cbar = draw_listed_colorbar(
            fig,
            cax,
            entries=[(1.0, LINE_COLORS[0]), (2.0, LINE_COLORS[1]), (3.0, LINE_COLORS[2])],
            font_cfg=fc,
            label="Focus",
        )
        if idx == 1:
            style_colorbar(cbar, fc)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.86, wspace=0.35)
    _save_example_figure(fig, output_path)


def generate_style_legend_comparison(output_path: Path) -> None:
    from plotting import FontCfg, LINE_COLORS, style_legend

    fc = FontCfg(base=10.0, scale=1.4).resolve()
    x = np.linspace(0.0, 8.0, 160)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.8), sharey=True)
    for idx, ax in enumerate(axs):
        ax.plot(x, np.sin(x / 1.6), color=LINE_COLORS[0], linewidth=2.0, label="Trace A")
        ax.plot(x, np.cos(x / 1.4), color=LINE_COLORS[1], linewidth=2.0, label="Trace B")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Before style_legend" if idx == 0 else "After style_legend")
        if idx == 1:
            style_legend(ax, fc)
    fig.tight_layout()
    _save_example_figure(fig, output_path)


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

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.2))

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

    panel_c_spec = axs[1, 0].get_subplotspec()
    fig.delaxes(axs[1, 0])
    panel_c_grid = panel_c_spec.subgridspec(1, 2, width_ratios=[1.0, 0.14], wspace=0.18)
    ax_lines = fig.add_subplot(panel_c_grid[0, 0])
    cax = fig.add_subplot(panel_c_grid[0, 1])
    for idx, trace in enumerate(traces, start=1):
        ax_lines.plot(time * 180.0, trace, linewidth=2.2, color=LINE_COLORS[idx - 1], label=f"{idx} focus")
    ax_lines.set_xlabel("Driven input [a.u.]")
    ax_lines.set_ylabel("Mean rate")
    ax_lines.set_title("Lines with discrete colorbar")
    ax_lines.set_ylim(0.0, 0.8)
    style_axes(ax_lines, fc)
    ax_lines.legend(loc="upper left", frameon=False)
    style_legend(ax_lines, fc)
    cbar = draw_listed_colorbar(
        fig,
        cax,
        entries=[(1.0, LINE_COLORS[0]), (2.0, LINE_COLORS[1]), (3.0, LINE_COLORS[2])],
        font_cfg=fc,
        label="Focus",
    )
    style_colorbar(cbar, fc)
    add_panel_label(ax_lines, "C", fc)

    add_image_ax(axs[1, 1], str(image_example_path), fc=fc)
    axs[1, 1].set_title("Embedded image via add_image_ax")
    add_panel_label(axs[1, 1], "D", fc)

    fig.tight_layout()
    _save_example_figure(fig, output_path)


def main() -> None:
    assets_dir = ROOT / "docs" / "plotting_assets"
    spike_raster = assets_dir / "spike_raster_example.png"
    grouped_raster = assets_dir / "grouped_spike_raster_example.png"
    binary_raster = assets_dir / "binary_raster_example.png"
    image_ax = assets_dir / "add_image_ax_example.png"
    corner_tag = assets_dir / "add_corner_tag_example.png"
    panel_label = assets_dir / "add_panel_label_example.png"
    panel_label_column = assets_dir / "add_panel_labels_column_left_of_ylabel_example.png"
    listed_colorbar = assets_dir / "draw_listed_colorbar_example.png"
    style_axes = assets_dir / "style_axes_comparison.png"
    style_colorbar = assets_dir / "style_colorbar_comparison.png"
    style_legend = assets_dir / "style_legend_comparison.png"
    showcase = assets_dir / "plotting_showcase.png"

    generate_spike_raster_example(spike_raster)
    generate_grouped_spike_raster_example(grouped_raster)
    generate_binary_raster_example(binary_raster)
    generate_add_image_ax_example(image_ax, grouped_raster)
    generate_corner_tag_example(corner_tag)
    generate_panel_label_example(panel_label)
    generate_panel_label_column_example(panel_label_column)
    generate_listed_colorbar_example(listed_colorbar)
    generate_style_axes_comparison(style_axes)
    generate_style_colorbar_comparison(style_colorbar)
    generate_style_legend_comparison(style_legend)
    generate_plotting_showcase(showcase, grouped_raster)
    print(f"Wrote {assets_dir}")


if __name__ == "__main__":
    main()
