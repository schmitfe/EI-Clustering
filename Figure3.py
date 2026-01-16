#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import figure_helpers as helpers  # noqa: E402
from plotting import (
    BinaryStateSource,
    FontCfg,
    RasterLabels,
    add_panel_label,
    draw_listed_colorbar,
    plot_binary_raster,
    plot_spike_raster,
    style_axes,
    _time_axis_scale,
    _prepare_line_color_map,
)  # noqa: E402
from sim_config import add_override_arguments, load_from_args  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 3 by running the EI pipeline, performing legacy binary simulations, "
            "and visualizing raster/rate traces alongside the maximum-rate distribution."
        )
    )
    add_override_arguments(parser)
    parser.add_argument("--r-eplus", type=float, action="append", help="Explicit R_Eplus values to analyze.")
    parser.add_argument("--r-eplus-start", type=float, help="Start of an R_Eplus sweep (inclusive).")
    parser.add_argument("--r-eplus-end", type=float, help="End of an R_Eplus sweep (inclusive).")
    parser.add_argument("--r-eplus-step", type=float, help="Step size for the R_Eplus sweep.")
    parser.add_argument("--v-start", type=float, default=0.0, help="ERF sweep start value (default: %(default)s).")
    parser.add_argument("--v-end", type=float, default=1.0, help="ERF sweep end value (default: %(default)s).")
    parser.add_argument("--v-steps", type=int, default=1000, help="ERF samples per sweep (default: %(default)s).")
    parser.add_argument("--retry-step", type=float, help="Optional retry increment for solver restarts.")
    parser.add_argument(
        "--erf-jobs",
        type=int,
        default=1,
        help="Number of workers for the ERF stage (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite-erf",
        action="store_true",
        help="Re-run the ERF stage even if matching files exist.",
    )
    parser.add_argument(
        "--focus-counts",
        type=int,
        nargs="+",
        help="Focus counts to include (default: all values from 1..Q).",
    )
    parser.add_argument(
        "--stability-filter",
        choices=("stable", "unstable", "any"),
        default="stable",
        help="Select fixpoints of the given stability (default: %(default)s).",
    )
    parser.add_argument("--simulations", type=int, default=20, help="Number of seeds to simulate (default: %(default)s).")
    parser.add_argument("--bin-size", type=int, default=50, help="Samples per bin when computing maxima (default: %(default)s).")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count (default: %(default)s).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for the legacy simulations (default: %(default)s).",
    )
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size.")
    parser.add_argument("--seed", type=int, help="Base seed for the legacy binary simulations.")
    parser.add_argument("--output-name", type=str, help="Custom prefix for saved legacy traces.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new legacy simulations and reuse existing traces.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if matching traces exist.",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="Recompute maxima even if cached files exist.",
    )
    parser.add_argument(
        "--raster-duration",
        type=float,
        help="Restrict the raster plot to this duration (same units as the trace time axis).",
    )
    parser.add_argument(
        "--rates-duration",
        type=float,
        help="Restrict the rate plot to this duration (same units as the trace time axis).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="Figures/Figure3",
        help="Prefix for the saved figure files (default: %(default)s.{png,pdf}).",
    )
    return parser.parse_args()


def _resolve_r_eplus_list(args: argparse.Namespace, parameter: Dict[str, Any]) -> List[float]:
    if args.r_eplus:
        return [float(val) for val in args.r_eplus]
    if (
        args.r_eplus_start is not None
        and args.r_eplus_end is not None
        and args.r_eplus_step is not None
    ):
        start = float(args.r_eplus_start)
        end = float(args.r_eplus_end)
        step = float(args.r_eplus_step)
        if step <= 0:
            raise ValueError("--r-eplus-step must be positive.")
        count = int(np.floor((end - start) / step)) + 1
        values = np.round(np.linspace(start, start + step * (count - 1), count), decimals=6)
        valid = [float(val) for val in values if (step > 0 and val <= end + 1e-12)]
        if not valid:
            raise ValueError("R_Eplus sweep produced no valid values.")
        return valid
    base = parameter.get("R_Eplus")
    if base is None:
        raise ValueError("Provide at least one R_Eplus value via the config or --r-eplus.")
    return [float(base)]


def _load_trace_payload(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace {path} does not exist.")
    with np.load(path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
        times = np.asarray(data.get("times"), dtype=float) if "times" in data else np.arange(rates.shape[0])
        states = np.asarray(data.get("neuron_states"), dtype=np.uint8) if "neuron_states" in data else np.zeros((0, 0), dtype=np.uint8)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_updates = np.asarray(data["state_updates"], dtype=np.uint16) if "state_updates" in data else None
        state_deltas = np.asarray(data["state_deltas"], dtype=np.int8) if "state_deltas" in data else None
        initial_state = np.asarray(data["initial_state"], dtype=np.uint8) if "initial_state" in data else None
        spike_times = np.asarray(data["spike_times"], dtype=float).ravel() if "spike_times" in data else np.zeros(0, dtype=float)
        spike_ids = np.asarray(data["spike_ids"], dtype=np.int64).ravel() if "spike_ids" in data else np.zeros(0, dtype=np.int64)
        spike_trials = np.asarray(data["spike_trials"], dtype=np.int16).ravel() if "spike_trials" in data else np.zeros(0, dtype=np.int16)
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "state_updates": state_updates,
        "state_deltas": state_deltas,
        "initial_state": initial_state,
        "sample_interval": sample_interval,
        "spike_times": spike_times,
        "spike_ids": spike_ids,
        "spike_trials": spike_trials,
    }


@dataclass(frozen=True)
class FocusMarker:
    focus: int
    value: float
    stable: bool


def _collect_focus_markers(focus_rates: Dict[int, Dict[str, List[float]]]) -> List[FocusMarker]:
    markers: List[FocusMarker] = []
    for focus in sorted(int(key) for key in focus_rates.keys()):
        payload = focus_rates.get(int(focus), {})
        stable_values = [float(val) for val in (payload.get("stable") or []) if np.isfinite(val)]
        unstable_values = [float(val) for val in (payload.get("unstable") or []) if np.isfinite(val)]
        if stable_values:
            markers.append(FocusMarker(focus=int(focus), value=max(stable_values), stable=True))
        if unstable_values:
            markers.append(FocusMarker(focus=int(focus), value=max(unstable_values), stable=False))
    return markers


def _format_time_ticks(start: float, end: float, count: int = 4) -> Tuple[np.ndarray, List[str]]:
    if count <= 1 or end <= start:
        values = np.array([start, end], dtype=float)
    else:
        values = np.linspace(start, end, count, dtype=float)
    labels: List[str] = []
    for value in values:
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            labels.append(f"{rounded:d}")
        else:
            labels.append(f"{value:.0f}")
    return values, labels


def _marker_data_step(
    ax: plt.Axes,
    span: float,
    orientation: str,
    *,
    marker_size: float,
    fraction: float = 2.0 / 3.0,
) -> tuple[float, float]:
    span = max(float(span), 1e-9)
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
    except Exception:
        bbox = ax.get_window_extent()
    extent = bbox.width if orientation == "x" else bbox.height
    extent = max(extent, 1e-9)
    data_per_pixel = span / extent
    pixels_per_point = fig.dpi / 72.0 if fig.dpi else 1.0
    data_per_point = data_per_pixel * pixels_per_point
    diameter_points = 2.0 * math.sqrt(marker_size / math.pi)
    diameter_data = diameter_points * data_per_point
    step_data = diameter_data * fraction
    return step_data, diameter_data


def _compute_linear_offsets(
    values: Sequence[float],
    focus_keys: Sequence[int],
    *,
    step: float,
    threshold: float = 0.025,
) -> Tuple[List[float], List[List[int]]]:
    count = len(values)
    offsets = [0.0] * count
    clusters: List[List[int]] = []
    if count == 0:
        return offsets, clusters
    if count == 1 or step <= 0.0:
        clusters = [[idx] for idx in range(count)]
        return offsets, clusters
    order = sorted(range(count), key=lambda idx: values[idx])
    i = 0
    while i < count:
        base_idx = order[i]
        cluster = [base_idx]
        j = i + 1
        while j < count and abs(values[order[j]] - values[base_idx]) <= threshold:
            cluster.append(order[j])
            j += 1
        cluster_sorted = sorted(cluster, key=lambda idx: focus_keys[idx], reverse=True)
        clusters.append(cluster_sorted)
        for pos, idx in enumerate(cluster_sorted):
            offsets[idx] = -step * pos
        i = j
    return offsets, clusters


def _plot_example_traces(
    ax_raster: plt.Axes,
    ax_rates: plt.Axes,
    payload: Dict[str, Any] | None,
    *,
    parameter: Dict[str, Any],
    raster_duration: float | None,
    rates_duration: float | None,
    font_cfg: FontCfg,
    focus_markers: Sequence[FocusMarker],
    color_map: Dict[int, str],
) -> None:
    if payload is None:
        for ax, label in ((ax_raster, "raster"), (ax_rates, "rate traces")):
            ax.text(0.5, 0.5, f"No {label} available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    excit_neurons = int(parameter.get("N_E", 0) or 0)
    states_raw = payload.get("states")
    if states_raw is None:
        states_arr = np.zeros((0, 0), dtype=np.uint8)
    else:
        states_arr = np.asarray(states_raw)
    sample_interval = max(1, int(payload.get("sample_interval", 1) or 1))
    raster_window = (0.0, float(raster_duration)) if raster_duration is not None and raster_duration > 0 else None
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
    total_neurons = excit_neurons + int(parameter.get("N_I", 0) or 0)
    updates_raw = payload.get("state_updates")
    deltas_raw = payload.get("state_deltas")
    init_state = payload.get("initial_state")
    if init_state is not None and not np.asarray(init_state).size:
        init_state = None
    if updates_raw is not None and deltas_raw is not None and np.asarray(updates_raw).size:
        state_source = BinaryStateSource.from_diff_logs(
            updates_raw,
            deltas_raw,
            neuron_count=total_neurons,
            initial_state=init_state,
        )
    else:
        state_source = BinaryStateSource.from_array(states_arr)
    existing = {id(text) for text in ax_raster.texts}
    times_raw = payload.get("times")
    time_axis = np.asarray(times_raw, dtype=float) if times_raw is not None else np.empty(0, dtype=float)
    rates_raw = payload.get("rates")
    rates = np.asarray(rates_raw) if rates_raw is not None else np.zeros((0, 0), dtype=float)
    names = payload.get("names") or []
    state_steps = int(states_arr.shape[0])
    rate_steps = int(rates.shape[0])
    total_steps = max(state_steps, rate_steps)
    finite_times = time_axis[np.isfinite(time_axis)]
    if finite_times.size:
        window_start = float(finite_times.min())
        window_end = float(finite_times.max())
    else:
        window_start = 0.0
        window_end = float(total_steps * sample_interval)
    if window_end <= window_start:
        window_end = window_start + float(sample_interval)
    scale_start = window_start
    scale_end = window_end
    if raster_duration is not None and raster_duration > 0:
        scale_start = 0.0
        scale_end = float(raster_duration)
    if rates_duration is not None and rates_duration > 0:
        scale_start = 0.0
        scale_end = float(rates_duration)
    if scale_end <= scale_start:
        scale_end = scale_start + float(sample_interval)
    time_scale, time_label = _time_axis_scale(scale_start, scale_end)
    safe_scale = time_scale if time_scale > 0 else 1.0
    plot_window = raster_window if raster_window is not None else (window_start, window_end)
    spike_times = np.asarray(payload.get("spike_times"), dtype=float)
    spike_ids = np.asarray(payload.get("spike_ids"), dtype=np.int64)
    n_inh = max(total_neurons - excit_neurons, 0)
    if spike_times.size and spike_ids.size:
        scaled_spike_times = spike_times / safe_scale
        if plot_window is not None:
            t_start = plot_window[0] / safe_scale
            t_end = plot_window[1] / safe_scale
        else:
            t_start = None
            t_end = None
        plot_spike_raster(
            ax=ax_raster,
            spike_times_ms=scaled_spike_times,
            spike_ids=spike_ids,
            n_exc=excit_neurons,
            n_inh=n_inh,
            stride=1,
            t_start=t_start,
            t_end=t_end,
            marker=".",
            marker_size=3.0,
            labels=labels,
        )
    else:
        plot_binary_raster(
            ax=ax_raster,
            state_source=state_source,
            sample_interval=sample_interval,
            n_exc=excit_neurons,
            total_neurons=total_neurons,
            window=plot_window,
            time_scale=time_scale,
            stride=1,
            labels=labels,
            marker=".",
            marker_size=2.0,
            empty_text="No neuron state samples",
        )
    for text in ax_raster.texts:
        if id(text) not in existing:
            label = text.get_text().strip().lower()
            if label.startswith("inh"):
                text.set_color("#8B0000")
            elif label.startswith("exc"):
                text.set_color("black")
    ax_raster.set_title("")
    ax_raster.set_xlabel("")
    ax_raster.set_ylabel("")
    ax_raster.tick_params(axis="x", labelbottom=False)
    ax_raster.tick_params(axis="y", left=False, labelleft=False)
    _plot_grayscale_rates(
        ax_rates,
        time_axis,
        rates,
        names,
        time_scale=time_scale,
        sample_interval=sample_interval,
    )
    scaled_start = scale_start / safe_scale
    scaled_end = scale_end / safe_scale
    if not np.isfinite(scaled_start):
        scaled_start = 0.0
    if not np.isfinite(scaled_end):
        scaled_end = scaled_start + 1.0
    if scaled_end <= scaled_start:
        scaled_end = scaled_start + 1.0
    ax_raster.set_xlim(scaled_start, scaled_end)
    ax_rates.set_xlim(scaled_start, scaled_end)
    ax_rates.set_ylabel(r"$m_c$")
    ax_rates.set_xlabel(time_label)
    tick_vals, tick_labels = _format_time_ticks(scaled_start, scaled_end, 4)
    ax_rates.set_xticks(tick_vals)
    ax_rates.set_xticklabels(tick_labels)
    _plot_fixpoint_overlays(
        ax_rates,
        focus_markers,
        color_map=color_map,
        x_start=scaled_start,
        x_end=scaled_end,
    )


def _plot_fixpoint_overlays(
    ax: plt.Axes,
    markers: Sequence[FocusMarker],
    *,
    color_map: Dict[int, str],
    x_start: float,
    x_end: float,
) -> None:
    if not markers:
        return
    if not np.isfinite(x_start) or not np.isfinite(x_end):
        return
    if x_end <= x_start:
        x_end = x_start + 1.0
    span = x_end - x_start
    step_x, diameter_x = _marker_data_step(ax, span, "x", marker_size=25.0)
    marker_x = x_end - max(diameter_x / 2.0, 0.0)
    fallback_color = "#444444"
    valid_entries: List[Tuple[int, float, int]] = []
    for idx, marker in enumerate(markers):
        value = float(marker.value)
        if not np.isfinite(value):
            continue
        valid_entries.append((idx, value, marker.focus))
    if not valid_entries:
        return
    values = [entry[1] for entry in valid_entries]
    focuses = [entry[2] for entry in valid_entries]
    offsets, clusters = _compute_linear_offsets(values, focuses, step=step_x, threshold=0.025)
    aligned_levels = list(values)
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        leftmost_idx = min(cluster, key=lambda idx: offsets[idx])
        ref_value = values[leftmost_idx]
        for member in cluster:
            aligned_levels[member] = ref_value
    for (entry, offset) in zip(valid_entries, offsets):
        idx, value, _ = entry
        marker = markers[idx]
        color = color_map.get(marker.focus, fallback_color)
        linestyle = "-" if marker.stable else "--"
        value = aligned_levels[idx]
        ax.hlines(
            value,
            x_start,
            x_end,
            colors=color,
            linewidth=0.8,
            linestyles=linestyle,
            alpha=0.5,
            zorder=0.4,
        )
        facecolor = color if marker.stable else "white"
        ax.scatter(
            marker_x + offset,
            value,
            marker="o",
            s=25.0,
            facecolors=facecolor,
            edgecolors=color,
            linewidths=0.8,
            zorder=0.5,
        )


def _plot_grayscale_rates(
    ax: plt.Axes,
    times: np.ndarray,
    rates: np.ndarray,
    names: Sequence[str],
    *,
    time_scale: float,
    sample_interval: int,
) -> None:
    if rates.ndim != 2 or rates.shape[0] == 0:
        ax.text(0.5, 0.5, "No excitatory spikes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    excit_indices = [idx for idx, name in enumerate(names) if str(name).startswith("E")]
    if not excit_indices:
        ax.text(0.5, 0.5, "No excitatory populations", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    excit_rates = rates[:, excit_indices]
    if times.size:
        time_axis = np.asarray(times, dtype=float)
        valid = np.isfinite(time_axis)
        if time_axis.size != excit_rates.shape[0] or not np.all(valid):
            time_axis = np.arange(excit_rates.shape[0], dtype=float) * float(sample_interval)
        else:
            time_axis = time_axis.astype(float, copy=False)
    else:
        time_axis = np.arange(excit_rates.shape[0], dtype=float) * float(sample_interval)
    safe_scale = time_scale if time_scale > 0 else 1.0
    scaled_time_axis = time_axis / safe_scale
    cmap = plt.get_cmap("Greys")
    if excit_rates.shape[1] <= 1:
        shades = [0.55]
    else:
        shades = np.linspace(0.25, 0.85, excit_rates.shape[1])
    for idx, shade in enumerate(shades):
        ax.plot(scaled_time_axis, excit_rates[:, idx], color=cmap(shade), linewidth=1.2)
    finite_scaled = scaled_time_axis[np.isfinite(scaled_time_axis)]
    if finite_scaled.size:
        ax.set_xlim(float(finite_scaled.min()), float(finite_scaled.max()))
    ax.set_ylim(bottom=0.0)


def _plot_histogram(
    ax: plt.Axes,
    pooled: np.ndarray,
    focus_markers: Sequence[FocusMarker],
    bins: int,
    *,
    color_map: Dict[int, str],
    colorbar_entries: Sequence[Tuple[int, str]],
    fig: plt.Figure,
    font_cfg: FontCfg,
) -> None:
    edges = np.linspace(0.0, 1.0, max(2, int(bins) + 1), endpoint=True)
    counts, _, _ = ax.hist(
        pooled,
        bins=edges,
        color="#7fb0ff",
        alpha=0.8,
        label="Max excitatory bin activity",
        density=True,
    )
    ax.set_xlim(0.0, 1.0)
    max_density = float(counts.max()) if counts.size else 0.0
    marker_level = max_density * 1.05 if max_density > 0 else 0.05
    fallback_color = "#444444"
    marker_size = 80.0
    line_kwargs = {"linewidth": 1.2, "alpha": 0.8}
    has_stable = False
    has_unstable = False
    valid_entries: List[Tuple[int, float, int]] = []
    for idx, marker in enumerate(focus_markers):
        value = float(marker.value)
        if np.isfinite(value):
            valid_entries.append((idx, value, marker.focus))
    values = [entry[1] for entry in valid_entries]
    focus_keys = [entry[2] for entry in valid_entries]
    pad = max(0.2, marker_level * 0.15)
    ymax = marker_level + pad
    step_y, _ = _marker_data_step(ax, ymax, "y", marker_size=marker_size)
    offsets, clusters = _compute_linear_offsets(values, focus_keys, step=step_y, threshold=0.025)
    aligned_positions = list(values)
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        reference_idx = cluster[0]
        anchor_value = values[reference_idx]
        for member in cluster:
            aligned_positions[member] = anchor_value
    for (entry, offset) in zip(valid_entries, offsets):
        idx, _value, _ = entry
        marker = focus_markers[idx]
        color = color_map.get(marker.focus, fallback_color)
        linestyle = "-" if marker.stable else "--"
        facecolors = color if marker.stable else "white"
        edgecolors = color
        value = aligned_positions[idx]
        ax.scatter(
            value,
            marker_level + offset,
            marker="o",
            s=marker_size,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=1.0,
            zorder=3,
        )
        ax.vlines(
            value,
            0.0,
            marker_level,
            colors=color,
            linestyles=linestyle,
            zorder=2,
            **line_kwargs,
        )
        if marker.stable:
            has_stable = True
        else:
            has_unstable = True
    ax.set_ylim(0.0, ymax)
    if colorbar_entries:
        draw_listed_colorbar(
            fig,
            ax,
            colorbar_entries,
            font_cfg=font_cfg,
            label="MF prediction: # active clusters",
            use_parent_axis=True,
        )
    if has_stable or has_unstable:
        handles = []
        if has_stable:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="None",
                    markerfacecolor="black",
                    markersize=6,
                    label="stable",
                )
            )
        if has_unstable:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="None",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=6,
                    label="unstable",
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            ncol=2,
            frameon=False,
            fontsize=font_cfg.legend,
            columnspacing = 0.4,
            handletextpad= -0.3,
            bbox_to_anchor = [0.62, 1.04]
        )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="upper"))


def _save_figure(fig: plt.Figure, output_prefix: str, r_value: float) -> None:
    encoded_r = f"{r_value:.2f}".replace(".", "_")
    base = Path(output_prefix)#.with_name(f"{Path(output_prefix).name}_REplus{encoded_r}")
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path, dpi=600)
    print(f"Stored Figure 3 panel at {png_path} and {pdf_path}")


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    focus_counts = helpers.resolve_focus_counts(parameter, args.focus_counts)
    r_eplus_values = _resolve_r_eplus_list(args, parameter)
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    color_map, colorbar_entries = _prepare_line_color_map(
        focus_counts,
        colormap="viridis_r",
    )
    sweep_cfg = helpers.PipelineSweepSettings(
        v_start=args.v_start,
        v_end=args.v_end,
        v_steps=args.v_steps,
        retry_step=args.retry_step,
        jobs=max(1, int(args.erf_jobs or 1)),
        overwrite_simulation=bool(args.overwrite_erf),
        plot_erfs=False,
    )
    binary_overrides = helpers.BinaryRunSettings(
        warmup_steps=args.warmup_steps,
        simulation_steps=args.simulation_steps,
        sample_interval=args.sample_interval,
        batch_size=args.batch_size,
        seed=args.seed,
        output_name=args.output_name,
    )
    for r_value in r_eplus_values:
        print(f"=== Figure 3 workflow for R_Eplus = {r_value:.4f} ===")
        param_copy = deepcopy(parameter)
        param_copy["R_Eplus"] = float(r_value)
        folder, bundle_path = helpers.ensure_fixpoint_bundle(
            param_copy,
            focus_counts,
            [float(r_value)],
            sweep_cfg,
        )
        binary_cfg = helpers.resolve_binary_config(param_copy, binary_overrides)
        base_seed = int(binary_cfg.get("seed", 0) or 0)
        result = helpers.run_legacy_max_rate_analysis(
            param_copy,
            binary_cfg,
            folder_hint=folder,
            bundle_path=bundle_path,
            focus_counts=focus_counts,
            stability_filter=args.stability_filter,
            bin_size=max(1, int(args.bin_size)),
            total_simulations=max(0, int(args.simulations)),
            base_seed=base_seed,
            jobs=max(1, int(args.jobs or 1)),
            analysis_only=args.analysis_only,
            overwrite_simulation=args.overwrite_simulation,
            overwrite_analysis=args.overwrite_analysis,
        )
        payload = None
        if result.example_trace_path and os.path.exists(result.example_trace_path):
            try:
                payload = _load_trace_payload(result.example_trace_path)
            except Exception as exc:
                print(f"Warning: could not load example trace {result.example_trace_path}: {exc}")
                payload = None
        focus_markers = _collect_focus_markers(result.focus_rates)
        fig = plt.figure(figsize=(13, 5))
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.2, 1.0],
            wspace=0.15,
            left=0.075,
            right=0.995,
            top=0.94,
            bottom=0.13,
        )
        left_grid = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.6], hspace=0.08)
        ax_raster = fig.add_subplot(left_grid[0, 0])
        ax_rates = fig.add_subplot(left_grid[1, 0], sharex=ax_raster)
        ax_hist = fig.add_subplot(outer[0, 1])
        _plot_example_traces(
            ax_raster,
            ax_rates,
            payload,
            parameter=param_copy,
            raster_duration=args.raster_duration,
            rates_duration=args.rates_duration,
            font_cfg=font_cfg,
            focus_markers=focus_markers,
            color_map=color_map,
        )
        ax_hist.set_xlabel(r"$\max_{c}\,m_c$")
        ax_hist.set_ylabel("Density")
        _plot_histogram(
            ax_hist,
            result.pooled_maxima,
            focus_markers,
            bins=max(1, int(args.bins)),
            color_map=color_map,
            colorbar_entries=colorbar_entries,
            fig=fig,
            font_cfg=font_cfg,
        )
        kappa_value = float(param_copy.get("kappa", 0.0) or 0.0)
        ax_raster.set_title(rf"$\kappa={kappa_value:.1f}$", fontsize=font_cfg.title)
        style_axes(ax_raster, font_cfg, set_xlabel=False, set_ylabel=False)
        style_axes(ax_rates, font_cfg)
        style_axes(ax_hist, font_cfg)
        add_panel_label(ax_raster, "a1", font_cfg, x=-0.15, y=1.04)
        add_panel_label(ax_rates, "a2", font_cfg, x=-0.15, y=1.06)
        add_panel_label(ax_raster, "b", font_cfg, x=1.06, y=1.04)
        _save_figure(fig, args.output_prefix, r_value)
        plt.close(fig)


if __name__ == "__main__":
    main()
