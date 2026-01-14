#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figure_helpers as helpers  # noqa: E402
from plotting import BinaryStateSource, FontCfg, RasterLabels, add_panel_label, plot_binary_raster, style_axes  # noqa: E402
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
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "sample_interval": sample_interval,
    }


def _plot_example_traces(
    ax_raster: plt.Axes,
    ax_rates: plt.Axes,
    payload: Dict[str, Any] | None,
    *,
    parameter: Dict[str, Any],
    raster_duration: float | None,
    rates_duration: float | None,
    font_cfg: FontCfg,
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
    sample_interval = int(payload.get("sample_interval", 1))
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
    state_source = BinaryStateSource.from_array(states_arr)
    total_neurons = excit_neurons + int(parameter.get("N_I", 0) or 0)
    existing = {id(text) for text in ax_raster.texts}
    plot_binary_raster(
        ax=ax_raster,
        state_source=state_source,
        sample_interval=sample_interval,
        n_exc=excit_neurons,
        total_neurons=total_neurons,
        window=raster_window,
        stride=1,
        labels=labels,
        marker=".",
        marker_size=3.0,
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
    times = np.asarray(payload.get("times")) if payload.get("times") is not None else np.arange(states_arr.shape[0])
    rates_raw = payload.get("rates")
    rates = np.asarray(rates_raw) if rates_raw is not None else np.zeros((0, 0), dtype=float)
    names = payload.get("names") or []
    _plot_grayscale_rates(ax_rates, times, rates, names)
    if raster_duration is not None and raster_duration > 0:
        ax_raster.set_xlim(0.0, raster_duration)
    if rates_duration is not None and rates_duration > 0:
        ax_rates.set_xlim(0.0, rates_duration)
    ax_rates.set_ylabel(r"$m_c$")
    ax_rates.set_xlabel("Time [a.u.]")


def _plot_grayscale_rates(ax: plt.Axes, times: np.ndarray, rates: np.ndarray, names: Sequence[str]) -> None:
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
    time_axis = np.asarray(times, dtype=float)
    if time_axis.size != excit_rates.shape[0]:
        time_axis = np.arange(excit_rates.shape[0], dtype=float)
    cmap = plt.get_cmap("Greys")
    if excit_rates.shape[1] <= 1:
        shades = [0.55]
    else:
        shades = np.linspace(0.25, 0.85, excit_rates.shape[1])
    for idx, shade in enumerate(shades):
        ax.plot(time_axis, excit_rates[:, idx], color=cmap(shade), linewidth=1.2)
    if time_axis.size:
        ax.set_xlim(time_axis.min(), time_axis.max())
    ax.set_ylim(bottom=0.0)


def _focus_boundaries(values: Sequence[int]) -> List[float]:
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) == 1:
        val = float(ordered[0])
        return [val - 0.5, val + 0.5]
    boundaries = [float(ordered[0]) - 0.5]
    for prev_val, next_val in zip(ordered[:-1], ordered[1:]):
        boundaries.append((float(prev_val) + float(next_val)) / 2.0)
    boundaries.append(float(ordered[-1]) + 0.5)
    return boundaries


def _focus_color_mapping(
    focus_counts: Sequence[int] | None,
    data_counts: Sequence[int],
) -> Tuple[Dict[int, Tuple[float, float, float, float]], mcolors.BoundaryNorm | None, Sequence[int], mcolors.Colormap | None]:
    requested = {int(value) for value in (focus_counts or []) if value is not None}
    data_only = {int(value) for value in data_counts if value is not None}
    ordered = sorted(requested | data_only) if (requested or data_only) else []
    if not ordered:
        return {}, None, [], None
    base_cmap = plt.get_cmap("viridis_r")
    samples = np.linspace(0.15, 0.95, len(ordered))
    colors = [base_cmap(sample) for sample in samples]
    listed = mcolors.ListedColormap(colors)
    boundaries = _focus_boundaries(ordered)
    norm = mcolors.BoundaryNorm(boundaries, listed.N)
    mapping = {focus: colors[idx] for idx, focus in enumerate(ordered)}
    return mapping, norm, ordered, listed


def _plot_histogram(
    ax: plt.Axes,
    pooled: np.ndarray,
    focus_rates: Dict[int, Dict[str, List[float]]],
    focus_expectations: Dict[int, float],
    bins: int,
    *,
    focus_counts: Sequence[int],
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
        density=True
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    color_map, norm, tick_values, cmap = _focus_color_mapping(focus_counts, focus_rates.keys())
    fallback_color = "#444444"
    max_count = float(counts.max()) if counts.size else 1.0
    marker_level = max_count * 1.08
    marker_size = 160.0
    has_stable = False
    has_unstable = False
    for focus in sorted(focus_rates.keys()):
        payload = focus_rates.get(focus, {})
        color = color_map.get(int(focus), fallback_color)
        stable_vals = payload.get("stable", [])
        unstable_vals = payload.get("unstable", [])
        best_value = None
        best_stable = False
        if stable_vals:
            best_value = max(stable_vals)
            best_stable = True
        if unstable_vals:
            candidate = max(unstable_vals)
            if best_value is None or candidate > best_value:
                best_value = candidate
                best_stable = False
        if best_value is None:
            continue
        if best_stable:
            has_stable = True
            ax.scatter(
                best_value,
                marker_level,
                marker="v",
                s=marker_size,
                color=color,
                edgecolor="black",
                linewidths=0.8,
                zorder=3,
            )
        else:
            has_unstable = True
            ax.scatter(
                best_value,
                marker_level,
                marker="v",
                s=marker_size,
                facecolors="white",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )
        expectation = focus_expectations.get(int(focus))
        if expectation is not None:
            ax.axvline(
                expectation,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.5,
            )
    if cmap is not None and norm is not None and tick_values:
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(
            scalar,
            ax=ax,
            pad=0.02,
            fraction=0.08,
            ticks=tick_values,
        )
        colorbar.ax.tick_params(labelsize=font_cfg.tick)
        colorbar.ax.set_ylabel("MF prediction: # active clusters", fontsize=font_cfg.label)
    if has_stable and has_unstable:
        handles = [
            Line2D([], [], marker="v", color="black", linestyle="None", markersize=8, label="stable"),
            Line2D(
                [],
                [],
                marker="v",
                color="black",
                linestyle="None",
                markerfacecolor="white",
                markersize=8,
                label="unstable",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=font_cfg.legend, frameon=False)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


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
        fig = plt.figure(figsize=(13, 5))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.2, left=0.1, right=0.97, top=0.94, bottom=0.13)
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
        )
        ax_hist.set_xlabel(r"$\max_{c}\,m_c$")
        ax_hist.set_ylabel("Density")
        _plot_histogram(
            ax_hist,
            result.pooled_maxima,
            result.focus_rates,
            result.focus_expectations,
            bins=max(1, int(args.bins)),
            focus_counts=focus_counts,
            fig=fig,
            font_cfg=font_cfg,
        )
        style_axes(ax_raster, font_cfg, set_xlabel=False, set_ylabel=False)
        style_axes(ax_rates, font_cfg)
        style_axes(ax_hist, font_cfg)
        add_panel_label(ax_raster, "a1", font_cfg, x=-0.15, y=1.04)
        add_panel_label(ax_rates, "a2", font_cfg, x=-0.15, y=1.06)
        add_panel_label(ax_hist, "b", font_cfg, x=-0.08, y=1.04)
        _save_figure(fig, args.output_prefix, r_value)
        plt.close(fig)


if __name__ == "__main__":
    main()
