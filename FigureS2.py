#!/usr/bin/env python3
"""Generate Supplementary Figure 4 raster grids across kappa/connectivity sweeps."""
from __future__ import annotations

import argparse
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import binary_simulation_multi_init as binary_multi  # noqa: E402
import figure_helpers as helpers  # noqa: E402
from plotting import (  # noqa: E402
    BinaryStateSource,
    FontCfg,
    RasterLabels,
    plot_binary_raster,
    plot_spike_raster,
    style_axes,
)
from sim_config import add_override_arguments, load_from_args  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

DEFAULT_KAPPA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CELL_WIDTH = 2.0
DEFAULT_RASTER_HEIGHT = 1.45
DEFAULT_RATE_HEIGHT = 0.85


@dataclass
class ConnectivityInstance:
    parameter: Dict[str, Any]
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Supplementary Figure 4 raster grid by sweeping kappa and mean connectivity "
            "and rendering a raster (optionally with rates) for each condition."
        )
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--kappas",
        type=str,
        nargs="+",
        help="Explicit list of kappas or range expressions (e.g., 0.0 0.5 1.0 or 0:1:0.25).",
    )
    parser.add_argument("--kappa-start", type=float, help="Start of the kappa sweep (inclusive).")
    parser.add_argument("--kappa-stop", type=float, help="End of the kappa sweep (inclusive).")
    parser.add_argument("--kappa-step", type=float, help="Step size for the kappa sweep.")
    parser.add_argument(
        "--mean-connectivity",
        type=float,
        nargs="+",
        help="List of target mean connectivities (e.g., 0.2 0.25 0.3). Defaults to the base config.",
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
        help="Select fixpoints with the desired stability (default: %(default)s).",
    )
    parser.add_argument(
        "--n-networks",
        type=int,
        default=1,
        help="Number of network seeds in the pool (default: %(default)s).",
    )
    parser.add_argument(
        "--network-index",
        type=int,
        default=0,
        help="Index of the network seed to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--n-inits",
        type=int,
        default=1,
        help="Number of fixpoint initializations drawn per network (default: %(default)s).",
    )
    parser.add_argument(
        "--init-index",
        type=int,
        default=0,
        help="Index of the initialization to plot (default: %(default)s).",
    )
    parser.add_argument("--seed-inits", type=int, help="Base seed used to draw fixpoint initializations.")
    parser.add_argument("--seed-network", type=int, help="Optional base seed for the network-seed sequence.")
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size.")
    parser.add_argument("--seed", type=int, help="Override binary.seed (base seed for traces).")
    parser.add_argument("--output-name", type=str, help="Custom prefix for saved traces.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new simulations and reuse existing traces.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if traces exist.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for simulations (currently unused; default: %(default)s).",
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
        "--raster-stride",
        type=int,
        default=1,
        help="Plot every Nth neuron in the raster (default: %(default)s).",
    )
    parser.add_argument(
        "--show-rates",
        action="store_true",
        help="Include a rate trace under each raster panel.",
    )
    parser.add_argument(
        "--show-raster-labels",
        action="store_true",
        help="Show Exc/Inh labels on raster plots.",
    )
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
        help="Re-run ERF generations even if matching files exist.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="Figures/SuppFigure4",
        help="Prefix for the saved figure files (default: %(default)s.{png,pdf}).",
    )
    return parser.parse_args()


def _kappa_sequence(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--kappa-step must be positive.")
    values = np.arange(start, stop + 1e-12, step, dtype=float)
    if values.size == 0:
        raise ValueError("Kappa sweep produced no values. Adjust the start/stop/step inputs.")
    return [float(val) for val in values]


def _parse_explicit_kappas(values: Iterable[str]) -> List[float]:
    resolved: List[float] = []
    for raw in values:
        if ":" in raw:
            try:
                start, stop, step = (float(part) for part in raw.split(":"))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid kappa range '{raw}'. Use start:stop:step.") from exc
            resolved.extend(_kappa_sequence(start, stop, step))
        else:
            resolved.append(float(raw))
    if not resolved:
        raise ValueError("No kappas were provided via --kappas.")
    ordered_unique = list(dict.fromkeys(resolved))
    return ordered_unique


def _resolve_kappa_values(args: argparse.Namespace) -> List[float]:
    if args.kappas:
        return _parse_explicit_kappas(args.kappas)
    if args.kappa_start is not None and args.kappa_stop is not None and args.kappa_step is not None:
        return _kappa_sequence(float(args.kappa_start), float(args.kappa_stop), float(args.kappa_step))
    if args.kappa_start is not None or args.kappa_stop is not None or args.kappa_step is not None:
        raise ValueError("Provide --kappa-start/stop/step together or rely on --kappas.")
    return list(DEFAULT_KAPPA_VALUES)


def _build_instances(parameter: Dict[str, Any], targets: Sequence[float] | None) -> List[ConnectivityInstance]:
    instances: List[ConnectivityInstance] = []
    if targets:
        for target in targets:
            scaled = helpers.scale_connectivity(parameter, target)
            conn_value = helpers.mean_connectivity(scaled)
            instances.append(ConnectivityInstance(parameter=deepcopy(scaled), value=conn_value))
        return instances
    base_conn = helpers.mean_connectivity(parameter)
    return [ConnectivityInstance(parameter=deepcopy(parameter), value=base_conn)]


def _generate_seed_sequences(
    *,
    base_seed: int | None,
    n_networks: int,
    init_seed_base: int | None,
    network_seed_override: int | None,
) -> Tuple[List[int], List[int]]:
    if n_networks <= 0:
        raise ValueError("--n-networks must be positive.")
    if network_seed_override is not None:
        network_seeds = [int(network_seed_override) + idx for idx in range(n_networks)]
    else:
        rng_network = np.random.default_rng(0 if base_seed is None else int(base_seed))
        network_seeds = rng_network.integers(0, 2**31 - 1, size=n_networks).tolist()
    init_seed_source = int(init_seed_base) if init_seed_base is not None else (0 if base_seed is None else int(base_seed) + 1)
    rng_init = np.random.default_rng(init_seed_source)
    init_seeds = rng_init.integers(0, 2**31 - 1, size=n_networks).tolist()
    return network_seeds, init_seeds


def _resolve_base_output_name(parameter: Dict[str, Any], overrides: helpers.BinaryRunSettings) -> str:
    if overrides.output_name:
        return str(overrides.output_name)
    binary_cfg = parameter.get("binary") or {}
    candidate = binary_cfg.get("output_name")
    if candidate:
        return str(candidate)
    return "activity_trace"


def _network_output_name(base_name: str, network_index: int) -> str:
    return f"{base_name}_net{network_index:03d}"


def _time_axis_scale_from_taus(parameter: Dict[str, Any]) -> Tuple[float, str]:
    tau_e = float(parameter.get("tau_e", math.nan))
    tau_i = float(parameter.get("tau_i", math.nan))
    n_e = int(parameter.get("N_E", 0) or 0)
    n_i = int(parameter.get("N_I", 0) or 0)
    if not np.isfinite(tau_e) or not np.isfinite(tau_i) or tau_e <= 0.0 or tau_i <= 0.0:
        return 1.0, "Time [s]"
    max_tau = max(tau_e, tau_i)
    min_tau = min(tau_e, tau_i)
    n_max = n_e if tau_e >= tau_i else n_i
    n_min = n_i if tau_e >= tau_i else n_e
    expected_updates = float(n_max) + (max_tau / min_tau) * float(n_min)
    if not np.isfinite(expected_updates) or expected_updates <= 0.0 or max_tau <= 0.0:
        return 1.0, "Time [s]"
    max_tau_seconds = max_tau / 1000.0
    if max_tau_seconds <= 0.0 or not np.isfinite(max_tau_seconds):
        return 1.0, "Time [s]"
    return float(expected_updates / max_tau_seconds), "Time [s]"


def _format_value(value: float, *, precision: int = 2) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


def _load_trace_payload(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Trace {path} does not exist.")
    with np.load(path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
        times = np.asarray(data.get("times"), dtype=float) if "times" in data else np.arange(rates.shape[0])
        states = np.asarray(data.get("neuron_states"), dtype=np.uint8) if "neuron_states" in data else np.zeros((0, 0), dtype=np.uint8)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_updates = np.asarray(data.get("state_updates"), dtype=np.uint16) if "state_updates" in data else None
        state_deltas = np.asarray(data.get("state_deltas"), dtype=np.int8) if "state_deltas" in data else None
        initial_state = np.asarray(data.get("initial_state"), dtype=np.uint8) if "initial_state" in data else None
        spike_times = np.asarray(data.get("spike_times"), dtype=float).ravel() if "spike_times" in data else np.zeros(0, dtype=float)
        spike_ids = np.asarray(data.get("spike_ids"), dtype=np.int64).ravel() if "spike_ids" in data else np.zeros(0, dtype=np.int64)
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "sample_interval": sample_interval,
        "state_updates": state_updates,
        "state_deltas": state_deltas,
        "initial_state": initial_state,
        "spike_times": spike_times,
        "spike_ids": spike_ids,
    }


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


def _plot_raster_panel(
    ax: plt.Axes,
    payload: Dict[str, Any],
    *,
    parameter: Dict[str, Any],
    raster_duration: float | None,
    raster_stride: int,
    font_cfg: FontCfg,
    show_labels: bool,
) -> Tuple[float, float]:
    excit_neurons = int(parameter.get("N_E", 0) or 0)
    total_neurons = excit_neurons + int(parameter.get("N_I", 0) or 0)
    states_raw = payload.get("states")
    states_arr = np.asarray(states_raw) if states_raw is not None else np.zeros((0, 0), dtype=np.uint8)
    sample_interval = max(1, int(payload.get("sample_interval", 1) or 1))
    raster_window = (0.0, float(raster_duration)) if raster_duration is not None and raster_duration > 0 else None
    stride = max(1, int(raster_stride))
    labels = None
    if show_labels:
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
    existing = {id(text) for text in ax.texts}
    time_scale, _ = _time_axis_scale_from_taus(parameter)
    spike_times = np.asarray(payload.get("spike_times"), dtype=float)
    spike_ids = np.asarray(payload.get("spike_ids"), dtype=np.int64)
    n_inh = max(total_neurons - excit_neurons, 0)
    if spike_times.size and spike_ids.size:
        safe_scale = time_scale if time_scale > 0 else 1.0
        scaled_times = spike_times / safe_scale
        t_start = raster_window[0] / safe_scale if raster_window else None
        t_end = raster_window[1] / safe_scale if raster_window else None
        plot_spike_raster(
            ax=ax,
            spike_times_ms=scaled_times,
            spike_ids=spike_ids,
            n_exc=excit_neurons,
            n_inh=n_inh,
            stride=stride,
            t_start=t_start,
            t_end=t_end,
            marker=".",
            marker_size=1.5,
            labels=labels,
        )
    else:
        plot_binary_raster(
            ax=ax,
            state_source=state_source,
            sample_interval=sample_interval,
            n_exc=excit_neurons,
            total_neurons=total_neurons,
            window=raster_window,
            time_scale=time_scale,
            stride=stride,
            labels=labels,
            marker=".",
            marker_size=2.0,
            empty_text="No neuron state samples",
        )
    for text in ax.texts:
        if id(text) not in existing:
            label = text.get_text().strip().lower()
            if label.startswith("inh"):
                text.set_color("#8B0000")
            elif label.startswith("exc"):
                text.set_color("black")
    if raster_window is not None:
        safe_scale = time_scale if time_scale > 0 else 1.0
        ax.set_xlim(raster_window[0] / safe_scale, raster_window[1] / safe_scale)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    return (raster_window[0] if raster_window else 0.0, raster_window[1] if raster_window else 0.0)


def _plot_rate_panel(
    ax: plt.Axes,
    payload: Dict[str, Any],
    *,
    parameter: Dict[str, Any],
    rates_duration: float | None,
    font_cfg: FontCfg,
    show_x: bool,
    show_y: bool,
) -> None:
    time_scale, _ = _time_axis_scale_from_taus(parameter)
    sample_interval = max(1, int(payload.get("sample_interval", 1) or 1))
    _plot_grayscale_rates(
        ax,
        np.asarray(payload.get("times"), dtype=float),
        np.asarray(payload.get("rates"), dtype=float),
        payload.get("names") or [],
        time_scale=time_scale,
        sample_interval=sample_interval,
    )
    if rates_duration is not None and rates_duration > 0:
        safe_scale = time_scale if time_scale > 0 else 1.0
        ax.set_xlim(0.0, float(rates_duration) / safe_scale)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 1.0])
    ax.set_xlabel("Time[s]" if show_x else "")
    ax.set_ylabel(r"$m_c$" if show_y else "")
    ax.tick_params(axis="x", labelbottom=show_x)
    ax.tick_params(axis="y", labelleft=show_y)
    style_axes(ax, font_cfg)


def _axes_bbox_union(axes: Sequence[plt.Axes]) -> Tuple[float, float, float, float]:
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    return x0, x1, y0, y1


def _resolve_trace(
    parameter: Dict[str, Any],
    *,
    kappa: float,
    focus_counts: Sequence[int],
    stability_filter: str,
    sweep_cfg: helpers.PipelineSweepSettings,
    binary_overrides: helpers.BinaryRunSettings,
    network_seed: int,
    init_seed: int,
    init_index: int,
    n_inits: int,
    network_index: int,
    analysis_only: bool,
    overwrite_simulation: bool,
) -> Dict[str, Any]:
    param = deepcopy(parameter)
    param["kappa"] = float(kappa)
    target_rep = param.get("R_Eplus")
    if target_rep is None:
        raise ValueError("Parameter 'R_Eplus' must be defined (or overridden via -O).")
    target_rep = float(target_rep)
    folder, bundle_path = helpers.ensure_fixpoint_bundle(
        deepcopy(param),
        focus_counts,
        [target_rep],
        sweep_cfg,
    )
    base_output_name = _resolve_base_output_name(param, binary_overrides)
    output_name = _network_output_name(base_output_name, network_index)
    param_binary = dict(param.get("binary") or {})
    param_binary["output_name"] = output_name
    param["binary"] = param_binary
    binary_cfg = helpers.resolve_binary_config(param, binary_overrides)
    binary_cfg["seed"] = int(network_seed)
    _, binary_dir, _ = binary_multi._prepare_multi_init_folder(param, folder)
    trace_label = f"{output_name}_init{init_index:04d}"
    trace_path = Path(binary_dir) / f"{trace_label}.npz"
    if trace_path.exists() and not overwrite_simulation:
        return _load_trace_payload(str(trace_path))
    if analysis_only:
        raise FileNotFoundError(f"Trace {trace_path} does not exist. Re-run without --analysis-only.")
    bundle = binary_multi._load_fixpoint_bundle(bundle_path)
    Q_value = int(param.get("Q", 0) or 0)
    pop_length = 2 * Q_value
    candidates = binary_multi._load_fixpoint_candidates(
        bundle,
        focus_counts,
        stability_filter,
        pop_length,
        target_rep,
    )
    if not candidates:
        raise ValueError("No fixpoint candidates available for raster plotting.")
    picks = binary_multi._select_fixpoints(
        candidates,
        seed=int(init_seed),
        count=max(1, int(n_inits)),
    )
    if init_index >= len(picks):
        raise ValueError(
            f"Requested init_index={init_index} but only {len(picks)} initializations were selected."
        )
    candidate = picks[init_index]
    init_rates = tuple(float(value) for value in candidate["rates"])
    print(
        f"Simulating kappa={kappa:.3f} with mean conn={helpers.mean_connectivity(param):.3f} "
        f"(network seed={network_seed}, init seed={init_seed}, init index={init_index})."
    )
    binary_multi.run_legacy_binary_simulation(
        param,
        binary_cfg,
        str(binary_dir),
        trace_label,
        init_rates,
        seed=int(network_seed),
        capture_state_dynamics=True,
        cluster_seed=int(init_index),
    )
    return _load_trace_payload(str(trace_path))


def _save_figure(fig: plt.Figure, output_prefix: str) -> None:
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path, dpi=600)
    print(f"Stored Supplementary Figure 4 at {png_path} and {pdf_path}")


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    focus_counts = helpers.resolve_focus_counts(base_parameter, args.focus_counts)
    kappa_values = sorted(_resolve_kappa_values(args))
    instances = _build_instances(base_parameter, args.mean_connectivity)
    instances.sort(key=lambda inst: inst.value)
    if not instances:
        raise ValueError("No connectivity instances were generated.")
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
    base_seed = args.seed if args.seed is not None else 0
    network_seeds, init_seed_bases = _generate_seed_sequences(
        base_seed=base_seed,
        n_networks=int(args.n_networks),
        init_seed_base=args.seed_inits,
        network_seed_override=args.seed_network,
    )
    if not network_seeds:
        raise ValueError("No network seeds were generated.")
    network_index = int(args.network_index)
    if network_index < 0 or network_index >= len(network_seeds):
        raise ValueError(f"--network-index must be within [0, {len(network_seeds) - 1}].")
    network_seed = int(network_seeds[network_index])
    init_seed = int(init_seed_bases[network_index])
    init_index = max(0, int(args.init_index))
    n_inits = max(init_index + 1, int(args.n_inits))
    n_rows = len(instances)
    n_cols = len(kappa_values)
    show_rates = bool(args.show_rates)
    raster_duration = args.raster_duration
    rates_duration = args.rates_duration
    if show_rates and rates_duration is None:
        rates_duration = raster_duration
    row_height = DEFAULT_RASTER_HEIGHT + (DEFAULT_RATE_HEIGHT if show_rates else 0.0)
    fig_width = n_cols * DEFAULT_CELL_WIDTH + 1.2
    fig_height = n_rows * row_height + 0.8
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.06,
        right=0.92,
        top=0.9,
        bottom=0.08,
        wspace=0.18,
        hspace=0.22 if show_rates else 0.18,
    )
    raster_axes: List[List[plt.Axes]] = []
    rate_axes: List[List[plt.Axes | None]] = []
    for row_idx, instance in enumerate(instances):
        row_rasters: List[plt.Axes] = []
        row_rates: List[plt.Axes | None] = []
        for col_idx, kappa in enumerate(kappa_values):
            if show_rates:
                cell = outer[row_idx, col_idx].subgridspec(
                    2, 1, height_ratios=[1.0, 0.6], hspace=0.08
                )
                ax_raster = fig.add_subplot(cell[0, 0])
                ax_rate = fig.add_subplot(cell[1, 0], sharex=ax_raster)
            else:
                ax_raster = fig.add_subplot(outer[row_idx, col_idx])
                ax_rate = None
            param_copy = deepcopy(instance.parameter)
            payload = _resolve_trace(
                param_copy,
                kappa=float(kappa),
                focus_counts=focus_counts,
                stability_filter=args.stability_filter,
                sweep_cfg=sweep_cfg,
                binary_overrides=binary_overrides,
                network_seed=network_seed,
                init_seed=init_seed,
                init_index=init_index,
                n_inits=n_inits,
                network_index=network_index,
                analysis_only=args.analysis_only,
                overwrite_simulation=args.overwrite_simulation,
            )
            _plot_raster_panel(
                ax_raster,
                payload,
                parameter=param_copy,
                raster_duration=raster_duration,
                raster_stride=args.raster_stride,
                font_cfg=font_cfg,
                show_labels=args.show_raster_labels,
            )
            style_axes(ax_raster, font_cfg, set_xlabel=False, set_ylabel=False)
            if ax_rate is not None:
                show_x = row_idx == n_rows - 1
                show_y = col_idx == 0
                _plot_rate_panel(
                    ax_rate,
                    payload,
                    parameter=param_copy,
                    rates_duration=rates_duration,
                    font_cfg=font_cfg,
                    show_x=show_x,
                    show_y=show_y,
                )
            row_rasters.append(ax_raster)
            row_rates.append(ax_rate)
        raster_axes.append(row_rasters)
        rate_axes.append(row_rates)
    fig.canvas.draw()
    col_pad = 0.01
    for col_idx, kappa in enumerate(kappa_values):
        ax = raster_axes[0][col_idx]
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2.0
        label = rf"$\kappa={_format_value(float(kappa), precision=3)}$"
        fig.text(x_center, bbox.y1 + col_pad, label, ha="center", va="bottom", fontsize=font_cfg.title)
    row_pad = 0.01
    for row_idx, instance in enumerate(instances):
        axes = [raster_axes[row_idx][-1]]
        if show_rates and rate_axes[row_idx][-1] is not None:
            axes.append(rate_axes[row_idx][-1])
        x0, x1, y0, y1 = _axes_bbox_union(axes)
        y_center = (y0 + y1) / 2.0
        label = rf"$\overline{{p}}={_format_value(instance.value)}$"
        fig.text(x1 + row_pad, y_center, label, ha="left", va="center", fontsize=font_cfg.label)
    _save_figure(fig, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
