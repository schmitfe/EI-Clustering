#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib import ticker as mpl_ticker  # noqa: E402
import numpy as np  # noqa: E402

from figure_cli import add_v_sweep_arguments, parse_int_values, resolve_float_values, resolve_v_sweep
from pipelines import figure_helpers as helpers  # noqa: E402
from plotting import (  # noqa: E402
    FontCfg,
    _prepare_value_color_map,
    add_panel_label,
    draw_listed_colorbar,
    style_axes,
)
from sim_config import add_override_arguments, load_from_args  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
DEFAULT_MAX_PAIRS = 200_000
DEFAULT_KAPPA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
MEASURES = ("output", "input")
CATEGORIES = ("within", "across")
FISHER_EPS = 1e-6
BAND_ALPHA = 0.15
WITHIN_STYLE = {"linestyle": "-", "marker": "o", "fillstyle": "full"}
ACROSS_STYLE = {"linestyle": "--", "marker": "s", "fillstyle": "none"}
MEASURE_KEY_MAP = {
    ("output", "within"): "state_excit_within",
    ("output", "across"): "state_excit_between",
    ("input", "within"): "field_excit_within",
    ("input", "across"): "field_excit_between",
}


@dataclass
class ConnectivityInstance:
    parameter: Dict[str, Any]
    label: str
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 4 by sweeping kappa values, running the multi-initialization BinaryNetwork workflow, "
            "and plotting state/subthreshold correlations."
        )
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--kappas",
        type=str,
        nargs="+",
        default=["0.0", "0.25", "0.5", "0.75", "1.0"],
        help="Explicit list of kappas or range expressions (e.g., 0.0 0.5 1.0 or 0:1:0.25).",
    )
    parser.add_argument(
        "--mean-connectivity",
        type=str,
        nargs="+",
        help="Optional list of target mean connectivities or range expressions (e.g., 0.2 0.25 0.3 or 0.2:0.3:0.05). Defaults to the base config.",
    )
    parser.add_argument(
        "--focus-counts",
        type=str,
        nargs="+",
        help="Focus counts or integer ranges to include (e.g., 1 2 4 or 1:4:1). Defaults to all values from 1..Q.",
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
        default=20,
        help="Number of network seeds simulated per kappa (default: %(default)s).",
    )
    parser.add_argument(
        "--n-inits",
        type=int,
        default=20,
        help="Fixpoint initializations per network (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for confidence intervals (default: %(default)s).",
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
        "--stride-analysis",
        type=int,
        default=1,
        help="Down-sampling factor when analyzing neuron states (default: %(default)s).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Maximum number of neuron pairs sampled per category (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for the multi-init simulations (default: %(default)s).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new multi-init simulations and reuse existing traces.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if traces exist.",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="Recompute correlations even if cached results exist.",
    )
    add_v_sweep_arguments(parser)
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
        default="Figures/Figure4",
        help="Prefix for the saved figure files (default: %(default)s.{png,pdf}).",
    )
    parser.add_argument(
        "--no-std-shading",
        action="store_true",
        help="Disable the shaded confidence regions around the correlation traces.",
    )
    return parser.parse_args()


def _resolve_kappa_values(args: argparse.Namespace) -> List[float]:
    values = resolve_float_values(args.kappas, option_name="--kappas", default=DEFAULT_KAPPA_VALUES)
    return list(values or [])


def _resolve_mean_connectivity_values(values: Sequence[str] | None) -> List[float] | None:
    return resolve_float_values(values, option_name="--mean-connectivity")


def _build_instances(parameter: Dict[str, Any], targets: Sequence[float] | None) -> List[ConnectivityInstance]:
    instances: List[ConnectivityInstance] = []
    if targets:
        for target in targets:
            scaled = helpers.scale_connectivity(parameter, target)
            conn_value = helpers.mean_connectivity(scaled)
            label = f"conn={conn_value:.2f}"
            instances.append(ConnectivityInstance(parameter=deepcopy(scaled), label=label, value=conn_value))
        return instances
    base_conn = helpers.mean_connectivity(parameter)
    label = f"conn={base_conn:.2f}"
    return [ConnectivityInstance(parameter=deepcopy(parameter), label=label, value=base_conn)]


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


def _format_progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    total = max(1, int(total))
    completed = max(0, min(int(completed), total))
    filled = int(round(width * completed / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render_progress_line(
    *,
    conn_index: int,
    conn_total: int,
    conn_label: str,
    kappa_index: int,
    kappa_total: int,
    kappa_value: float,
    network_completed: int,
    network_total: int,
    kappa_completed: bool,
) -> str:
    combo_done = (conn_index - 1) * max(1, kappa_total) + kappa_index - 1 + int(bool(kappa_completed))
    combo_total = max(1, conn_total * max(1, kappa_total))
    combo_bar = _format_progress_bar(combo_done, combo_total, width=16)
    network_bar = _format_progress_bar(network_completed, network_total, width=16)
    return (
        f"{combo_bar} sweep {combo_done}/{combo_total} | "
        f"{conn_label} ({conn_index}/{conn_total}) | "
        f"kappa={kappa_value:.3f} ({kappa_index}/{kappa_total}) | "
        f"{network_bar} networks {network_completed}/{network_total}"
    )


def _emit_progress(message: str, *, done: bool = False) -> None:
    if sys.stdout.isatty():
        suffix = "\n" if done else "\r"
        padding = " " * 8 if done else ""
        print(f"{message}{padding}", end=suffix, flush=True)
        return
    print(message, flush=True)


def _network_output_name(base_name: str, network_index: int) -> str:
    return f"{base_name}_net{network_index:03d}"


def _with_output_name(settings: helpers.BinaryRunSettings, output_name: str) -> helpers.BinaryRunSettings:
    return helpers.BinaryRunSettings(
        warmup_steps=settings.warmup_steps,
        simulation_steps=settings.simulation_steps,
        sample_interval=settings.sample_interval,
        batch_size=settings.batch_size,
        seed=settings.seed,
        output_name=output_name,
    )


def _empty_metric_tracker() -> Dict[str, List[float]]:
    return {"mean": [], "lower": [], "upper": []}


def _init_metric_map() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    return {measure: {category: _empty_metric_tracker() for category in CATEGORIES} for measure in MEASURES}


@dataclass
class SeriesStats:
    mean: float
    lower: float
    upper: float


def _append_metric(metrics: Dict[str, Dict[str, Dict[str, List[float]]]], measure: str, category: str, stats: SeriesStats) -> None:
    metrics[measure][category]["mean"].append(stats.mean)
    metrics[measure][category]["lower"].append(stats.lower)
    metrics[measure][category]["upper"].append(stats.upper)


def _fisher_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    clipped = np.clip(finite, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS)
    return float(np.mean(np.arctanh(clipped)))


def _load_analysis_payload(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _summarize_init_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {measure: {category: float("nan") for category in CATEGORIES} for measure in MEASURES}
    for (measure, category), key in MEASURE_KEY_MAP.items():
        values = payload.get(key)
        summary[measure][category] = _fisher_mean(values if values is not None else np.zeros(0, dtype=float))
    return summary


def _merge_init_summaries(entries: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {measure: {category: float("nan") for category in CATEGORIES} for measure in MEASURES}
    if not entries:
        return merged
    for measure in MEASURES:
        for category in CATEGORIES:
            values = [entry[measure][category] for entry in entries if np.isfinite(entry[measure][category])]
            merged[measure][category] = float(np.mean(values)) if values else float("nan")
    return merged


def _summarize_network_dir(analysis_dir: Path, n_inits: int) -> Dict[str, Dict[str, float]]:
    payloads: List[Dict[str, Dict[str, float]]] = []
    for init_idx in range(n_inits):
        path = analysis_dir / f"analysis_init{init_idx:04d}.npz"
        if not path.exists():
            continue
        payload = _load_analysis_payload(path)
        payloads.append(_summarize_init_payload(payload))
    if not payloads:
        raise FileNotFoundError(f"No analysis files were found in {analysis_dir}.")
    return _merge_init_summaries(payloads)


def _series_stats_from_values(values: Sequence[float], bootstrap_samples: int, rng: np.random.Generator) -> SeriesStats:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return SeriesStats(mean=float("nan"), lower=float("nan"), upper=float("nan"))
    mean_z = float(np.mean(arr))
    mean_r = float(np.tanh(mean_z))
    if arr.size == 1 or bootstrap_samples <= 0:
        return SeriesStats(mean=mean_r, lower=mean_r, upper=mean_r)
    indices = rng.integers(0, arr.size, size=(bootstrap_samples, arr.size))
    boot_means = arr[indices].mean(axis=1)
    boot_r = np.tanh(boot_means)
    lower, upper = np.percentile(boot_r, [2.5, 97.5])
    return SeriesStats(mean=mean_r, lower=float(lower), upper=float(upper))


def _run_network_initializations(
    parameter: Dict[str, Any],
    *,
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
    n_inits: int,
    network_seeds: Sequence[int],
    init_seed_bases: Sequence[int],
    binary_overrides: helpers.BinaryRunSettings,
    base_output_name: str,
    stride_analysis: int,
    max_pairs: int,
    jobs: int,
    analysis_only: bool,
    overwrite_simulation: bool,
    overwrite_analysis: bool,
    progress_callback=None,
    verbose: bool = False,
) -> List[Dict[str, Dict[str, float]]]:
    if len(network_seeds) != len(init_seed_bases):
        raise ValueError("Network seed and initialization seed lists must have the same length.")
    specs: List[helpers.MultiInitCorrelationSpec] = []
    ordered_keys: List[str] = []
    for net_idx, (network_seed, init_seed) in enumerate(zip(network_seeds, init_seed_bases)):
        run_param = deepcopy(parameter)
        run_param_binary = dict(run_param.get("binary") or {})
        output_name = _network_output_name(base_output_name, net_idx)
        run_param_binary["output_name"] = output_name
        run_param["binary"] = run_param_binary
        per_run_overrides = _with_output_name(binary_overrides, output_name)
        binary_cfg = helpers.resolve_binary_config(run_param, per_run_overrides)
        ordered_keys.append(output_name)
        specs.append(
            helpers.MultiInitCorrelationSpec(
                key=output_name,
                label=f"net{net_idx:03d}",
                parameter=run_param,
                binary_cfg=binary_cfg,
                bundle_path=bundle_path,
                focus_counts=tuple(focus_counts),
                stability_filter=stability_filter,
                n_inits=int(n_inits),
                seed_inits=int(init_seed),
                seed_network=int(network_seed),
                stride_analysis=stride_analysis,
                max_pairs=max_pairs,
                analysis_only=analysis_only,
                overwrite_simulation=overwrite_simulation,
                overwrite_analysis=overwrite_analysis,
                verbose=verbose,
            )
        )
    results = helpers.run_multi_init_correlation_batch(
        specs,
        jobs=max(1, int(jobs)),
        progress_callback=progress_callback,
        verbose=verbose,
    )
    network_summaries: List[Dict[str, Dict[str, float]]] = []
    for _, (_, _, output_name) in enumerate(zip(network_seeds, init_seed_bases, ordered_keys)):
        compact_summary = results[output_name].network_summary
        if compact_summary is None:
            raise RuntimeError(f"Missing compact network summary for '{output_name}'.")
        network_summaries.append(compact_summary)
    return network_summaries


def _plot_series(
    ax: plt.Axes,
    kappa: Sequence[float],
    tracker: Dict[str, List[float]],
    *,
    color: Any,
    style: Dict[str, Any],
    show_std_shading: bool,
) -> None:
    if not kappa:
        return
    ax.plot(
        kappa,
        tracker["mean"],
        color=color,
        linewidth=1.5,
        marker=style.get("marker"),
        linestyle=style.get("linestyle"),
        fillstyle=style.get("fillstyle", "full"),
        markersize=4,
    )
    if show_std_shading:
        ax.fill_between(kappa, tracker["lower"], tracker["upper"], color=color, alpha=BAND_ALPHA, linewidth=0)


def _plot_correlation_figure(
    results: Dict[str, Dict[str, Any]],
    *,
    font_cfg: FontCfg,
    output_prefix: str,
    show_std_shading: bool,
) -> None:
    if not results:
        raise ValueError("No simulation data available to plot.")
    fig, (ax_input, ax_output) = plt.subplots(1, 2, sharey=True, figsize=(13 / 2, 4.0))

    plt.subplots_adjust(
        top=0.85,
        bottom=0.15,
        left=0.125,
        right=0.82,
    )
    connectivity_values = [payload["connectivity"] for payload in results.values()]
    color_map, colorbar_entries = _prepare_value_color_map(connectivity_values)
    for ax in (ax_input, ax_output):
        ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
    for payload in results.values():
        connectivity_value = float(payload["connectivity"])
        color = color_map[connectivity_value]
        kappa = payload["kappa"]
        metrics = payload["metrics"]
        _plot_series(
            ax_input,
            kappa,
            metrics["input"]["within"],
            color=color,
            style=WITHIN_STYLE,
            show_std_shading=show_std_shading,
        )
        _plot_series(
            ax_input,
            kappa,
            metrics["input"]["across"],
            color=color,
            style=ACROSS_STYLE,
            show_std_shading=show_std_shading,
        )
        _plot_series(
            ax_output,
            kappa,
            metrics["output"]["within"],
            color=color,
            style=WITHIN_STYLE,
            show_std_shading=show_std_shading,
        )
        _plot_series(
            ax_output,
            kappa,
            metrics["output"]["across"],
            color=color,
            style=ACROSS_STYLE,
            show_std_shading=show_std_shading,
        )
    ax_input.set_ylabel(r"$\overline{r_{\mathrm{in}}}$")
    ax_output.set_ylabel(r"$\overline{r_{\mathrm{out}}}$")
    for ax in (ax_output, ax_input):
        ax.set_xlabel(r"$\kappa$")
        style_axes(ax, font_cfg)
    add_panel_label(ax_input, "a", font_cfg, x=-0.08, y=1.08)
    add_panel_label(ax_output, "b", font_cfg, x=-0.08, y=1.08)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=WITHIN_STYLE["linestyle"],
            marker=WITHIN_STYLE["marker"],
            fillstyle=WITHIN_STYLE["fillstyle"],
            markersize=4,
            label="Within cluster",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ACROSS_STYLE["linestyle"],
            marker=ACROSS_STYLE["marker"],
            fillstyle=ACROSS_STYLE["fillstyle"],
            markersize=4,
            label="Across clusters",
        ),
    ]
    legend = fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=font_cfg.legend,
    )
    if len(colorbar_entries) > 1:
        colorbar_ax = fig.add_axes([0.84, 0.15, 0.06, 0.7])
        colorbar = draw_listed_colorbar(
            fig,
            colorbar_ax,
            colorbar_entries,
            font_cfg=font_cfg,
            label=r"$\overline{p}$",
            height_fraction=0.5,
            width_fraction=0.25,
            label_kwargs={"labelpad": 8},
        )
        if colorbar is not None:
            colorbar.ax.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%.2f"))
    #fig.tight_layout(rect=(0, 0, 1, 0.95))
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Stored Figure 4 at {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    parsed_focus_counts = parse_int_values(args.focus_counts, option_name="--focus-counts")
    focus_counts = helpers.resolve_focus_counts(parameter, parsed_focus_counts)
    kappa_values = _resolve_kappa_values(args)
    v_start, v_stop, v_steps = resolve_v_sweep(args)
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    sweep_cfg = helpers.PipelineSweepSettings(
        v_start=v_start,
        v_end=v_stop,
        v_steps=v_steps,
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
    instances = _build_instances(parameter, _resolve_mean_connectivity_values(args.mean_connectivity))
    if not instances:
        raise ValueError("No connectivity instances were generated.")
    base_seed = args.seed if args.seed is not None else 0
    network_seeds, init_seed_bases = _generate_seed_sequences(
        base_seed=base_seed,
        n_networks=int(args.n_networks),
        init_seed_base=args.seed_inits,
        network_seed_override=args.seed_network,
    )
    bootstrap_rng = np.random.default_rng(int(base_seed) + 2024)
    n_inits = max(1, int(args.n_inits))
    max_pairs = max(1, int(args.max_pairs or DEFAULT_MAX_PAIRS))
    jobs = max(1, int(args.jobs or 1))
    stride_analysis = max(1, int(args.stride_analysis or 1))
    bootstrap_samples = max(0, int(args.bootstrap_samples))
    results: Dict[str, Dict[str, Any]] = {}
    total_instances = len(instances)
    total_kappas = len(kappa_values)
    for instance_idx, instance in enumerate(instances, start=1):
        label = instance.label
        print(f"=== Correlation workflow for {label} (mean connectivity {instance.value:.4f}) ===")
        instance_store = {"kappa": [], "metrics": _init_metric_map(), "connectivity": instance.value}
        base_output_name = _resolve_base_output_name(instance.parameter, binary_overrides)
        for kappa_idx, kappa in enumerate(kappa_values, start=1):
            param_kappa = deepcopy(instance.parameter)
            param_kappa["kappa"] = float(kappa)
            r_value = param_kappa.get("R_Eplus")
            if r_value is None:
                raise ValueError("Parameter 'R_Eplus' must be defined (or overridden via -O).")
            print(f"  -> kappa = {kappa:.4f}, R_Eplus = {float(r_value):.4f}")
            _, bundle_path = helpers.ensure_fixpoint_bundle(
                deepcopy(param_kappa),
                focus_counts,
                [float(r_value)],
                sweep_cfg,
            )
            progress_state = {"completed": 0}

            def _progress_callback(_entry: Dict[str, Any], completed: int, total: int) -> None:
                progress_state["completed"] = completed
                _emit_progress(
                    _render_progress_line(
                        conn_index=instance_idx,
                        conn_total=total_instances,
                        conn_label=label,
                        kappa_index=kappa_idx,
                        kappa_total=total_kappas,
                        kappa_value=float(kappa),
                        network_completed=completed,
                        network_total=total,
                        kappa_completed=False,
                    ),
                    done=False,
                )

            network_summaries = _run_network_initializations(
                deepcopy(param_kappa),
                bundle_path=bundle_path,
                focus_counts=focus_counts,
                stability_filter=args.stability_filter,
                n_inits=n_inits,
                network_seeds=network_seeds,
                init_seed_bases=init_seed_bases,
                binary_overrides=binary_overrides,
                base_output_name=base_output_name,
                stride_analysis=stride_analysis,
                max_pairs=max_pairs,
                jobs=jobs,
                analysis_only=args.analysis_only,
                overwrite_simulation=args.overwrite_simulation,
                overwrite_analysis=args.overwrite_analysis,
                progress_callback=_progress_callback,
                verbose=False,
            )
            _emit_progress(
                _render_progress_line(
                    conn_index=instance_idx,
                    conn_total=total_instances,
                    conn_label=label,
                    kappa_index=kappa_idx,
                    kappa_total=total_kappas,
                    kappa_value=float(kappa),
                    network_completed=progress_state["completed"] or len(network_seeds),
                    network_total=len(network_seeds),
                    kappa_completed=True,
                ),
                done=True,
            )
            network_payloads: Dict[str, Dict[str, List[float]]] = {
                measure: {category: [] for category in CATEGORIES} for measure in MEASURES
            }
            for network_summary in network_summaries:
                for measure in MEASURES:
                    for category in CATEGORIES:
                        network_payloads[measure][category].append(network_summary[measure][category])
            for measure in MEASURES:
                for category in CATEGORIES:
                    stats = _series_stats_from_values(
                        network_payloads[measure][category],
                        bootstrap_samples,
                        bootstrap_rng,
                    )
                    _append_metric(instance_store["metrics"], measure, category, stats)
            instance_store["kappa"].append(float(kappa))
        results[label] = instance_store
    _plot_correlation_figure(
        results,
        font_cfg=font_cfg,
        output_prefix=args.output_prefix,
        show_std_shading=not args.no_std_shading,
    )


if __name__ == "__main__":
    main()
