#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figure_helpers as helpers  # noqa: E402
from plotting import FontCfg, add_panel_label, style_axes  # noqa: E402
from sim_config import add_override_arguments, load_from_args  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
DEFAULT_MAX_PAIRS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 4 by sweeping kappa values, running the multi-initialization legacy workflow, "
            "and plotting state/subthreshold correlations."
        )
    )
    add_override_arguments(parser)
    parser.add_argument("--kappa-start", type=float, required=True, help="Start of the kappa sweep (inclusive).")
    parser.add_argument("--kappa-stop", type=float, required=True, help="End of the kappa sweep (inclusive).")
    parser.add_argument("--kappa-step", type=float, required=True, help="Step size for the kappa sweep.")
    parser.add_argument(
        "--mean-connectivity",
        type=float,
        nargs="+",
        help="Optional list of target mean connectivities (e.g., 0.1 0.3). Defaults to the base config.",
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
    parser.add_argument("--n-inits", type=int, default=50, help="Number of fixpoint initializations per kappa (default: %(default)s).")
    parser.add_argument("--seed-inits", type=int, default=0, help="Seed governing fixpoint sampling (default: %(default)s).")
    parser.add_argument("--seed-network", type=int, help="Seed for building the network connectivity.")
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
        default="plots/Figure4",
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


def _stat_summary(array: Any) -> Tuple[float, float]:
    values = np.asarray(array, dtype=float).ravel()
    if values.size == 0:
        return float("nan"), float("nan")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmedian(finite)), float(np.nanmean(finite))


def _build_instances(parameter: Dict[str, Any], targets: Sequence[float] | None) -> List[Tuple[Dict[str, Any], str]]:
    instances: List[Tuple[Dict[str, Any], str]] = []
    if targets:
        for target in targets:
            scaled = helpers.scale_connectivity(parameter, target)
            label = f"conn={float(target):.2f}"
            instances.append((deepcopy(scaled), label))
    else:
        base_conn = helpers.mean_connectivity(parameter)
        instances.append((deepcopy(parameter), f"conn={base_conn:.2f}"))
    return instances


def _append_result(
    store: Dict[str, Dict[str, List[float]]],
    label: str,
    kappa: float,
    summary: Dict[str, Any],
) -> None:
    target = store.setdefault(
        label,
        {
            "kappa": [],
            "state_within_median": [],
            "state_within_mean": [],
            "state_across_median": [],
            "state_across_mean": [],
            "field_within_median": [],
            "field_within_mean": [],
            "field_across_median": [],
            "field_across_mean": [],
        },
    )
    sw_med, sw_mean = _stat_summary(summary.get("state_excit_within", np.zeros(0)))
    sa_med, sa_mean = _stat_summary(summary.get("state_excit_between", np.zeros(0)))
    fw_med, fw_mean = _stat_summary(summary.get("field_excit_within", np.zeros(0)))
    fa_med, fa_mean = _stat_summary(summary.get("field_excit_between", np.zeros(0)))
    target["kappa"].append(float(kappa))
    target["state_within_median"].append(sw_med)
    target["state_within_mean"].append(sw_mean)
    target["state_across_median"].append(sa_med)
    target["state_across_mean"].append(sa_mean)
    target["field_within_median"].append(fw_med)
    target["field_within_mean"].append(fw_mean)
    target["field_across_median"].append(fa_med)
    target["field_across_mean"].append(fa_mean)


def _plot_metric_set(
    ax: plt.Axes,
    kappa: Sequence[float],
    median_values: Sequence[float],
    mean_values: Sequence[float],
    *,
    color: str,
    marker: str,
    label_prefix: str,
) -> None:
    ax.plot(kappa, median_values, color=color, linestyle="-", marker=marker, label=f"{label_prefix} median")
    ax.plot(
        kappa,
        mean_values,
        color=color,
        linestyle="--",
        marker=marker,
        fillstyle="none",
        label=f"{label_prefix} mean",
    )


def _plot_correlation_figure(
    results: Dict[str, Dict[str, List[float]]],
    *,
    kappa_bounds: Tuple[float, float],
    font_cfg: FontCfg,
    output_prefix: str,
) -> None:
    fig, (ax_state, ax_field) = plt.subplots(2, 1, sharex=True, figsize=(13, 9))
    within_color = "#4c72b0"
    across_color = "#dd8452"
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, (label, payload) in enumerate(sorted(results.items())):
        marker = markers[idx % len(markers)]
        kappa = payload["kappa"]
        _plot_metric_set(
            ax_state,
            kappa,
            payload["state_within_median"],
            payload["state_within_mean"],
            color=within_color,
            marker=marker,
            label_prefix=f"E within ({label})",
        )
        _plot_metric_set(
            ax_state,
            kappa,
            payload["state_across_median"],
            payload["state_across_mean"],
            color=across_color,
            marker=marker,
            label_prefix=f"E across ({label})",
        )
        _plot_metric_set(
            ax_field,
            kappa,
            payload["field_within_median"],
            payload["field_within_mean"],
            color=within_color,
            marker=marker,
            label_prefix=f"E within ({label})",
        )
        _plot_metric_set(
            ax_field,
            kappa,
            payload["field_across_median"],
            payload["field_across_mean"],
            color=across_color,
            marker=marker,
            label_prefix=f"E across ({label})",
        )
    ax_state.set_ylabel("State correlation")
    ax_field.set_ylabel("Field correlation")
    ax_field.set_xlabel(r"$\kappa$")
    ax_state.set_title("State correlations")
    ax_field.set_title("Subthreshold correlations")
    ax_field.set_xlim(*kappa_bounds)
    style_axes(ax_state, font_cfg)
    style_axes(ax_field, font_cfg)
    add_panel_label(ax_state, "a", font_cfg)
    add_panel_label(ax_field, "b", font_cfg)
    handles, labels = ax_state.get_legend_handles_labels()
    if handles:
        ax_state.legend(handles, labels, loc="best", fontsize=font_cfg.legend)
    fig.tight_layout()
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Stored Figure 4 at {png_path} and {pdf_path}")


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    focus_counts = helpers.resolve_focus_counts(parameter, args.focus_counts)
    kappa_values = _kappa_sequence(float(args.kappa_start), float(args.kappa_stop), float(args.kappa_step))
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
    instances = _build_instances(parameter, args.mean_connectivity)
    results: Dict[str, Dict[str, List[float]]] = {}
    for idx, (base_param, label) in enumerate(instances):
        print(f"=== Correlation workflow for {label} ===")
        for kappa in kappa_values:
            param_copy = deepcopy(base_param)
            param_copy["kappa"] = float(kappa)
            r_value = param_copy.get("R_Eplus")
            if r_value is None:
                raise ValueError(
                    "Parameter 'R_Eplus' must be defined (or overridden via -O) for the correlation workflow."
                )
            print(f"  -> kappa = {kappa:.4f}, R_Eplus = {float(r_value):.4f}")
            folder, bundle_path = helpers.ensure_fixpoint_bundle(
                param_copy,
                focus_counts,
                [float(r_value)],
                sweep_cfg,
            )
            binary_cfg = helpers.resolve_binary_config(param_copy, binary_overrides)
            seed_network = args.seed_network if args.seed_network is not None else int(binary_cfg.get("seed", 0) or 0)
            corr_result = helpers.run_multi_init_correlation(
                param_copy,
                binary_cfg,
                folder_hint=folder,
                bundle_path=bundle_path,
                focus_counts=focus_counts,
                stability_filter=args.stability_filter,
                n_inits=max(0, int(args.n_inits)),
                seed_inits=int(args.seed_inits),
                seed_network=seed_network,
                stride_analysis=max(1, int(args.stride_analysis or 1)),
                max_pairs=max(1, int(args.max_pairs or DEFAULT_MAX_PAIRS)),
                jobs=max(1, int(args.jobs or 1)),
                analysis_only=args.analysis_only,
                overwrite_simulation=args.overwrite_simulation,
                overwrite_analysis=args.overwrite_analysis,
            )
            _append_result(results, label, float(kappa), corr_result.summary)
    kappa_bounds = (min(kappa_values), max(kappa_values))
    _plot_correlation_figure(results, kappa_bounds=kappa_bounds, font_cfg=font_cfg, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
