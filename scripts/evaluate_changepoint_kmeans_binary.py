from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.io import analysis_input_from_binary_trace
from analysis.pipeline import run_population_state_analysis
from pipelines.binary import run_binary_simulation
from sim_config import load_config, sim_tag_from_cfg, write_yaml_config


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")


def _parse_float_grid(values: Iterable[str] | None, default: List[float]) -> List[float]:
    if not values:
        return default
    parsed: List[float] = []
    for item in values:
        if ":" not in item:
            parsed.append(float(item))
            continue
        start, stop, step = (float(part) for part in item.split(":"))
        if step == 0:
            raise ValueError("Range step must be non-zero.")
        count = int(np.floor((stop - start) / step)) + 1
        parsed.extend(float(start + idx * step) for idx in range(max(count, 0)))
    return parsed


def _base_parameter(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    parameter = deepcopy(cfg)
    parameter.update(
        {
            "N_E": int(args.n_exc),
            "N_I": int(args.n_inh),
            "Q": int(args.clusters),
            "R_j": float(args.r_j),
            "kappa": float(args.kappa),
            "connection_type": str(args.connection_type),
            "collapse_types": False,
        }
    )
    if parameter["N_E"] % parameter["Q"] != 0 or parameter["N_I"] % parameter["Q"] != 0:
        raise ValueError("N_E and N_I must both be divisible by Q.")
    return parameter


def _binary_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "warmup_steps": int(args.warmup_steps),
        "simulation_steps": int(args.simulation_steps),
        "sample_interval": int(args.sample_interval),
        "batch_size": int(args.batch_size),
        "state_chunk_size": 0,
        "population_rate_init": float(args.population_rate_init),
        "output_name": "activity_trace",
        "plot_activity": True,
        "weight_mode": str(args.weight_mode),
        "ram_budget_gb": float(args.ram_budget_gb),
        "weight_dtype": "float32",
    }


def _analysis_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "enabled": True,
        "source_type": "binary",
        "preprocessing": {
            "use_counts": False,
            "use_rates": True,
            "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
            "sqrt_transform": False,
            "zscore": True,
            "binary_threshold_mode": "percentile",
            "binary_threshold_percentile": 80,
            "hysteresis": {"enabled": False},
            "temporal_window_bins": 0,
        },
        "methods": {
            "threshold_filter": {"enabled": False},
            "kmeans_filter": {"enabled": False},
            "changepoint_kmeans": {
                "enabled": True,
                "algorithm": str(args.changepoint_algorithm),
                "cost": str(args.changepoint_cost),
                "penalty": None if args.penalty is None else float(args.penalty),
                "n_bkps": None,
                "n_states": int(args.n_states),
                "feature_type": "smoothed_rates",
                "n_init": 30,
                "max_iter": 300,
                "min_dwell_bins": int(args.min_dwell_bins),
                "merge_strategy": "nearest",
                "segment_features": {
                    "mean": True,
                    "max": False,
                    "active_fraction": True,
                    "start_end_difference": False,
                    "duration": True,
                },
                "random_state": 0,
                "crops_min_penalty": float(args.crops_min_penalty),
                "crops_max_penalty": float(args.crops_max_penalty),
                "crops_n_penalties": int(args.crops_n_penalties),
                "crops_selection": str(args.crops_selection),
                "crops_target_segments": args.crops_target_segments,
                "crops_max_changepoints_for_elbow": args.crops_max_changepoints_for_elbow,
                "crops_min_adjacent_distance": args.crops_min_adjacent_distance,
                "crops_min_loss_reduction": args.crops_min_loss_reduction,
                "crops_min_selected_penalty": args.crops_min_selected_penalty,
                "crops_quality_source": str(args.crops_quality_source),
                "merge_adjacent_segments": bool(args.merge_adjacent_segments),
                "merge_min_adjacent_distance": args.merge_min_adjacent_distance,
                "merge_min_dwell_bins": args.merge_min_dwell_bins,
                "template_state_assignment": bool(args.template_state_assignment),
                "template_source": str(args.template_source),
                "template_merge_adjacent": bool(args.template_merge_adjacent),
                "template_threshold_mode": str(args.template_threshold_mode),
                "template_threshold_percentile": float(args.template_threshold_percentile),
                "template_fixed_threshold": args.template_fixed_threshold,
                "template_min_active_clusters": int(args.template_min_active_clusters),
                "template_min_dwell_bins": args.template_min_dwell_bins,
            },
            "hmm": {"enabled": False},
        },
        "evaluation": {
            "compute_ground_truth_metrics": False,
            "compute_method_agreement": False,
        },
        "plotting": {
            "enabled": True,
            "save_format": "png",
            "dpi": int(args.dpi),
        },
        "state_count_sweep": {"enabled": False},
    }


def _rates_per_state_table(result, names: List[str]) -> pd.DataFrame:
    rows = []
    means = np.asarray(result.state_means, dtype=float)
    for state in range(means.shape[0]):
        row = {"state": int(state)}
        row.update({name: float(value) for name, value in zip(names, means[state])})
        rows.append(row)
    return pd.DataFrame(rows)


def _dwell_table(result) -> pd.DataFrame:
    rows = []
    for state, values in result.dwell_times.items():
        arr = np.asarray(values, dtype=float)
        for value in arr:
            rows.append({"state": int(state), "duration_time": float(value)})
    return pd.DataFrame(rows, columns=["state", "duration_time"])


def _state_quality(data, result) -> Dict[str, Any]:
    X = np.asarray(data.preferred_matrix(), dtype=float)
    labels = np.asarray(result.labels, dtype=np.int64)
    unique = np.unique(labels)
    means = np.asarray(result.state_means, dtype=float)
    recon = means[labels]
    total_var = float(np.mean((X - X.mean(axis=0, keepdims=True)) ** 2)) if X.size else 0.0
    residual_var = float(np.mean((X - recon) ** 2)) if X.size else 0.0
    explained = 1.0 - residual_var / total_var if total_var > 0 else 0.0
    silhouette = np.nan
    if unique.size > 1 and unique.size < labels.size:
        try:
            silhouette = float(silhouette_score(X, labels))
        except ValueError:
            silhouette = np.nan
    pairwise = []
    for idx in range(means.shape[0]):
        for jdx in range(idx + 1, means.shape[0]):
            pairwise.append(float(np.linalg.norm(means[idx] - means[jdx])))
    dwell_counts = {int(state): int(np.asarray(values).size) for state, values in result.dwell_times.items()}
    occupancy = {int(key): float(value) for key, value in result.state_occupancy.items()}
    degenerate = [state for state, frac in occupancy.items() if frac < 0.01]
    return {
        "n_identified_states": int(unique.size),
        "n_segments": int(result.segments.shape[0]),
        "explained_variance_fraction": float(explained),
        "residual_variance": residual_var,
        "silhouette": silhouette,
        "min_state_distance": float(np.min(pairwise)) if pairwise else np.nan,
        "median_state_distance": float(np.median(pairwise)) if pairwise else np.nan,
        "dwell_count_min": int(min(dwell_counts.values())) if dwell_counts else 0,
        "degenerate_states": degenerate,
        "occupancy": occupancy,
    }


def _plot_batch_quality(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    x = np.arange(summary.shape[0])
    axes[0].bar(x, summary["explained_variance_fraction"], color="#2A7F62")
    axes[0].set_ylabel("Explained variance")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].bar(x, summary["silhouette"], color="#415A77")
    axes[1].set_ylabel("Silhouette")
    axes[2].bar(x, summary["n_segments"], color="#A23E48")
    axes[2].set_ylabel("Segments")
    axes[2].set_xlabel("Simulation")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(summary["simulation"].astype(str), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "batch_quality_summary.png", dpi=dpi)
    plt.close(fig)


def _plot_kappa_summary(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if summary.empty or "kappa" not in summary:
        return
    grouped = (
        summary.groupby("kappa", as_index=False)
        .agg(
            mean_explained=("explained_variance_fraction", "mean"),
            mean_states=("n_identified_states", "mean"),
            mean_segments=("n_segments", "mean"),
            mean_silhouette=("silhouette", "mean"),
        )
        .sort_values("kappa")
    )
    grouped.to_csv(output_dir / "kappa_summary.csv", index=False)
    fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    axes[0].plot(grouped["kappa"], grouped["mean_explained"], marker="o", color="#2A7F62")
    axes[0].set_ylabel("Explained variance")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].plot(grouped["kappa"], grouped["mean_silhouette"], marker="o", color="#415A77")
    axes[1].set_ylabel("Silhouette")
    axes[2].plot(grouped["kappa"], grouped["mean_states"], marker="o", color="#6D597A")
    axes[2].set_ylabel("States")
    axes[3].plot(grouped["kappa"], grouped["mean_segments"], marker="o", color="#A23E48")
    axes[3].set_ylabel("Segments")
    axes[3].set_xlabel("kappa")
    fig.tight_layout()
    fig.savefig(output_dir / "kappa_quality_summary.png", dpi=dpi)
    plt.close(fig)


def _verdict(summary: pd.DataFrame, requested_states: int) -> Dict[str, Any]:
    if summary.empty:
        return {"verdict": "insufficient_data", "reason": "No successful simulations."}
    state_match = summary["n_identified_states"] == requested_states
    nondegenerate = summary["degenerate_state_count"] == 0
    enough_dwells = summary["dwell_count_min"] >= 2
    explained = summary["explained_variance_fraction"] >= 0.65
    adequate = state_match & nondegenerate & enough_dwells & explained
    pass_fraction = float(adequate.mean())
    verdict = "good_and_sufficient" if pass_fraction >= 0.8 else "mixed_or_insufficient"
    return {
        "verdict": verdict,
        "pass_fraction": pass_fraction,
        "criteria": {
            "identified_requested_state_count": float(state_match.mean()),
            "no_degenerate_states": float(nondegenerate.mean()),
            "at_least_two_dwells_per_state": float(enough_dwells.mean()),
            "explained_variance_at_least_0.65": float(explained.mean()),
        },
        "note": (
            "These are unsupervised diagnostics because the binary network simulations do not provide ground-truth "
            "state labels. Inspect per-simulation activity/state plots before treating the verdict as final."
        ),
    }


def run_batch(args: argparse.Namespace) -> Path:
    parameter = _base_parameter(args)
    binary_template = _binary_cfg(args)
    analysis_cfg = _analysis_cfg(args)
    r_eplus_values = _parse_float_grid(args.r_eplus, [float(parameter.get("R_Eplus") or 6.0)])
    kappa_values = _parse_float_grid(args.kappas, [float(args.kappa)])
    r_j_values = _parse_float_grid(args.r_js, [float(args.r_j)])
    if not r_eplus_values:
        raise ValueError("At least one R_Eplus value is required.")
    seeds = [int(args.seed_start) + idx for idx in range(int(args.n_simulations))]
    conditions = [
        (float(kappa), float(r_j), float(r_eplus))
        for kappa in kappa_values
        for r_j in r_j_values
        for r_eplus in r_eplus_values
    ]
    if not conditions:
        raise ValueError("At least one condition is required.")
    batch_cfg = {
        "parameter": parameter,
        "binary": binary_template,
        "analysis": analysis_cfg,
        "r_eplus_values": r_eplus_values,
        "kappa_values": kappa_values,
        "r_j_values": r_j_values,
        "seeds": seeds,
    }
    output_dir = Path(args.output_dir) / sim_tag_from_cfg(batch_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_config(batch_cfg, output_dir / "batch_config.yaml")

    rows = []
    failures = []
    for sim_idx, seed in enumerate(seeds):
        kappa, r_j, r_eplus = conditions[sim_idx % len(conditions)]
        sim_parameter = deepcopy(parameter)
        sim_parameter["kappa"] = float(kappa)
        sim_parameter["R_j"] = float(r_j)
        sim_parameter["R_Eplus"] = float(r_eplus)
        sim_parameter["analysis"] = analysis_cfg
        binary_cfg = deepcopy(binary_template)
        binary_cfg["seed"] = int(seed)
        run_name = f"sim_{sim_idx:02d}_kappa_{kappa:.3g}_rj_{r_j:.3g}_seed_{seed}".replace(".", "_")
        try:
            simulation = run_binary_simulation(sim_parameter, binary_cfg, output_name=run_name)
            data = analysis_input_from_binary_trace(
                simulation["trace_path"],
                parameter=sim_parameter,
                analysis_cfg=analysis_cfg,
            )
            analysis = run_population_state_analysis(
                data,
                analysis_cfg,
                output_dir=output_dir / run_name / "analysis",
            )
            result = analysis["results"]["changepoint_kmeans"]
            method_dir = output_dir / run_name / "analysis" / "changepoint_kmeans"
            names = data.cluster_names or [f"C{idx + 1}" for idx in range(data.n_clusters)]
            _rates_per_state_table(result, names).to_csv(method_dir / "rates_per_state.csv", index=False)
            dwell = _dwell_table(result)
            dwell.to_csv(method_dir / "dwell_times_by_state.csv", index=False)
            quality = _state_quality(data, result)
            _write_json(method_dir / "state_quality.json", quality)
            rows.append(
                {
                    "simulation": run_name,
                    "seed": int(seed),
                    "kappa": float(sim_parameter["kappa"]),
                    "R_j": float(sim_parameter["R_j"]),
                    "R_Eplus": float(sim_parameter["R_Eplus"]),
                    "trace_path": simulation["trace_path"],
                    "analysis_dir": str((output_dir / run_name / "analysis").resolve()),
                    "n_identified_states": quality["n_identified_states"],
                    "n_segments": quality["n_segments"],
                    "explained_variance_fraction": quality["explained_variance_fraction"],
                    "silhouette": quality["silhouette"],
                    "min_state_distance": quality["min_state_distance"],
                    "median_state_distance": quality["median_state_distance"],
                    "dwell_count_min": quality["dwell_count_min"],
                    "degenerate_state_count": len(quality["degenerate_states"]),
                }
            )
        except Exception as exc:  # pragma: no cover - batch resilience path
            failures.append({"simulation": run_name, "seed": int(seed), "error": repr(exc)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "simulation_summary.csv", index=False)
    _write_json(output_dir / "failures.json", {"failures": failures})
    verdict = _verdict(summary, int(args.n_states))
    _write_json(output_dir / "verdict.json", verdict)
    _plot_batch_quality(summary, output_dir, int(args.dpi))
    _plot_kappa_summary(summary, output_dir, int(args.dpi))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate changepoint + k-means state inference on repeated clustered binary-network simulations."
    )
    parser.add_argument("--config", default="default_simulation", help="Base simulation config name or YAML path.")
    parser.add_argument("--n-simulations", type=int, default=20, help="Number of binary simulations to run.")
    parser.add_argument("--seed-start", type=int, default=100, help="First random seed; later runs increment by one.")
    parser.add_argument("--r-eplus", nargs="*", help="R_Eplus values or start:stop:step ranges. Defaults to 6.0.")
    parser.add_argument("--output-dir", default="plots/changepoint_kmeans_binary_batch", help="Batch output directory.")
    parser.add_argument("--n-exc", type=int, default=320, help="Number of excitatory neurons for validation runs.")
    parser.add_argument("--n-inh", type=int, default=80, help="Number of inhibitory neurons for validation runs.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of E/I clusters.")
    parser.add_argument("--r-j", type=float, default=0.8, help="Inhibitory clustering factor.")
    parser.add_argument("--r-js", nargs="*", help="R_j values or start:stop:step ranges. Defaults to --r-j.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Weight/probability mixing exponent.")
    parser.add_argument("--kappas", nargs="*", help="kappa values or start:stop:step ranges. Defaults to --kappa.")
    parser.add_argument("--connection-type", default="poisson", choices=["bernoulli", "poisson", "fixed-indegree"])
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--simulation-steps", type=int, default=50000)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--population-rate-init", type=float, default=0.1)
    parser.add_argument("--weight-mode", default="auto", choices=["auto", "dense", "sparse"])
    parser.add_argument("--ram-budget-gb", type=float, default=4.0)
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--changepoint-algorithm", default="pelt", choices=["pelt", "pelt_crops", "crops", "binseg", "window"])
    parser.add_argument("--changepoint-cost", default="rbf", choices=["rbf", "l1", "l2", "normal", "linear"])
    parser.add_argument(
        "--penalty",
        type=float,
        default=20.0,
        help=(
            "PELT penalty. The generic analysis default is log(n_samples) * n_features, "
            "which is too conservative for Figure3-like traces sampled at population level."
        ),
    )
    parser.add_argument("--min-dwell-bins", type=int, default=2)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--crops-min-penalty", type=float, default=0.5)
    parser.add_argument("--crops-max-penalty", type=float, default=300.0)
    parser.add_argument("--crops-n-penalties", type=int, default=40)
    parser.add_argument("--crops-selection", default="elbow", choices=["elbow", "min_penalty", "max_penalty"])
    parser.add_argument("--crops-target-segments", type=int)
    parser.add_argument("--crops-max-changepoints-for-elbow", type=int, default=20)
    parser.add_argument("--crops-min-adjacent-distance", type=float)
    parser.add_argument("--crops-min-loss-reduction", type=float)
    parser.add_argument("--crops-min-selected-penalty", type=float)
    parser.add_argument("--crops-quality-source", default="excitatory", choices=["all", "excitatory", "e"])
    parser.add_argument("--merge-adjacent-segments", action="store_true")
    parser.add_argument("--merge-min-adjacent-distance", type=float)
    parser.add_argument("--merge-min-dwell-bins", type=int)
    parser.add_argument("--template-state-assignment", action="store_true")
    parser.add_argument("--template-source", default="excitatory", choices=["all", "excitatory", "e"])
    parser.add_argument("--template-merge-adjacent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--template-threshold-mode", default="relative", choices=["relative", "percentile", "fixed"])
    parser.add_argument("--template-threshold-percentile", type=float, default=80.0)
    parser.add_argument("--template-fixed-threshold", type=float)
    parser.add_argument("--template-min-active-clusters", type=int, default=1)
    parser.add_argument("--template-min-dwell-bins", type=int)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    output_dir = run_batch(parse_args())
    print(f"Wrote changepoint+k-means validation batch to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
