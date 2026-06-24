#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.io import analysis_input_from_binary_trace
from analysis.methods import (
    _bounds_to_breakpoints,
    _changepoint_bounds,
    _finalize_result,
    _fit_template_labels,
    _merge_bounds_by_template_and_dwell,
    _reference_matrix,
    _resolve_cluster_indices,
    _run_pelt_crops,
)
from analysis.preprocessing import build_feature_matrix
from scripts.evaluate_changepoint_kmeans_binary import _state_quality


def _base_analysis_cfg() -> dict[str, Any]:
    return {
        "preprocessing": {
            "smoothing_sigma_bins": 1.0,
            "binary_threshold_mode": "percentile",
            "binary_threshold_percentile": 80.0,
            "hysteresis": {"enabled": False},
        }
    }


def _method_cfg(threshold: float, min_dwell_bins: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "algorithm": "pelt_crops",
        "cost": "rbf",
        "feature_type": "smoothed_rates",
        "n_states": 20,
        "n_init": 30,
        "max_iter": 300,
        "min_dwell_bins": 1,
        "merge_strategy": "nearest",
        "segment_features": {
            "mean": True,
            "max": False,
            "active_fraction": True,
            "start_end_difference": False,
            "duration": True,
        },
        "random_state": 0,
        "crops_min_penalty": 0.5,
        "crops_max_penalty": 300.0,
        "crops_n_penalties": 30,
        "crops_selection": "elbow",
        "crops_max_changepoints_for_elbow": 50,
        "crops_quality_source": "excitatory",
        "merge_adjacent_segments": False,
        "template_state_assignment": True,
        "template_source": "excitatory",
        "template_merge_adjacent": True,
        "template_threshold_mode": "relative",
        "template_fixed_threshold": float(threshold),
        "template_threshold_percentile": 80.0,
        "template_min_active_clusters": 1,
        "template_min_dwell_bins": int(min_dwell_bins),
    }


def _score_row(n_cps: int, label: str) -> tuple[bool | None, bool]:
    label = str(label).strip()
    if label == "0":
        return n_cps == 0, n_cps == 0
    if label == "1":
        return n_cps == 1, n_cps == 1
    if label == "2":
        return n_cps == 2, n_cps == 2
    if label in {"gt1", ">1", ">1_hard"}:
        return None, n_cps > 1
    if label in {"gt5", ">5"}:
        return None, n_cps > 5
    if label in {"gt10", ">10"}:
        return None, n_cps > 10
    return None, False


def evaluate(labels_path: Path, output_dir: Path, thresholds: list[float], min_dwell_bins: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    label_col = "eye_label" if "eye_label" in labels.columns else "human_cp_category"
    analysis_cfg = _base_analysis_cfg()
    parameter = {"Q": 20, "N_E": 8000, "N_I": 2000}
    all_rows: list[dict[str, Any]] = []
    for _, label_row in labels.iterrows():
        data = analysis_input_from_binary_trace(
            label_row["trace_path"],
            parameter=parameter,
            analysis_cfg=analysis_cfg,
        )
        base_cfg = _method_cfg(thresholds[0], min_dwell_bins)
        features = build_feature_matrix(data, base_cfg["feature_type"], analysis_cfg["preprocessing"])
        quality_indices = _resolve_cluster_indices(data, str(base_cfg["crops_quality_source"]))
        quality_matrix = _reference_matrix(data)[:, quality_indices]
        bkps, crops_metadata = _run_pelt_crops(features, base_cfg["cost"], base_cfg, quality_matrix=quality_matrix)
        base_bounds = _changepoint_bounds([int(value) for value in bkps])
        for threshold in thresholds:
            method_cfg = _method_cfg(threshold, min_dwell_bins)
            template_indices = _resolve_cluster_indices(data, str(method_cfg["template_source"]))
            template_matrix = _reference_matrix(data)[:, template_indices]
            bounds, template_merge_metadata = _merge_bounds_by_template_and_dwell(
                list(base_bounds),
                template_matrix,
                method_cfg,
            )
            labels_arr, template_label_metadata = _fit_template_labels(data, bounds, template_matrix, method_cfg)
            breakpoints = _bounds_to_breakpoints(bounds)
            metadata = {
                "breakpoints": [int(value) for value in breakpoints],
                "feature_type": base_cfg["feature_type"],
                "segment_kmeans_inertia": 0.0,
                "segment_count": int(len(bounds)),
                "n_requested_states": int(method_cfg["n_states"]),
                "n_fitted_states": int(np.unique(labels_arr).size),
            }
            metadata.update(crops_metadata)
            metadata.update(template_merge_metadata)
            metadata.update(template_label_metadata)
            result = _finalize_result(
                "changepoint_kmeans",
                labels_arr,
                data,
                method_cfg,
                metadata=metadata,
            )
            quality = _state_quality(data, result)
            n_cps = max(0, len(breakpoints) - 1)
            eye_label = str(label_row[label_col])
            exact_known, satisfies_label = _score_row(n_cps, eye_label)
            all_rows.append(
                {
                    "threshold": float(threshold),
                    "template_min_dwell_bins": int(min_dwell_bins),
                    "simulation": str(label_row["simulation"]),
                    "montage_index": int(label_row["montage_index"]),
                    "seed": int(label_row["seed"]),
                    "kappa": float(label_row["kappa"]),
                    "R_j": float(label_row["R_j"]),
                    "eye_label": eye_label,
                    "trace_path": str(label_row["trace_path"]),
                    "n_changepoints": int(n_cps),
                    "n_segments": int(quality["n_segments"]),
                    "n_identified_states": int(quality["n_identified_states"]),
                    "breakpoints": ";".join(str(value) for value in breakpoints),
                    "exact_known": np.nan if exact_known is None else bool(exact_known),
                    "satisfies_eye_label": bool(satisfies_label),
                    "explained_variance_fraction": float(quality["explained_variance_fraction"]),
                    "silhouette": float(quality["silhouette"]) if np.isfinite(quality["silhouette"]) else np.nan,
                    "segment_count_before_template_merge": int(result.metadata.get("segment_count", quality["n_segments"])),
                    "template_merge_count": int(result.metadata.get("template_merge_count", 0)),
                    "crops_selected_penalty": float(result.metadata.get("crops_selected_penalty", np.nan)),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary.to_csv(output_dir / "human_eye_template_prior_sweep.csv", index=False)
    score = (
        summary.assign(is_known=summary["exact_known"].notna())
        .groupby("threshold", as_index=False)
        .agg(
            exact_known=("exact_known", lambda x: float(np.nanmean(x.astype(float)))),
            satisfies_all=("satisfies_eye_label", "mean"),
            mean_cps=("n_changepoints", "mean"),
            median_cps=("n_changepoints", "median"),
            mean_states=("n_identified_states", "mean"),
            mean_explained=("explained_variance_fraction", "mean"),
        )
        .sort_values(["satisfies_all", "exact_known", "mean_explained"], ascending=[False, False, False])
    )
    score.to_csv(output_dir / "human_eye_template_prior_sweep_scores.csv", index=False)
    best_threshold = float(score.iloc[0]["threshold"])
    best = summary[summary["threshold"] == best_threshold].copy()
    best.to_csv(output_dir / "human_eye_template_prior_best.csv", index=False)
    return score


def evaluate_from_candidates(
    candidates_path: Path,
    output_dir: Path,
    thresholds: list[float],
    min_dwell_bins: int,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path)
    analysis_cfg = _base_analysis_cfg()
    parameter = {"Q": 20, "N_E": 8000, "N_I": 2000}
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        data = analysis_input_from_binary_trace(
            candidate["trace_path"],
            parameter=parameter,
            analysis_cfg=analysis_cfg,
        )
        candidate_bkps = [int(value) for value in str(candidate["breakpoints"]).split(";") if str(value)]
        base_bounds = _changepoint_bounds(candidate_bkps)
        for threshold in thresholds:
            method_cfg = _method_cfg(threshold, min_dwell_bins)
            template_indices = _resolve_cluster_indices(data, str(method_cfg["template_source"]))
            template_matrix = _reference_matrix(data)[:, template_indices]
            bounds, template_merge_metadata = _merge_bounds_by_template_and_dwell(
                list(base_bounds),
                template_matrix,
                method_cfg,
            )
            labels_arr, template_label_metadata = _fit_template_labels(data, bounds, template_matrix, method_cfg)
            breakpoints = _bounds_to_breakpoints(bounds)
            metadata = {
                "breakpoints": [int(value) for value in breakpoints],
                "feature_type": method_cfg["feature_type"],
                "segment_kmeans_inertia": 0.0,
                "segment_count": int(len(bounds)),
                "n_requested_states": int(method_cfg["n_states"]),
                "n_fitted_states": int(np.unique(labels_arr).size),
            }
            metadata.update(template_merge_metadata)
            metadata.update(template_label_metadata)
            result = _finalize_result(
                "changepoint_kmeans",
                labels_arr,
                data,
                method_cfg,
                metadata=metadata,
            )
            quality = _state_quality(data, result)
            n_cps = max(0, len(breakpoints) - 1)
            eye_label = str(candidate["human_cp_category"])
            exact_known, satisfies_label = _score_row(n_cps, eye_label)
            rows.append(
                {
                    "threshold": float(threshold),
                    "template_min_dwell_bins": int(min_dwell_bins),
                    "simulation": str(candidate["simulation"]),
                    "montage_index": int(candidate["montage_index"]),
                    "seed": int(candidate["seed"]),
                    "kappa": float(candidate["kappa"]),
                    "R_j": float(candidate["R_j"]),
                    "R_Eplus": float(candidate["R_Eplus"]),
                    "eye_label": eye_label,
                    "trace_path": str(candidate["trace_path"]),
                    "candidate_n_changepoints": int(candidate["algorithm_cp"]),
                    "n_changepoints": int(n_cps),
                    "n_segments": int(quality["n_segments"]),
                    "n_identified_states": int(quality["n_identified_states"]),
                    "breakpoints": ";".join(str(value) for value in breakpoints),
                    "exact_known": np.nan if exact_known is None else bool(exact_known),
                    "satisfies_eye_label": bool(satisfies_label),
                    "explained_variance_fraction": float(quality["explained_variance_fraction"]),
                    "silhouette": float(quality["silhouette"]) if np.isfinite(quality["silhouette"]) else np.nan,
                    "template_merge_count": int(result.metadata.get("template_merge_count", 0)),
                    "candidate_selected_penalty": float(candidate.get("selected_penalty", np.nan)),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "human_eye_template_prior_candidate_sweep.csv", index=False)
    score = (
        summary.groupby("threshold", as_index=False)
        .agg(
            exact_known=("exact_known", lambda x: float(np.nanmean(x.astype(float)))),
            satisfies_all=("satisfies_eye_label", "mean"),
            mean_cps=("n_changepoints", "mean"),
            median_cps=("n_changepoints", "median"),
            mean_states=("n_identified_states", "mean"),
            mean_template_merges=("template_merge_count", "mean"),
            mean_explained=("explained_variance_fraction", "mean"),
        )
        .sort_values(["satisfies_all", "exact_known", "mean_explained"], ascending=[False, False, False])
    )
    score.to_csv(output_dir / "human_eye_template_prior_candidate_sweep_scores.csv", index=False)
    best_threshold = float(score.iloc[0]["threshold"])
    summary[summary["threshold"] == best_threshold].copy().to_csv(
        output_dir / "human_eye_template_prior_candidate_best.csv",
        index=False,
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("plots/changepoint_kmeans_figure3_fullsize_batch/human_eye_cp_labels.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/changepoint_kmeans_figure3_fullsize_batch"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.65])
    parser.add_argument("--min-dwell-bins", type=int, default=2)
    parser.add_argument("--candidate-breakpoints-csv", type=Path)
    args = parser.parse_args()
    if args.candidate_breakpoints_csv is None:
        score = evaluate(args.labels, args.output_dir, args.thresholds, args.min_dwell_bins)
    else:
        score = evaluate_from_candidates(
            args.candidate_breakpoints_csv,
            args.output_dir,
            args.thresholds,
            args.min_dwell_bins,
        )
    print(score.to_string(index=False))


if __name__ == "__main__":
    main()
