from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from sim_config import write_yaml_config

from .methods import run_changepoint_kmeans, run_hmm, run_kmeans_filter
from .preprocessing import build_feature_matrix
from .types import AnalysisInput, StateInferenceResult


SWEEP_METHODS = {
    "kmeans_filter": run_kmeans_filter,
    "changepoint_kmeans": run_changepoint_kmeans,
    "hmm": run_hmm,
}


def gaussian_sigma_bins_to_cutoff_hz(sigma_bins: float, dt: float) -> float:
    """Approximate Gaussian 3 dB cutoff frequency for a temporal smoothing sigma."""

    sigma = float(sigma_bins)
    if sigma <= 0:
        return float("inf")
    dt_seconds = float(dt) / 1000.0 if float(dt) > 1.0 else float(dt)
    sigma_seconds = sigma * dt_seconds
    return float(np.sqrt(np.log(2.0)) / (2.0 * np.pi * sigma_seconds))


def detect_knee(x: Iterable[float], y: Iterable[float]) -> Dict[str, Any]:
    """Detect an elbow/knee by maximum distance to the endpoint chord."""

    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.size < 3 or np.allclose(y_arr, y_arr[0]):
        idx = 0 if x_arr.size else None
        return {"index": idx, "x": None if idx is None else float(x_arr[idx]), "strength": 0.0}
    x_norm = (x_arr - x_arr.min()) / max(x_arr.max() - x_arr.min(), 1e-12)
    y_norm = (y_arr - y_arr.min()) / max(y_arr.max() - y_arr.min(), 1e-12)
    line_x = np.array([x_norm[0], x_norm[-1]], dtype=float)
    line_y = np.array([y_norm[0], y_norm[-1]], dtype=float)
    points = np.column_stack([x_norm, y_norm])
    line_vec = np.array([line_x[1] - line_x[0], line_y[1] - line_y[0]], dtype=float)
    line_len = np.linalg.norm(line_vec)
    if line_len <= 0:
        idx = int(np.argmax(np.abs(y_norm - y_norm.mean())))
        return {"index": idx, "x": float(x_arr[idx]), "strength": 0.0}
    distances = []
    for point in points:
        offset = point - np.array([line_x[0], line_y[0]])
        area = abs(line_vec[0] * offset[1] - line_vec[1] * offset[0])
        distances.append(area / line_len)
    idx = int(np.argmax(distances))
    return {"index": idx, "x": float(x_arr[idx]), "strength": float(distances[idx])}


def _hmm_parameter_count(emission: str, n_states: int, n_features: int) -> int:
    base = max(0, n_states - 1) + max(0, n_states * (n_states - 1))
    emission = str(emission).lower()
    if emission == "gaussian":
        return base + 2 * n_states * n_features
    if emission in {"poisson", "bernoulli"}:
        return base + n_states * n_features
    return base + 2 * n_states * n_features


def _silhouette_or_nan(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(np.asarray(labels, dtype=np.int64))
    if unique.size <= 1 or unique.size >= features.shape[0]:
        return float("nan")
    try:
        return float(silhouette_score(features, labels))
    except Exception:
        return float("nan")


def _result_row(
    method: str,
    sigma_bins: float,
    dt: float,
    requested_n_states: int,
    result: StateInferenceResult,
    features: np.ndarray,
    method_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    occupancy = np.asarray(list(result.state_occupancy.values()), dtype=float)
    dominant = float(occupancy.max()) if occupancy.size else float("nan")
    row = {
        "method": method,
        "sigma_bins": float(sigma_bins),
        "cutoff_hz": gaussian_sigma_bins_to_cutoff_hz(float(sigma_bins), dt),
        "requested_n_states": int(requested_n_states),
        "effective_n_states": int(result.n_states),
        "n_segments": int(len(result.segments)),
        "dominant_occupancy": dominant,
        "silhouette": _silhouette_or_nan(features, result.labels),
        "log_likelihood": float(result.log_likelihood) if result.log_likelihood is not None else float("nan"),
        "inertia": float(result.metadata.get("inertia", result.metadata.get("segment_kmeans_inertia", float("nan")))),
        "bic": float("nan"),
        "aic": float("nan"),
    }
    if method == "hmm" and result.log_likelihood is not None:
        emission = str(method_cfg.get("emission", "auto"))
        n_params = _hmm_parameter_count(emission, int(result.n_states), int(features.shape[1]))
        n_obs = int(features.shape[0])
        row["bic"] = float(np.log(max(n_obs, 2)) * n_params - 2.0 * float(result.log_likelihood))
        row["aic"] = float(2.0 * n_params - 2.0 * float(result.log_likelihood))
    return row


def _recommended_rows(frame: pd.DataFrame) -> Dict[str, Any]:
    recommendations: Dict[str, Any] = {}
    for method in sorted(frame["method"].unique()):
        subset = frame[frame["method"] == method].copy()
        method_summary: Dict[str, Any] = {"per_sigma": []}
        for sigma in sorted(subset["sigma_bins"].unique()):
            sigma_df = subset[subset["sigma_bins"] == sigma].sort_values("requested_n_states")
            entry: Dict[str, Any] = {
                "sigma_bins": float(sigma),
                "cutoff_hz": float(sigma_df["cutoff_hz"].iloc[0]),
            }
            if method in {"kmeans_filter", "changepoint_kmeans"}:
                knee = detect_knee(sigma_df["requested_n_states"], sigma_df["inertia"])
                best_silhouette_idx = int(sigma_df["silhouette"].fillna(-np.inf).argmax())
                best_silhouette_row = sigma_df.iloc[best_silhouette_idx]
                entry["recommended_n_states_knee"] = None if knee["x"] is None else int(knee["x"])
                entry["knee_strength"] = float(knee["strength"])
                entry["best_silhouette_n_states"] = int(best_silhouette_row["requested_n_states"])
                entry["best_silhouette"] = float(best_silhouette_row["silhouette"])
            elif method == "hmm":
                bic_row = sigma_df.loc[sigma_df["bic"].idxmin()]
                aic_row = sigma_df.loc[sigma_df["aic"].idxmin()]
                entry["recommended_n_states_bic"] = int(bic_row["requested_n_states"])
                entry["recommended_n_states_aic"] = int(aic_row["requested_n_states"])
                entry["best_bic"] = float(bic_row["bic"])
                entry["best_aic"] = float(aic_row["aic"])
            method_summary["per_sigma"].append(entry)
        recommendations[method] = method_summary
    return recommendations


def _save_metric_plot(frame: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in sorted(frame["sigma_bins"].unique()):
        subset = frame[frame["sigma_bins"] == sigma].sort_values("requested_n_states")
        label = f"sigma={sigma:g}"
        ax.plot(subset["requested_n_states"], subset[metric], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Requested number of states")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_state_count_sweep(
    data: AnalysisInput,
    analysis_cfg: Dict[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Sweep smoothing levels and state counts to assess stable model complexity."""

    sweep_cfg = dict(analysis_cfg.get("state_count_sweep") or {})
    if not bool(sweep_cfg.get("enabled", False)):
        return {}
    sigma_values = [float(value) for value in sweep_cfg.get("smoothing_sigma_bins", [0, 1, 2, 4])]
    n_state_values = [int(value) for value in sweep_cfg.get("n_states", [2, 3, 4, 5, 6, 8])]
    methods = [name for name in sweep_cfg.get("methods", ["kmeans_filter", "changepoint_kmeans", "hmm"]) if name in SWEEP_METHODS]
    preprocessing_base = dict(analysis_cfg.get("preprocessing") or {})
    methods_cfg = dict(analysis_cfg.get("methods") or {})

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for sigma in sigma_values:
            preprocessing_cfg = deepcopy(preprocessing_base)
            preprocessing_cfg["smoothing_sigma_bins"] = float(sigma)
            feature_type = str((methods_cfg.get(method) or {}).get("feature_type", "smoothed_rates"))
            features = build_feature_matrix(data, feature_type, preprocessing_cfg)
            for n_states in n_state_values:
                method_cfg = deepcopy(methods_cfg.get(method) or {})
                method_cfg["enabled"] = True
                method_cfg["n_states"] = int(n_states)
                result = SWEEP_METHODS[method](data, preprocessing_cfg, method_cfg)
                rows.append(_result_row(method, sigma, float(data.dt), int(n_states), result, features, method_cfg))

    frame = pd.DataFrame(rows)
    recommendations = _recommended_rows(frame)

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / "state_count_sweep.csv", index=False)
        write_yaml_config(sweep_cfg, root / "state_count_sweep_config.yaml")
        (root / "state_count_recommendations.json").write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        for method in methods:
            method_frame = frame[frame["method"] == method]
            if method in {"kmeans_filter", "changepoint_kmeans"}:
                _save_metric_plot(method_frame, "inertia", root / f"{method}_inertia_vs_states.png", f"{method} inertia sweep")
                _save_metric_plot(method_frame, "silhouette", root / f"{method}_silhouette_vs_states.png", f"{method} silhouette sweep")
            elif method == "hmm":
                _save_metric_plot(method_frame, "bic", root / f"{method}_bic_vs_states.png", f"{method} BIC sweep")
                _save_metric_plot(method_frame, "log_likelihood", root / f"{method}_loglik_vs_states.png", f"{method} log-likelihood sweep")

    return {
        "table": frame,
        "recommendations": recommendations,
        "output_dir": None if output_dir is None else str(Path(output_dir)),
    }
