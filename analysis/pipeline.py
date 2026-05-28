from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from sim_config import write_yaml_config

from .evaluation import compare_results, evaluate_result
from .io import (
    analysis_input_from_binary_trace,
    analysis_input_from_spiking_payload,
    load_analysis_input,
    resolve_analysis_output_dir,
)
from .methods import run_changepoint_kmeans, run_hmm, run_kmeans_filter, run_threshold_filter
from .model_selection import run_state_count_sweep
from .plotting import plot_method_comparison, save_result_plots
from .preprocessing import apply_cluster_selection, apply_population_filter, validate_analysis_input
from .types import AnalysisInput, StateInferenceResult


METHOD_REGISTRY = {
    "threshold_filter": run_threshold_filter,
    "kmeans_filter": run_kmeans_filter,
    "changepoint_kmeans": run_changepoint_kmeans,
    "hmm": run_hmm,
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")


def _save_result(method_dir: Path, result: StateInferenceResult, metrics: Optional[Dict[str, Any]]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "labels.npy", np.asarray(result.labels, dtype=np.int64))
    np.savez_compressed(
        method_dir / "state_inference_result.npz",
        labels=np.asarray(result.labels, dtype=np.int64),
        transition_matrix=np.asarray(result.transition_matrix, dtype=float),
        state_means=np.asarray(result.state_means, dtype=float),
        state_templates=np.asarray(result.state_templates, dtype=np.uint8) if result.state_templates is not None else np.zeros((0, 0), dtype=np.uint8),
        posterior_probs=np.asarray(result.posterior_probs, dtype=float) if result.posterior_probs is not None else np.zeros((0, 0), dtype=float),
    )
    result.segments.to_csv(method_dir / "segments.csv", index=False)
    dwell_rows = []
    for state, values in result.dwell_times.items():
        for value in np.asarray(values, dtype=float):
            dwell_rows.append({"state": int(state), "duration_time": float(value)})
    pd.DataFrame(dwell_rows, columns=["state", "duration_time"]).to_csv(method_dir / "dwell_times.csv", index=False)
    np.save(method_dir / "transition_matrix.npy", np.asarray(result.transition_matrix, dtype=float))
    np.savetxt(method_dir / "transition_matrix.csv", np.asarray(result.transition_matrix, dtype=float), delimiter=",")
    np.save(method_dir / "state_means.npy", np.asarray(result.state_means, dtype=float))
    np.savetxt(method_dir / "state_means.csv", np.asarray(result.state_means, dtype=float), delimiter=",")
    if result.state_templates is not None:
        np.save(method_dir / "state_templates.npy", np.asarray(result.state_templates, dtype=np.uint8))
        np.savetxt(method_dir / "state_templates.csv", np.asarray(result.state_templates, dtype=np.uint8), delimiter=",", fmt="%d")
    if result.posterior_probs is not None:
        np.save(method_dir / "posterior_probs.npy", np.asarray(result.posterior_probs, dtype=float))
    summary = {
        "method": result.method,
        "log_likelihood": result.log_likelihood,
        "state_occupancy": result.state_occupancy,
        "metadata": result.metadata,
    }
    _write_json(method_dir / "result_summary.json", summary)
    write_yaml_config({"method": result.method, "config": result.config}, method_dir / "config.yaml")
    if metrics is not None:
        _write_json(method_dir / "metrics.json", metrics)


def run_population_state_analysis(
    data: AnalysisInput,
    analysis_cfg: Dict[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run all enabled state-inference methods on a common input object."""

    validated = validate_analysis_input(data)
    validated = apply_cluster_selection(validated, analysis_cfg.get("cluster_selection", {}))
    validated, population_filter_diagnostics = apply_population_filter(validated, analysis_cfg.get("population_filter", {}))
    preprocessing_cfg = dict(analysis_cfg.get("preprocessing") or {})
    methods_cfg = dict(analysis_cfg.get("methods") or {})
    evaluation_cfg = dict(analysis_cfg.get("evaluation") or {})
    plotting_cfg = dict(analysis_cfg.get("plotting") or {})

    results: Dict[str, StateInferenceResult] = {}
    metrics: Dict[str, Any] = {}
    for name, method in METHOD_REGISTRY.items():
        cfg = dict(methods_cfg.get(name) or {})
        if not bool(cfg.get("enabled", False)):
            continue
        try:
            result = method(validated, preprocessing_cfg, cfg)
        except ModuleNotFoundError as exc:
            metrics[name] = {"skipped": True, "reason": str(exc)}
            continue
        results[name] = result
        if validated.true_labels is not None and bool(evaluation_cfg.get("compute_ground_truth_metrics", True)):
            metrics[name] = evaluate_result(
                result,
                true_labels=np.asarray(validated.true_labels, dtype=np.int64),
                dt=float(validated.dt),
                boundary_tolerances=evaluation_cfg.get("boundary_tolerance_bins", [1, 2, 5]),
            )
        else:
            metrics[name] = {
                "state_occupancy": result.state_occupancy,
                "n_states": result.n_states,
            }
    comparison = compare_results(results) if bool(evaluation_cfg.get("compute_method_agreement", True)) else {}
    state_count_sweep = {}

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        write_yaml_config(analysis_cfg, root / "resolved_analysis_config.yaml")
        _write_json(root / "input_metadata.json", {"metadata": validated.metadata, "source_type": validated.source_type, "dt": validated.dt})
        if population_filter_diagnostics is not None:
            population_filter_diagnostics.to_csv(root / "population_filter_diagnostics.csv", index=False)
        for name, result in results.items():
            method_dir = root / name
            _save_result(method_dir, result, metrics.get(name))
            if bool(plotting_cfg.get("enabled", True)):
                save_result_plots(
                    result,
                    validated,
                    method_dir / "plots",
                    dpi=int(plotting_cfg.get("dpi", 150)),
                    save_format=str(plotting_cfg.get("save_format", "png")),
                )
        if results and bool(plotting_cfg.get("enabled", True)):
            plot_method_comparison(
                results,
                root / f"method_comparison.{str(plotting_cfg.get('save_format', 'png')).lower()}",
                dpi=int(plotting_cfg.get("dpi", 150)),
            )
        state_count_sweep = run_state_count_sweep(validated, analysis_cfg, output_dir=root / "state_count_sweep")
        _write_json(root / "metrics_summary.json", metrics)
        _write_json(root / "comparison_metrics.json", comparison)
        if state_count_sweep:
            recommendations = state_count_sweep.get("recommendations", {})
            _write_json(root / "state_count_recommendations.json", recommendations)
    else:
        state_count_sweep = run_state_count_sweep(validated, analysis_cfg)

    return {
        "results": results,
        "metrics": metrics,
        "comparison": comparison,
        "state_count_sweep": state_count_sweep,
        "output_dir": None if output_dir is None else str(Path(output_dir)),
    }


def run_analysis_on_binary_trace(
    trace_path: str | Path,
    *,
    parameter: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    base_output_dir: str | Path,
) -> Dict[str, Any]:
    data = analysis_input_from_binary_trace(trace_path, parameter=parameter, analysis_cfg=analysis_cfg)
    output_dir = resolve_analysis_output_dir(base_output_dir, analysis_cfg)
    return run_population_state_analysis(data, analysis_cfg, output_dir=output_dir)


def run_analysis_on_spiking_payload(
    payload: Dict[str, Any],
    *,
    parameter: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    base_output_dir: str | Path,
) -> Dict[str, Any]:
    data = analysis_input_from_spiking_payload(payload, parameter=parameter, analysis_cfg=analysis_cfg)
    output_dir = resolve_analysis_output_dir(base_output_dir, analysis_cfg)
    return run_population_state_analysis(data, analysis_cfg, output_dir=output_dir)


def run_analysis_from_source(
    source: str | Path,
    *,
    parameter: Optional[Dict[str, Any]],
    analysis_cfg: Dict[str, Any],
    source_type: str = "auto",
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    data = load_analysis_input(source, source_type=source_type, parameter=parameter, analysis_cfg=analysis_cfg)
    resolved_output = output_dir or resolve_analysis_output_dir(Path(source).parent if Path(source).is_file() else source, analysis_cfg)
    return run_population_state_analysis(data, analysis_cfg, output_dir=resolved_output)
