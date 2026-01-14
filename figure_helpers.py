from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import binary_simulation_multi_init as binary_multi
import ei_pipeline
from MeanField.rate_system import ensure_output_folder
from sim_config import sim_tag_from_cfg


@dataclass
class PipelineSweepSettings:
    v_start: float = 0.0
    v_end: float = 1.0
    v_steps: int = 1000
    retry_step: float | None = None
    jobs: int = 1
    overwrite_simulation: bool = False
    plot_erfs: bool = False


@dataclass
class BinaryRunSettings:
    warmup_steps: int | None = None
    simulation_steps: int | None = None
    sample_interval: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    output_name: str | None = None


@dataclass
class LegacyMaxRateResult:
    pooled_maxima: np.ndarray
    example_trace_path: str | None
    example_seed: int | None
    focus_rates: Dict[int, Dict[str, List[float]]]
    focus_expectations: Dict[int, float]
    binary_dir: str
    analysis_dir: str


@dataclass
class CorrelationRunResult:
    summary: Dict[str, Any]
    analysis_dir: str
    trace_paths: Sequence[str]


def resolve_focus_counts(parameter: Dict[str, Any], explicit: Iterable[int] | None = None) -> List[int]:
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    if explicit:
        values = sorted({max(1, int(val)) for val in explicit})
        if values:
            return values
    return list(range(1, Q_value + 1))


def resolve_r_eplus_values(
    parameter: Dict[str, Any],
    explicit: Iterable[float] | None = None,
) -> List[float]:
    if explicit:
        return [float(val) for val in explicit]
    base_value = parameter.get("R_Eplus")
    if base_value is None:
        raise ValueError(
            "No R_Eplus value available. Provide it via the config or --r-eplus overrides."
        )
    return [float(base_value)]


def resolve_binary_config(parameter: Dict[str, Any], overrides: BinaryRunSettings) -> Dict[str, Any]:
    args = SimpleNamespace(
        warmup_steps=overrides.warmup_steps,
        simulation_steps=overrides.simulation_steps,
        sample_interval=overrides.sample_interval,
        batch_size=overrides.batch_size,
        seed=overrides.seed,
        output_name=overrides.output_name,
    )
    return binary_multi._resolve_binary_config(parameter, args)


def _filtered_parameter_for_tag(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    filtered.pop("R_Eplus", None)
    filtered.pop("focus_count", None)
    filtered.pop("focus_counts", None)
    return filtered


def compute_fixpoint_bundle_path(parameter: Dict[str, Any]) -> str:
    filtered = _filtered_parameter_for_tag(parameter)
    tag = sim_tag_from_cfg(filtered)
    kappa = float(parameter.get("kappa", 0.0) or 0.0)
    conn = str(parameter.get("connection_type", "bernoulli")).lower().replace(" ", "_")
    encoded_kappa = f"{kappa:.2f}".replace(".", "_")
    R_j = parameter.get("R_j", 0.0)
    return os.path.join("data", f"all_fixpoints_{conn}_kappa{encoded_kappa}_Rj{R_j}_{tag}.pkl")


def ensure_fixpoint_bundle(
    parameter: Dict[str, Any],
    focus_counts: Sequence[int],
    r_eplus_values: Sequence[float],
    sweep_cfg: PipelineSweepSettings,
) -> Tuple[str, str]:
    param = dict(parameter)
    param["focus_counts"] = list(focus_counts)
    args = SimpleNamespace(
        v_start=sweep_cfg.v_start,
        v_end=sweep_cfg.v_end,
        v_steps=sweep_cfg.v_steps,
        retry_step=sweep_cfg.retry_step,
        jobs=sweep_cfg.jobs,
        overwrite_simulation=bool(sweep_cfg.overwrite_simulation),
    )
    folder = ei_pipeline.run_simulation(args, param, r_eplus_values, focus_counts)
    if folder is None:
        filtered = _filtered_parameter_for_tag(param)
        folder = ensure_output_folder(param, tag=sim_tag_from_cfg(filtered))
    ei_pipeline.run_analysis(folder, param, focus_counts, plot_erfs=sweep_cfg.plot_erfs)
    bundle_path = compute_fixpoint_bundle_path(param)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Fixpoint bundle {bundle_path} was not generated. "
            "Ensure ei_pipeline completed successfully."
        )
    return folder, bundle_path


def _legacy_candidate_selection(
    parameter: Dict[str, Any],
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
) -> Tuple[List[Dict[str, Any]], int]:
    bundle = binary_multi._load_fixpoint_bundle(bundle_path)
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive for the legacy workflow.")
    pop_vector = 2 * Q_value
    R_Eplus = parameter.get("R_Eplus")
    if R_Eplus is None:
        raise ValueError("Parameter 'R_Eplus' must be set before running legacy simulations.")
    raw_candidates = binary_multi._load_fixpoint_candidates(
        bundle,
        focus_counts,
        stability_filter,
        pop_vector,
        float(R_Eplus),
    )
    candidates = binary_multi._deduplicate_candidates_by_focus(raw_candidates, Q_value)
    if not candidates:
        raise ValueError("No fixpoint candidates matched the requested filters.")
    return candidates, Q_value


def run_legacy_max_rate_analysis(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
    folder_hint: str,
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
    bin_size: int,
    total_simulations: int,
    base_seed: int,
    jobs: int = 1,
    analysis_only: bool = False,
    overwrite_simulation: bool = False,
    overwrite_analysis: bool = False,
) -> LegacyMaxRateResult:
    folder, binary_dir, analysis_dir = binary_multi._prepare_max_rate_folder(parameter, folder_hint)
    candidates, Q_value = _legacy_candidate_selection(parameter, bundle_path, focus_counts, stability_filter)
    seeds = [int(base_seed) + idx for idx in range(total_simulations)]
    if not seeds:
        raise ValueError("No simulations were requested (total_simulations=0).")
    base_output = binary_cfg.get("output_name", "activity_trace")
    tasks: List[Dict[str, Any]] = []
    capture_assigned = False
    for seed in seeds:
        label = binary_multi._format_seed_label(base_output, seed)
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        maxima_path = os.path.join(analysis_dir, f"{label}_maxima.npz")
        candidate = binary_multi._candidate_for_seed(seed, candidates)
        trace_exists = os.path.exists(trace_path)
        needs_simulation = (not analysis_only) and (overwrite_simulation or not trace_exists)
        capture_spikes = False
        if needs_simulation and not capture_assigned:
            capture_spikes = True
            capture_assigned = True
        tasks.append(
            {
                "seed": int(seed),
                "label": label,
                "trace_path": trace_path,
                "maxima_path": maxima_path,
                "parameter": parameter,
                "binary_cfg": dict(binary_cfg),
                "binary_dir": binary_dir,
                "bin_size": bin_size,
                "needs_simulation": needs_simulation,
                "capture_spikes": capture_spikes,
                "overwrite_analysis": bool(overwrite_analysis),
                "candidate_id": candidate.get("id"),
                "candidate_focus": candidate.get("focus_count"),
                "candidate_stability": candidate.get("stability"),
                "init_rates": tuple(float(value) for value in candidate["rates"]),
            }
        )
    results: List[Dict[str, Any]] = []
    if jobs <= 1:
        for task in tasks:
            result = binary_multi._process_seed_task(task)
            binary_multi._emit_worker_logs(result)
            results.append(result)
    else:
        max_workers = min(int(jobs), len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(binary_multi._process_seed_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                binary_multi._emit_worker_logs(result)
                results.append(result)
    result_map = {entry.get("seed"): entry for entry in results}
    pooled_entries: List[float] = []
    example_trace_path: str | None = None
    example_seed: int | None = None
    states_available = False
    excitatory_clusters: int | None = None
    for task in tasks:
        seed = task["seed"]
        result = result_map.get(seed)
        if not result:
            continue
        maxima = result.get("maxima")
        if maxima:
            pooled_entries.extend(list(maxima))
        excit_count = result.get("excitatory_clusters")
        if excitatory_clusters is None and excit_count is not None:
            excitatory_clusters = int(excit_count)
        trace_path = result.get("trace_path") or task["trace_path"]
        if trace_path and os.path.exists(trace_path):
            if example_trace_path is None:
                example_trace_path = trace_path
                example_seed = seed
                states_available = bool(result.get("states_recorded"))
            elif not states_available:
                payload = binary_multi._load_trace_payload(trace_path)
                state_payload = np.asarray(payload.get("states"), dtype=np.uint8)
                if state_payload.size:
                    example_trace_path = trace_path
                    example_seed = seed
                    states_available = True
    if not pooled_entries:
        raise RuntimeError("No maxima were collected. Ensure the legacy simulations ran successfully.")
    excit_count_final = excitatory_clusters or Q_value
    focus_rates = binary_multi._focus_payload_from_candidates(candidates, excit_count_final)
    focus_expectations = binary_multi._focus_expectations_from_payload(focus_rates)
    pooled_array = np.asarray(pooled_entries, dtype=float)
    return LegacyMaxRateResult(
        pooled_maxima=pooled_array,
        example_trace_path=example_trace_path,
        example_seed=example_seed,
        focus_rates=focus_rates,
        focus_expectations=focus_expectations,
        binary_dir=binary_dir,
        analysis_dir=analysis_dir,
    )


def _mean_connectivity(parameter: Dict[str, Any]) -> float:
    N_E = float(parameter.get("N_E", 0.0) or 0.0)
    N_I = float(parameter.get("N_I", 0.0) or 0.0)
    p_ee = float(parameter.get("p0_ee", 0.0) or 0.0)
    p_ei = float(parameter.get("p0_ei", 0.0) or 0.0)
    p_ie = float(parameter.get("p0_ie", 0.0) or 0.0)
    p_ii = float(parameter.get("p0_ii", 0.0) or 0.0)
    total = (N_E + N_I) ** 2
    if total <= 0:
        return 0.0
    numerator = (N_E ** 2) * p_ee + (N_E * N_I) * (p_ei + p_ie) + (N_I ** 2) * p_ii
    return numerator / total


def scale_connectivity(parameter: Dict[str, Any], target_connectivity: float) -> Dict[str, Any]:
    base_conn = _mean_connectivity(parameter)
    if base_conn <= 0:
        return dict(parameter)
    scale = float(target_connectivity) / base_conn
    updated = dict(parameter)
    for key in ("p0_ee", "p0_ei", "p0_ie", "p0_ii"):
        value = float(updated.get(key, 0.0) or 0.0) * scale
        updated[key] = max(0.0, min(1.0, value))
    return updated


def run_multi_init_correlation(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
    folder_hint: str,
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
    n_inits: int,
    seed_inits: int,
    seed_network: int,
    stride_analysis: int,
    max_pairs: int,
    jobs: int,
    analysis_only: bool = False,
    overwrite_simulation: bool = False,
    overwrite_analysis: bool = False,
) -> CorrelationRunResult:
    folder, binary_dir, analysis_dir = binary_multi._prepare_multi_init_folder(parameter, folder_hint)
    binary_cfg = dict(binary_cfg)
    binary_cfg["seed"] = int(seed_network)
    target_rep = parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError("Parameter 'R_Eplus' must be set before running the correlation workflow.")
    target_rep = float(target_rep)
    metadata = binary_multi._prepare_metadata(
        analysis_dir,
        base_output=binary_cfg.get("output_name", "activity_trace"),
        fixpoints_path=bundle_path,
        focus_counts=focus_counts,
        stability_filter=stability_filter,
        seed_network=seed_network,
        binary_cfg=binary_cfg,
        target_rep=target_rep,
    )
    bundle = binary_multi._load_fixpoint_bundle(bundle_path)
    Q_value = int(parameter.get("Q", 0) or 0)
    pop_length = 2 * Q_value
    candidates = binary_multi._load_fixpoint_candidates(
        bundle,
        focus_counts,
        stability_filter,
        pop_length,
        target_rep,
    )
    if not candidates:
        raise ValueError("No fixpoints available for the correlation workflow.")
    picks = binary_multi._select_fixpoints(candidates, seed=seed_inits, count=max(0, int(n_inits)))
    assembly_ids, assembly_names = binary_multi._assembly_membership(parameter)
    worker_args = SimpleNamespace(
        analysis_only=analysis_only,
        overwrite_simulation=overwrite_simulation,
        overwrite_analysis=overwrite_analysis,
        stride_sweeps_analysis=stride_analysis,
        max_pairs=max_pairs,
    )
    tasks = binary_multi._prepare_tasks(
        picks,
        binary_dir=binary_dir,
        analysis_dir=analysis_dir,
        binary_cfg=binary_cfg,
        parameter=parameter,
        assembly_ids=assembly_ids,
        assembly_names=assembly_names,
        args=worker_args,
    )
    if not tasks:
        raise RuntimeError("No initialization tasks were prepared for the correlation workflow.")
    results = binary_multi._execute_tasks(tasks, max(1, int(jobs)))
    for entry in results:
        idx = entry.get("index")
        for line in entry.get("logs", []):
            print(f"[init {idx:04d}] {line}")
        if not entry.get("success", False):
            print(f"[init {idx:04d}] Failed: {entry.get('error')}")
    summary = binary_multi._collect_results(results)
    return CorrelationRunResult(summary=summary, analysis_dir=analysis_dir, trace_paths=summary.get("trace_paths", []))


def mean_connectivity(parameter: Dict[str, Any]) -> float:
    return _mean_connectivity(parameter)
