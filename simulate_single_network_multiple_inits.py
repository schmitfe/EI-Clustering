from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import maximum_rates_distribution as base
from maximum_rates_distribution_FP_init_legacy import (
    _legacy_cluster_sizes,
    _run_legacy_binary_simulation,
)

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

LEGACY_MODE_LABEL = "legacy_single_network_multi_init"
LEGACY_ANALYSIS_SUBDIR = "single_network_multi_init_corr_var"
DEFAULT_MAX_PAIRS = 200_000
MAX_VIOLIN_SAMPLES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple binary-network simulations on a single connectivity instance "
            "initialized from different mean-field fixpoints and analyze correlations/variances."
        )
    )
    parser.add_argument(
        "source",
        help="Path to a data folder (params.yaml) or an all_fixpoints_*.pkl bundle describing the network.",
    )
    parser.add_argument(
        "--fixpoints",
        type=str,
        help="Path to all_fixpoints_*.pkl (required if the source folder has no detectable bundle).",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="append",
        default=[],
        metavar="path=value",
        help="Override a parameter using dotted-path notation (may be repeated).",
    )
    parser.add_argument("--warmup-sweeps", type=int, help="Override binary.warmup_steps from the config.")
    parser.add_argument("--record-sweeps", type=int, help="Override binary.simulation_steps from the config.")
    parser.add_argument("--stride-sweeps", type=int, help="Override binary.sample_interval from the config.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size from the config.")
    parser.add_argument("--seed-network", type=int, help="Seed used to build the network connectivity.")
    parser.add_argument("--seed-inits", type=int, default=0, help="Seed controlling fixpoint sampling (default: %(default)s).")
    parser.add_argument("--n-inits", type=int, default=50, help="Number of distinct fixpoint initializations (default: %(default)s).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes used to parallelize per-initialization analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--focus-counts",
        type=int,
        nargs="+",
        help="Limit fixpoint sampling to the provided focus_count values (default: all).",
    )
    parser.add_argument(
        "--stability-filter",
        choices=("stable", "unstable", "any"),
        default="stable",
        help="Select only stable, unstable, or any fixpoints for initialization (default: %(default)s).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new simulations and only reuse existing traces.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if trace bundles already exist.",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="Recompute per-initialization analysis even if cached results exist.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for saved traces (defaults to binary.output_name or 'activity_trace').",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Maximum number of neuron pairs sampled for each correlation category (default: %(default)s).",
    )
    parser.add_argument(
        "--stride-sweeps-analysis",
        type=int,
        default=1,
        help="Down-sampling factor applied when analyzing neuron states (default: %(default)s).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        help="Override the kappa mixing value used by the legacy simulator.",
    )
    return parser.parse_args()


def _normalize_focus_list(values: Sequence[int] | None) -> List[int] | None:
    if not values:
        return None
    return sorted({int(value) for value in values})


def _candidate_sort_key(entry: Dict[str, Any]) -> Tuple[int, float, int]:
    focus_value = int(entry.get("focus_count", 0) or 0)
    try:
        val = float(entry.get("value", float("nan")))
    except (TypeError, ValueError):
        val = float("nan")
    finite_flag = 0 if math.isfinite(val) else 1
    safe_value = val if math.isfinite(val) else 0.0
    return (focus_value, finite_flag, safe_value)


def _load_fixpoint_candidates(
    bundle: Dict[str, Any],
    allowed_focus: Sequence[int] | None,
    stability_filter: str,
    expected_length: int,
    target_rep: float,
) -> List[Dict[str, Any]]:
    selection: List[Dict[str, Any]] = []
    fixpoints = bundle.get("fixpoints", {})
    allowed = set(int(value) for value in allowed_focus) if allowed_focus else None
    stability_filter = stability_filter.lower()
    for focus_label, focus_entries in fixpoints.items():
        try:
            focus_count = int(focus_label)
        except (TypeError, ValueError):
            continue
        if allowed and focus_count not in allowed:
            continue
        for rep_label, rep_entries in focus_entries.items():
            if not isinstance(rep_entries, dict):
                continue
            rep_value = base._parse_rep_from_key(str(rep_label))
            if rep_value is None or abs(rep_value - target_rep) > 1e-9:
                continue
            for idx, (fp_value, fixpoint) in enumerate(sorted(rep_entries.items(), key=lambda item: float(item[0]))):
                rates = fixpoint.get("rates")
                if rates is None:
                    continue
                values = np.asarray(rates, dtype=float).ravel()
                if values.size != expected_length:
                    raise ValueError(
                        f"Fixpoint entry {rep_label} lists {values.size} populations, expected {expected_length}."
                    )
                stability = str(fixpoint.get("stability", "") or "").lower() or "unknown"
                if stability_filter == "stable" and stability != "stable":
                    continue
                if stability_filter == "unstable" and stability == "stable":
                    continue
                candidate_id = f"{rep_label}_{idx}"
                try:
                    value = float(fp_value)
                except (TypeError, ValueError):
                    value = float("nan")
                selection.append(
                    {
                        "id": candidate_id,
                        "focus_count": focus_count,
                        "index": idx,
                        "value": value,
                        "rates": values.tolist(),
                        "stability": stability,
                        "rep_label": rep_label,
                    }
                )
    if not selection:
        raise ValueError("No fixpoints matched the provided filters.")
    selection.sort(key=_candidate_sort_key)
    return selection


def _select_fixpoints(
    candidates: Sequence[Dict[str, Any]],
    *,
    seed: int,
    count: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        raise ValueError("Fixpoint candidate list is empty.")
    rng = random.Random(int(seed))
    picks: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        picks.append(rng.choice(candidates))
    return picks


def _prepare_analysis_folder(parameter: Dict[str, Any], base_folder: str | None) -> Tuple[str, str, str]:
    folder, binary_dir, _ = base._prepare_output_folders(parameter, base_folder=base_folder)
    analysis_dir = os.path.join(binary_dir, LEGACY_ANALYSIS_SUBDIR)
    os.makedirs(analysis_dir, exist_ok=True)
    return folder, binary_dir, analysis_dir


def _resolve_binary_cfg(parameter: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Mirror maximum_rates_distribution._resolve_binary_config but adapt sweep naming.
    args.warmup_steps = args.warmup_sweeps
    args.simulation_steps = args.record_sweeps
    args.sample_interval = args.stride_sweeps
    if not hasattr(args, "seed") or args.seed is None:
        args.seed = args.seed_network
    cfg = base._resolve_binary_config(parameter, args)
    if args.kappa is not None:
        parameter["kappa"] = float(args.kappa)
    return cfg


def _assembly_membership(parameter: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    excit_size, inhib_size = _legacy_cluster_sizes(parameter)
    excit_labels = [f"E{idx + 1}" for idx in range(Q_value)]
    inhib_labels = [f"I{idx + 1}" for idx in range(Q_value)]
    assembly_names = excit_labels + inhib_labels
    membership: List[int] = []
    for idx in range(Q_value):
        membership.extend([idx] * excit_size)
    for idx in range(Q_value):
        membership.extend([Q_value + idx] * inhib_size)
    return np.asarray(membership, dtype=np.int64), assembly_names


def _downsample_states(states: np.ndarray, stride: int) -> np.ndarray:
    if states.ndim != 2 or states.size == 0:
        return states
    stride = max(1, int(stride))
    if stride == 1:
        return states
    return states[::stride]


def _pair_index_sample(
    assembly_ids: np.ndarray,
    *,
    max_pairs: int,
    within: bool,
    rng: np.random.Generator,
    allowed_mask: np.ndarray | None = None,
) -> List[Tuple[int, int]]:
    if allowed_mask is not None:
        base_indices = np.flatnonzero(allowed_mask)
        filtered_ids = assembly_ids[allowed_mask]
        if filtered_ids.size == 0:
            return []
    else:
        base_indices = np.arange(assembly_ids.size)
        filtered_ids = assembly_ids
    unique_ids = np.unique(filtered_ids)
    indices_by_assembly = {
        assembly: base_indices[filtered_ids == assembly] for assembly in unique_ids
    }
    pairs: List[Tuple[int, int]] = []
    if within:
        per_bucket = max(1, max_pairs // max(1, len(unique_ids)))
        for assembly, indices in indices_by_assembly.items():
            if indices.size < 2:
                continue
            samples = min(per_bucket, indices.size * (indices.size - 1) // 2)
            for _ in range(samples):
                i, j = rng.choice(indices, size=2, replace=False)
                pairs.append((int(i), int(j)))
    else:
        flat_indices = base_indices
        attempts = 0
        limit = max(10_000, 5 * max_pairs)
        while len(pairs) < max_pairs and attempts < limit:
            i, j = rng.choice(flat_indices, size=2, replace=False)
            if assembly_ids[i] == assembly_ids[j]:
                attempts += 1
                continue
            pairs.append((int(i), int(j)))
            attempts += 1
    return pairs[:max_pairs]


def _sample_specific_pairs(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
    same_cluster: bool,
) -> List[Tuple[int, int]]:
    if same_cluster:
        if first_indices.size < 2:
            return []
        samples = min(max_pairs, first_indices.size * (first_indices.size - 1) // 2)
        pairs = []
        for _ in range(samples):
            i, j = rng.choice(first_indices, size=2, replace=False)
            pairs.append((int(i), int(j)))
        return pairs
    if first_indices.size == 0 or second_indices.size == 0:
        return []
    samples = min(max_pairs, first_indices.size * second_indices.size)
    pairs = []
    for _ in range(samples):
        i = int(rng.choice(first_indices))
        j = int(rng.choice(second_indices))
        if i == j:
            continue
        pairs.append((i, j))
    return pairs


def _standardized_columns(data: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    if not indices:
        return np.zeros((data.shape[0], 0), dtype=np.float32), {}, np.zeros((0,), dtype=bool)
    unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    subset = data[:, unique].astype(np.float32, copy=True)
    means = subset.mean(axis=0)
    stds = subset.std(axis=0)
    valid = stds > 0
    valid_indices = np.nonzero(valid)[0]
    if valid_indices.size:
        subset[:, valid_indices] = (subset[:, valid_indices] - means[valid_indices]) / stds[valid_indices]
    mapping = {int(unique[idx]): int(idx) for idx in range(unique.size)}
    return subset, mapping, valid


def _compute_pairwise_correlations(
    data: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
    if not pairs or data.size == 0:
        return np.zeros((0,), dtype=np.float32)
    flat_indices = [idx for pair in pairs for idx in pair]
    subset, mapping, valid_mask = _standardized_columns(data, flat_indices)
    if subset.size == 0:
        return np.zeros((0,), dtype=np.float32)
    corr_values: List[float] = []
    for i, j in pairs:
        first = mapping.get(i)
        second = mapping.get(j)
        if first is None or second is None:
            continue
        if not (valid_mask[first] and valid_mask[second]):
            continue
        vec = subset[:, first] * subset[:, second]
        corr = float(np.mean(vec))
        corr_values.append(corr)
    return np.asarray(corr_values, dtype=np.float32)


def _cluster_pair_correlation_stats(
    data: np.ndarray,
    assembly_ids: np.ndarray,
    cluster_ids: Sequence[int],
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    size = len(cluster_ids)
    if data.size == 0 or size == 0:
        return (
            np.full((size, size), np.nan, dtype=np.float32),
            np.full((size, size), np.nan, dtype=np.float32),
        )
    cluster_ids = list(cluster_ids)
    mean_matrix = np.full((size, size), np.nan, dtype=np.float32)
    median_matrix = np.full((size, size), np.nan, dtype=np.float32)
    per_pair_limit = max(1, max_pairs // max(1, size * size))
    cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for i, cluster_i in enumerate(cluster_ids):
        indices_i = np.flatnonzero(assembly_ids == cluster_i)
        if indices_i.size == 0:
            continue
        for j in range(i, size):
            cluster_j = cluster_ids[j]
            indices_j = np.flatnonzero(assembly_ids == cluster_j)
            if indices_j.size == 0:
                continue
            same_cluster = i == j
            cache_key = (min(cluster_i, cluster_j), max(cluster_i, cluster_j))
            if cache_key in cache:
                mean_val, median_val = cache[cache_key]
            else:
                pairs = _sample_specific_pairs(indices_i, indices_j, per_pair_limit, rng, same_cluster=same_cluster)
                values = _compute_pairwise_correlations(data, pairs)
                if values.size == 0:
                    continue
                mean_val = float(np.mean(values))
                median_val = float(np.median(values))
                cache[cache_key] = (mean_val, median_val)
            mean_matrix[i, j] = mean_val
            median_matrix[i, j] = median_val
            if i != j:
                mean_matrix[j, i] = mean_val
                median_matrix[j, i] = median_val
    return mean_matrix, median_matrix


def _compute_variance_decomposition(values: np.ndarray, assembly_ids: np.ndarray, assembly_names: Sequence[str]) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "mu_emp": np.zeros((len(assembly_names),), dtype=np.float32),
            "var_total": np.zeros((len(assembly_names),), dtype=np.float32),
            "var_temporal": np.zeros((len(assembly_names),), dtype=np.float32),
            "var_quenched": np.zeros((len(assembly_names),), dtype=np.float32),
        }
    per_neuron_mean = values.mean(axis=0)
    per_neuron_var = values.var(axis=0)
    mu_emp: List[float] = []
    var_total: List[float] = []
    var_temporal: List[float] = []
    var_quenched: List[float] = []
    for idx, name in enumerate(assembly_names):
        mask = assembly_ids == idx
        if not np.any(mask):
            mu_emp.append(float("nan"))
            var_total.append(float("nan"))
            var_temporal.append(float("nan"))
            var_quenched.append(float("nan"))
            continue
        mu_slice = per_neuron_mean[mask]
        var_slice = per_neuron_var[mask]
        mu_emp.append(float(np.mean(mu_slice)))
        var_temporal.append(float(np.mean(var_slice)))
        var_quenched.append(float(np.var(mu_slice)))
        var_total.append(var_temporal[-1] + var_quenched[-1])
    return {
        "mu_emp": np.asarray(mu_emp, dtype=np.float32),
        "var_total": np.asarray(var_total, dtype=np.float32),
        "var_temporal": np.asarray(var_temporal, dtype=np.float32),
        "var_quenched": np.asarray(var_quenched, dtype=np.float32),
    }


def _candidate_prediction(candidate: Dict[str, Any]) -> np.ndarray:
    rates = np.asarray(candidate.get("rates", []), dtype=float)
    return rates.astype(np.float32, copy=False)


def _analyze_trace(
    trace_path: str,
    init_index: int,
    candidate: Dict[str, Any],
    parameter: Dict[str, Any],
    analysis_dir: str,
    assembly_ids: np.ndarray,
    assembly_names: Sequence[str],
    *,
    stride: int,
    max_pairs: int,
) -> Dict[str, Any]:
    payload = base._load_trace_payload(trace_path)
    states = np.asarray(payload.get("states"), dtype=np.uint8)
    fields = np.asarray(payload.get("subthreshold_fields"), dtype=np.float32)
    analysis_stride = max(1, int(stride))
    neuron_count = assembly_ids.size
    if states.ndim == 2 and states.size:
        states_ds = _downsample_states(states, analysis_stride).astype(np.float32, copy=False)
    else:
        states_ds = np.zeros((0, neuron_count), dtype=np.float32)
    if fields.ndim == 2 and fields.size:
        fields_ds = _downsample_states(fields, analysis_stride).astype(np.float32, copy=False)
    else:
        fields_ds = np.zeros((0, neuron_count), dtype=np.float32)
    rng = np.random.default_rng(init_index + 17)
    if states_ds.shape[1] != neuron_count:
        padded = np.zeros((states_ds.shape[0], neuron_count), dtype=np.float32)
        limit = min(states_ds.shape[1], neuron_count)
        padded[:, :limit] = states_ds[:, :limit]
        states_ds = padded
    if fields_ds.shape[1] != neuron_count:
        padded = np.zeros((fields_ds.shape[0], neuron_count), dtype=np.float32)
        limit = min(fields_ds.shape[1], neuron_count)
        padded[:, :limit] = fields_ds[:, :limit]
        fields_ds = padded
    Q_value = int(parameter.get("Q", len(assembly_names) // 2) or 0)
    excit_mask = assembly_ids < Q_value
    inhib_mask = assembly_ids >= Q_value
    correlations: Dict[str, Any] = {
        "state_excit_within": np.zeros((0,), dtype=np.float32),
        "state_excit_between": np.zeros((0,), dtype=np.float32),
        "state_inhib_within": np.zeros((0,), dtype=np.float32),
        "state_inhib_between": np.zeros((0,), dtype=np.float32),
        "field_excit_within": np.zeros((0,), dtype=np.float32),
        "field_excit_between": np.zeros((0,), dtype=np.float32),
        "field_inhib_within": np.zeros((0,), dtype=np.float32),
        "field_inhib_between": np.zeros((0,), dtype=np.float32),
    }
    if states_ds.size:
        within_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=excit_mask)
        between_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=excit_mask)
        correlations["state_excit_within"] = _compute_pairwise_correlations(states_ds, within_pairs)
        correlations["state_excit_between"] = _compute_pairwise_correlations(states_ds, between_pairs)
        inhib_within = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=inhib_mask)
        inhib_between = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=inhib_mask)
        correlations["state_inhib_within"] = _compute_pairwise_correlations(states_ds, inhib_within)
        correlations["state_inhib_between"] = _compute_pairwise_correlations(states_ds, inhib_between)
    if fields_ds.size:
        within_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=excit_mask)
        between_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=excit_mask)
        correlations["field_excit_within"] = _compute_pairwise_correlations(fields_ds, within_pairs)
        correlations["field_excit_between"] = _compute_pairwise_correlations(fields_ds, between_pairs)
        inhib_within = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=inhib_mask)
        inhib_between = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=inhib_mask)
        correlations["field_inhib_within"] = _compute_pairwise_correlations(fields_ds, inhib_within)
        correlations["field_inhib_between"] = _compute_pairwise_correlations(fields_ds, inhib_between)
    target_for_variance = fields_ds if fields_ds.size else states_ds
    variance_stats = _compute_variance_decomposition(target_for_variance, assembly_ids, assembly_names)
    cluster_indices = list(range(len(assembly_names)))
    state_cluster_mean, state_cluster_median = _cluster_pair_correlation_stats(
        states_ds,
        assembly_ids,
        cluster_indices,
        rng=rng,
        max_pairs=max_pairs,
    )
    field_cluster_mean, field_cluster_median = _cluster_pair_correlation_stats(
        fields_ds,
        assembly_ids,
        cluster_indices,
        rng=rng,
        max_pairs=max_pairs,
    )
    prediction = _candidate_prediction(candidate)
    per_init_path = os.path.join(analysis_dir, f"analysis_init{init_index:04d}.npz")
    np.savez_compressed(
        per_init_path,
        trace_path=np.array(trace_path),
        init_index=np.array(int(init_index)),
        focus_count=np.array(int(candidate.get("focus_count", -1))),
        stability=np.array(str(candidate.get("stability", "")), dtype=object),
        **{key: value for key, value in variance_stats.items()},
        **correlations,
        state_cluster_mean=state_cluster_mean,
        state_cluster_median=state_cluster_median,
        field_cluster_mean=field_cluster_mean,
        field_cluster_median=field_cluster_median,
        prediction=prediction,
    )
    result = dict(variance_stats)
    result.update(correlations)
    result["state_cluster_mean"] = state_cluster_mean
    result["state_cluster_median"] = state_cluster_median
    result["field_cluster_mean"] = field_cluster_mean
    result["field_cluster_median"] = field_cluster_median
    result["analysis_path"] = per_init_path
    result["prediction"] = prediction
    result["trace_path"] = trace_path
    return result


def _load_analysis_file(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["analysis_path"] = path
    return payload


@dataclass
class InitTask:
    index: int
    candidate: Dict[str, Any]
    trace_path: str
    analysis_dir: str
    binary_dir: str
    parameter: Dict[str, Any]
    binary_cfg: Dict[str, Any]
    assembly_ids: np.ndarray
    assembly_names: Sequence[str]
    stride: int
    max_pairs: int
    needs_simulation: bool
    needs_analysis: bool


def _process_task(task: InitTask) -> Dict[str, Any]:
    logs: List[str] = []
    success = True
    error: str | None = None
    trace_exists = os.path.exists(task.trace_path)
    try:
        if task.needs_simulation:
            logs.append(
                f"Simulating init {task.index}: fixpoint {task.candidate.get('id')} "
                f"(focus {task.candidate.get('focus_count')}, stability {task.candidate.get('stability')})."
            )
            trace_path = _run_legacy_binary_simulation(
                task.parameter,
                task.binary_cfg,
                task.binary_dir,
                os.path.splitext(os.path.basename(task.trace_path))[0],
                task.candidate["rates"],
                seed=int(task.binary_cfg.get("seed", 0) or 0),
                capture_state_dynamics=True,
                cluster_seed=task.index,
            )
            task.trace_path = trace_path
            trace_exists = True
        if not trace_exists:
            raise FileNotFoundError(f"Trace {task.trace_path} missing for init {task.index}.")
        if task.needs_analysis:
            analysis = _analyze_trace(
                task.trace_path,
                task.index,
                task.candidate,
                task.parameter,
                task.analysis_dir,
                task.assembly_ids,
                task.assembly_names,
                stride=task.stride,
                max_pairs=task.max_pairs,
            )
        else:
            analysis_path = os.path.join(task.analysis_dir, f"analysis_init{task.index:04d}.npz")
            if not os.path.exists(analysis_path):
                raise FileNotFoundError(f"Analysis file {analysis_path} missing for init {task.index}.")
            analysis = _load_analysis_file(analysis_path)
    except Exception as exc:  # pragma: no cover - runtime pipeline
        success = False
        error = str(exc)
        logs.append(f"Error: {exc}")
        analysis = {}
    return {
        "index": task.index,
        "candidate": task.candidate,
        "trace_path": task.trace_path,
        "analysis": analysis,
        "success": success,
        "error": error,
        "logs": logs,
    }


def _collect_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    corr_keys = (
        "state_excit_within",
        "state_excit_between",
        "state_inhib_within",
        "state_inhib_between",
        "field_excit_within",
        "field_excit_between",
        "field_inhib_within",
        "field_inhib_between",
    )
    vector_keys = ("mu_emp", "var_total", "var_temporal", "var_quenched")
    matrix_keys = ("state_cluster_mean", "state_cluster_median", "field_cluster_mean", "field_cluster_median")
    pooled_corr: Dict[str, List[np.ndarray]] = {key: [] for key in corr_keys}
    pooled_vectors: Dict[str, List[np.ndarray]] = {key: [] for key in vector_keys}
    pooled_matrices: Dict[str, List[np.ndarray]] = {key: [] for key in matrix_keys}
    analysis_paths: List[str] = []
    trace_paths: List[str] = []
    candidate_info: List[Dict[str, Any]] = []
    for entry in results:
        analysis = entry.get("analysis") or {}
        if not entry.get("success"):
            continue
        analysis_paths.append(analysis.get("analysis_path", ""))
        trace_paths.append(entry.get("trace_path", ""))
        candidate_info.append(entry.get("candidate", {}))
        for key in corr_keys:
            value = np.asarray(analysis.get(key), dtype=np.float32)
            if value.size:
                pooled_corr[key].append(value)
        for key in vector_keys:
            value = np.asarray(analysis.get(key), dtype=np.float32)
            if value.size:
                pooled_vectors[key].append(value)
        for key in matrix_keys:
            value = np.asarray(analysis.get(key), dtype=np.float32)
            if value.size:
                pooled_matrices[key].append(value)
    summary: Dict[str, Any] = {}
    for key, values in pooled_corr.items():
        summary[key] = np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)
    for key, values in pooled_vectors.items():
        if values:
            stacked = np.stack(values)
            summary[f"{key}_mean"] = np.nanmean(stacked, axis=0)
            summary[f"{key}_std"] = np.nanstd(stacked, axis=0)
        else:
            summary[f"{key}_mean"] = np.zeros((0,), dtype=np.float32)
            summary[f"{key}_std"] = np.zeros((0,), dtype=np.float32)
    for key, values in pooled_matrices.items():
        if values:
            stacked = np.stack(values)
            summary[key] = np.nanmean(stacked, axis=0)
        else:
            summary[key] = np.zeros((0, 0), dtype=np.float32)
    summary["analysis_paths"] = np.asarray(analysis_paths, dtype=object)
    summary["trace_paths"] = np.asarray(trace_paths, dtype=object)
    summary["candidates"] = np.asarray(candidate_info, dtype=object)
    return summary


def _save_summary(analysis_dir: str, summary: Dict[str, Any]) -> str:
    path = os.path.join(analysis_dir, "pooled_summary.npz")
    np.savez_compressed(path, **summary)
    return path


def _prepare_metadata(
    analysis_dir: str,
    *,
    base_output: str,
    fixpoints_path: str,
    focus_counts: Sequence[int] | None,
    stability_filter: str,
    seed_network: int,
    binary_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    metadata_path = os.path.join(analysis_dir, "metadata.yaml")
    metadata = base._load_metadata(metadata_path) if os.path.exists(metadata_path) else {}
    changed = False
    existing_base = metadata.get("base_output_name")
    if existing_base and existing_base != base_output:
        raise ValueError(
            f"Analysis folder {analysis_dir} already stores data for output '{existing_base}'. "
            f"Requested base name '{base_output}' would mix incompatible runs."
        )
    if existing_base is None:
        metadata["base_output_name"] = base_output
        changed = True
    stored_mode = metadata.get("mode")
    if stored_mode not in {None, LEGACY_MODE_LABEL}:
        raise ValueError(
            f"{analysis_dir} already contains mode '{stored_mode}'. Create a new folder for this workflow."
        )
    if stored_mode != LEGACY_MODE_LABEL:
        metadata["mode"] = LEGACY_MODE_LABEL
        changed = True
    stored_focus = _normalize_focus_list(metadata.get("focus_counts"))
    focus_filter = _normalize_focus_list(focus_counts)
    if stored_focus is not None:
        if focus_filter is not None and stored_focus != focus_filter:
            raise ValueError(
                f"Analysis folder already constrained focus_counts to {stored_focus}. Requested {focus_filter} incompatible."
            )
        focus_filter = stored_focus
    elif focus_filter is not None:
        metadata["focus_counts"] = focus_filter
        changed = True
    stored_fixpoints = metadata.get("fixpoints_file")
    if stored_fixpoints and os.path.abspath(stored_fixpoints) != os.path.abspath(fixpoints_path):
        raise ValueError(
            f"Analysis folder already references {stored_fixpoints}. Requested {fixpoints_path} incompatible."
        )
    elif not stored_fixpoints:
        metadata["fixpoints_file"] = os.path.abspath(fixpoints_path)
        changed = True
    stored_filter = metadata.get("stability_filter")
    if stored_filter and stored_filter != stability_filter:
        raise ValueError(
            f"Analysis folder already uses stability filter '{stored_filter}'. Requested '{stability_filter}' incompatible."
        )
    else:
        metadata["stability_filter"] = stability_filter
        changed = True if stored_filter is None else changed
    stored_seed = metadata.get("network_seed")
    if stored_seed is not None and stored_seed != seed_network:
        raise ValueError(
            f"Analysis folder already fixes network_seed={stored_seed}. Requested {seed_network} incompatible."
        )
    if stored_seed is None:
        metadata["network_seed"] = seed_network
        changed = True
    stride = int(binary_cfg.get("sample_interval", 1) or 1)
    warmup = int(binary_cfg.get("warmup_steps", 0) or 0)
    record = int(binary_cfg.get("simulation_steps", 0) or 0)
    metadata["sample_interval"] = stride
    metadata["warmup_steps"] = warmup
    metadata["record_steps"] = record
    metadata["base_seed"] = int(seed_network)
    if changed:
        base._save_metadata(metadata_path, metadata)
    return metadata


def _prepare_tasks(
    fixes: Sequence[Dict[str, Any]],
    *,
    binary_dir: str,
    analysis_dir: str,
    binary_cfg: Dict[str, Any],
    parameter: Dict[str, Any],
    assembly_ids: np.ndarray,
    assembly_names: Sequence[str],
    args: argparse.Namespace,
) -> List[InitTask]:
    tasks: List[InitTask] = []
    base_output = binary_cfg.get("output_name", "activity_trace")
    for idx, candidate in enumerate(fixes):
        label = f"{base_output}_init{idx:04d}"
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        analysis_path = os.path.join(analysis_dir, f"analysis_init{idx:04d}.npz")
        needs_simulation = (not args.analysis_only) and (args.overwrite_simulation or not os.path.exists(trace_path))
        needs_analysis = args.overwrite_analysis or not os.path.exists(analysis_path)
        tasks.append(
            InitTask(
                index=idx,
                candidate=candidate,
                trace_path=trace_path,
                analysis_dir=analysis_dir,
                binary_dir=binary_dir,
                parameter=parameter,
                binary_cfg=dict(binary_cfg),
                assembly_ids=assembly_ids,
                assembly_names=assembly_names,
                stride=max(1, int(args.stride_sweeps_analysis or 1)),
                max_pairs=max(1, int(args.max_pairs or DEFAULT_MAX_PAIRS)),
                needs_simulation=needs_simulation,
                needs_analysis=needs_analysis,
            )
        )
    return tasks


def _execute_tasks(tasks: Sequence[InitTask], jobs: int) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    if jobs <= 1:
        iterator = tqdm(tasks, desc="Simulations") if tqdm else tasks
        results = []
        try:
            for task in iterator:
                results.append(_process_task(task))
        finally:
            if tqdm and hasattr(iterator, "close"):
                iterator.close()
        return results
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_process_task, task) for task in tasks]
        progress = tqdm(total=len(futures), desc="Simulations") if tqdm else None
        results: List[Dict[str, Any]] = []
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    return results


def _plot_placeholder(path: str, message: str, plt_module) -> None:
    fig, ax = plt_module.subplots(figsize=(4, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt_module.close(fig)


def _plot_violins(path: str, within: np.ndarray, between: np.ndarray, title: str, ylabel: str, plt_module) -> None:
    datasets = [np.asarray(within, dtype=float), np.asarray(between, dtype=float)]
    if datasets[0].size == 0 or datasets[1].size == 0:
        which = []
        if datasets[0].size == 0:
            which.append("within")
        if datasets[1].size == 0:
            which.append("between")
        label = "/".join(which)
        _plot_placeholder(path, f"No {label} data for {title.lower()}", plt_module)
        return
    fig, ax = plt_module.subplots(figsize=(5, 3))
    labels = ["Within assembly", "Between assemblies"]
    parts = ax.violinplot(datasets, showmeans=True, showmedians=False)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt_module.close(fig)


def _violin_stats(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.median(values))


def _scatter_points(ax, values: np.ndarray, position: float, color: str, rng: np.random.Generator) -> None:
    if values.size == 0:
        return
    samples = values
    if values.size > MAX_VIOLIN_SAMPLES:
        idx = rng.choice(values.size, size=MAX_VIOLIN_SAMPLES, replace=False)
        samples = values[idx]
    jitter = (rng.random(samples.size) - 0.5) * 0.2
    ax.scatter(
        np.full(samples.size, position) + jitter,
        samples,
        color=color,
        alpha=0.2,
        s=6,
        linewidths=0,
    )


def _plot_violin_panel(ax, within: np.ndarray, between: np.ndarray, title: str) -> None:
    labels = ["Within cluster", "Across clusters"]
    datasets = [np.asarray(within, dtype=float), np.asarray(between, dtype=float)]
    if datasets[0].size == 0 or datasets[1].size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {title}", ha="center", va="center")
        return
    parts = ax.violinplot(datasets, positions=[1, 2], showmeans=True, showmedians=False)
    for body in parts["bodies"]:
        body.set_alpha(0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title(title)
    ax.set_ylabel("Correlation")
    rng = np.random.default_rng(42)
    colors = ["#1f77b4", "#ff7f0e"]
    for idx, (values, color) in enumerate(zip(datasets, colors), start=1):
        _scatter_points(ax, values, float(idx), color, rng)
    stats_lines = []
    for label, values in zip(labels, datasets):
        mean_val, median_val = _violin_stats(values)
        stats_lines.append(f"{label}: μ={mean_val:.3f}, med={median_val:.3f}")
    ax.text(0.02, 0.98, "\n".join(stats_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)


def _plot_state_field_correlations(
    path: str,
    state_within: np.ndarray,
    state_between: np.ndarray,
    field_within: np.ndarray,
    field_between: np.ndarray,
    plt_module,
) -> None:
    datasets = [
        ("State correlations", np.asarray(state_within, dtype=float), np.asarray(state_between, dtype=float)),
        ("Subthreshold correlations", np.asarray(field_within, dtype=float), np.asarray(field_between, dtype=float)),
    ]
    if not any(data[1].size and data[2].size for data in datasets):
        _plot_placeholder(path, "No correlation data available", plt_module)
        return
    fig, axes = plt_module.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, within, between) in zip(axes, datasets):
        _plot_violin_panel(ax, within, between, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt_module.close(fig)


def _plot_variance_compare(path: str, stats: Dict[str, Any], assembly_names: Sequence[str], plt_module) -> None:
    mu = np.asarray(stats.get("mu_emp_mean"), dtype=float)
    var_temporal = np.asarray(stats.get("var_temporal_mean"), dtype=float)
    var_quenched = np.asarray(stats.get("var_quenched_mean"), dtype=float)
    if mu.size == 0:
        _plot_placeholder(path, "No variance statistics available", plt_module)
        return
    x = np.arange(len(assembly_names))
    fig, (ax1, ax2) = plt_module.subplots(2, 1, figsize=(7, 6), sharex=True)
    width = 0.6
    ax1.bar(x, var_temporal, width, label="Temporal")
    ax1.bar(x, var_quenched, width, bottom=var_temporal, label="Quenched")
    ax1.set_ylabel("Variance")
    ax1.legend()
    ax1.set_title("Variance decomposition (proxy inputs)")
    ax2.bar(x, mu, width, color="#4444AA")
    ax2.set_xticks(x)
    ax2.set_xticklabels(assembly_names, rotation=45)
    ax2.set_ylabel("Mean (proxy)")
    ax2.set_title("Empirical mean per assembly")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt_module.close(fig)


def _plot_cluster_matrix(
    path: str,
    state_mean: np.ndarray,
    state_median: np.ndarray,
    field_mean: np.ndarray,
    field_median: np.ndarray,
    cluster_labels: Sequence[str],
    plt_module,
) -> None:
    if not cluster_labels:
        _plot_placeholder(path, "No cluster labels available", plt_module)
        return
    matrices = [
        ("State mean", np.asarray(state_mean, dtype=float)),
        ("State median", np.asarray(state_median, dtype=float)),
        ("Field mean", np.asarray(field_mean, dtype=float)),
        ("Field median", np.asarray(field_median, dtype=float)),
    ]
    if not any(matrix.size for _, matrix in matrices):
        _plot_placeholder(path, "No cluster correlation data", plt_module)
        return
    fig, axes = plt_module.subplots(2, 2, figsize=(12, 8))
    vmin, vmax = -1.0, 1.0
    images = []
    label_indices = range(len(cluster_labels))
    tick_positions = []
    tick_labels = []
    for label in ("E1", "E10", "E20", "I1", "I10", "I20"):
        if label in cluster_labels:
            idx = cluster_labels.index(label)
            tick_positions.append(idx)
            tick_labels.append(label)
    for ax, (title, matrix) in zip(axes.flat, matrices):
        if matrix.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="coolwarm")
        images.append(im)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_title(title)
    if images:
        fig.subplots_adjust(wspace=0.4, right=0.88)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("Correlation")
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    fig.savefig(path, dpi=200)
    plt_module.close(fig)


def main() -> None:
    args = parse_args()
    parameter, folder_hint, fixpoints_path = base._resolve_simulation_source(
        args.source,
        fixpoint_hint=args.fixpoints,
        overrides=args.overwrite,
    )
    target_rep = parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError("Parameter set must define R_Eplus to align fixpoints with the simulated network.")
    target_rep = float(target_rep)
    bundle = base._load_fixpoint_bundle(fixpoints_path)
    binary_cfg = _resolve_binary_cfg(parameter, args)
    seed_network = args.seed_network
    if seed_network is None:
        seed_network = int(binary_cfg.get("seed", 0) or 0)
    binary_cfg["seed"] = int(seed_network)
    folder, binary_dir, analysis_dir = _prepare_analysis_folder(parameter, folder_hint)
    metadata = _prepare_metadata(
        analysis_dir,
        base_output=binary_cfg.get("output_name", "activity_trace"),
        fixpoints_path=fixpoints_path,
        focus_counts=args.focus_counts,
        stability_filter=args.stability_filter,
        seed_network=seed_network,
        binary_cfg=binary_cfg,
    )
    assembly_ids, assembly_names = _assembly_membership(parameter)
    focus_filter = _normalize_focus_list(args.focus_counts) or metadata.get("focus_counts")
    pop_length = 2 * int(parameter.get("Q", 0) or 0)
    candidates = _load_fixpoint_candidates(
        bundle,
        focus_filter,
        args.stability_filter,
        pop_length,
        target_rep,
    )
    n_inits = max(0, int(args.n_inits or 0))
    picks = _select_fixpoints(candidates, seed=args.seed_inits, count=n_inits)
    tasks = _prepare_tasks(
        picks,
        binary_dir=binary_dir,
        analysis_dir=analysis_dir,
        binary_cfg=binary_cfg,
        parameter=parameter,
        assembly_ids=assembly_ids,
        assembly_names=assembly_names,
        args=args,
    )
    if not tasks:
        print("No initialization tasks to run.")
        return
    results = _execute_tasks(tasks, max(1, int(args.jobs or 1)))
    for entry in results:
        idx = entry.get("index")
        for line in entry.get("logs", []):
            print(f"[init {idx:04d}] {line}")
        if not entry.get("success", False):
            print(f"[init {idx:04d}] Failed: {entry.get('error')}")
    summary = _collect_results(results)
    summary_path = _save_summary(analysis_dir, summary)
    plt = base._prepare_matplotlib()
    _plot_state_field_correlations(
        os.path.join(analysis_dir, "violin_correlations.png"),
        summary.get("state_excit_within", np.zeros((0,), dtype=np.float32)),
        summary.get("state_excit_between", np.zeros((0,), dtype=np.float32)),
        summary.get("field_excit_within", np.zeros((0,), dtype=np.float32)),
        summary.get("field_excit_between", np.zeros((0,), dtype=np.float32)),
        plt,
    )
    _plot_variance_compare(
        os.path.join(analysis_dir, "variance_compare.png"),
        summary,
        assembly_names,
        plt,
    )
    cluster_labels = assembly_names
    _plot_cluster_matrix(
        os.path.join(analysis_dir, "cluster_correlation_matrix.png"),
        summary.get("state_cluster_mean", np.zeros((0, 0), dtype=np.float32)),
        summary.get("state_cluster_median", np.zeros((0, 0), dtype=np.float32)),
        summary.get("field_cluster_mean", np.zeros((0, 0), dtype=np.float32)),
        summary.get("field_cluster_median", np.zeros((0, 0), dtype=np.float32)),
        cluster_labels,
        plt,
    )
    print(f"Analysis complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
