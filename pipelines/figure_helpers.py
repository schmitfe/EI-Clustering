from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import math
import os
import random
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from BinaryNetwork.BinaryNetwork import warm_numba_caches
from BinaryNetwork.ClusteredEI_network import ClusteredEI_network
from MeanField.rate_system import ensure_output_folder
from pipelines.binary import (
    finalize_binary_config,
    run_binary_simulation,
)
from pipelines.mean_field import run_analysis as run_mean_field_analysis
from pipelines.mean_field import run_simulation as run_mean_field_simulation
from sim_config import sim_tag_from_cfg, write_yaml_config


DEFAULT_MAX_PAIRS = 200_000
CORR_ANALYSIS_SUBDIR = "single_network_multi_init_corr_var"
NETWORK_SUMMARY_FILE = "network_summary.json"
MEASURES = ("output", "input")
CATEGORIES = ("within", "across")
FISHER_EPS = 1e-6
MEASURE_KEY_MAP = {
    ("output", "within"): "state_excit_within",
    ("output", "across"): "state_excit_between",
    ("input", "within"): "field_excit_within",
    ("input", "across"): "field_excit_between",
}


@dataclass(frozen=True)
class PipelineSweepSettings:
    v_start: float = 0.0
    v_end: float = 1.0
    v_steps: int = 1000
    retry_step: float | None = None
    jobs: int = 1
    overwrite_simulation: bool = False
    plot_erfs: bool = False


@dataclass(frozen=True)
class BinaryRunSettings:
    warmup_steps: int | None = None
    simulation_steps: int | None = None
    sample_interval: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    output_name: str | None = None


@dataclass
class CorrelationRunResult:
    summary: Dict[str, Any]
    analysis_dir: str
    trace_paths: Sequence[str]
    network_summary: Dict[str, Dict[str, float]] | None = None


@dataclass(frozen=True)
class MultiInitCorrelationSpec:
    key: str
    label: str
    parameter: Dict[str, Any]
    binary_cfg: Dict[str, Any]
    bundle_path: str
    focus_counts: Sequence[int]
    stability_filter: str
    n_inits: int
    seed_inits: int
    seed_network: int
    stride_analysis: int
    max_pairs: int
    analysis_only: bool = False
    overwrite_simulation: bool = False
    overwrite_analysis: bool = False
    verbose: bool = True


@dataclass(frozen=True)
class InitTask:
    index: int
    group_key: str
    group_label: str
    candidate: Dict[str, Any]
    trace_path: str
    analysis_dir: str
    parameter: Dict[str, Any]
    binary_cfg: Dict[str, Any]
    stride: int
    max_pairs: int
    needs_simulation: bool
    needs_analysis: bool
    population_init_seed: int | None = None


def resolve_focus_counts(parameter: Dict[str, Any], explicit: Iterable[int] | None = None) -> List[int]:
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    if explicit:
        values = sorted({max(1, int(val)) for val in explicit})
        if values:
            return values
    return list(range(1, q_value + 1))


def resolve_binary_config(parameter: Dict[str, Any], overrides: BinaryRunSettings) -> Dict[str, Any]:
    cfg = dict(parameter.get("binary") or {})
    if overrides.warmup_steps is not None:
        cfg["warmup_steps"] = int(overrides.warmup_steps)
    else:
        cfg["warmup_steps"] = int(cfg.get("warmup_steps", 5000))
    if overrides.simulation_steps is not None:
        cfg["simulation_steps"] = int(overrides.simulation_steps)
    else:
        cfg["simulation_steps"] = int(cfg.get("simulation_steps", 20000))
    if overrides.sample_interval is not None:
        cfg["sample_interval"] = int(overrides.sample_interval)
    else:
        cfg["sample_interval"] = int(cfg.get("sample_interval", 10))
    if overrides.batch_size is not None:
        cfg["batch_size"] = int(overrides.batch_size)
    else:
        cfg["batch_size"] = int(cfg.get("batch_size", 200))
    cfg["seed"] = overrides.seed if overrides.seed is not None else cfg.get("seed")
    cfg["output_name"] = overrides.output_name or cfg.get("output_name", "activity_trace")
    return finalize_binary_config(parameter, cfg)


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
    r_j = parameter.get("R_j", 0.0)
    return os.path.join("data", f"all_fixpoints_{conn}_kappa{encoded_kappa}_Rj{r_j}_{tag}.pkl")


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
    folder = run_mean_field_simulation(args, param, r_eplus_values, focus_counts)
    if folder is None:
        folder = ensure_output_folder(param, tag=sim_tag_from_cfg(_filtered_parameter_for_tag(param)))
    bundle_path = compute_fixpoint_bundle_path(param)
    try:
        run_mean_field_analysis(folder, param, focus_counts, plot_erfs=sweep_cfg.plot_erfs)
    except FileNotFoundError as exc:
        if "No .pkl files found" not in str(exc):
            raise
        empty_payload = {
            "metadata": {
                "source_folder": os.path.abspath(folder),
                "analysis_parameter": dict(param),
                "analysis_focus_counts": list(focus_counts),
                "analysis_error": str(exc),
                "mean_field_completed": False,
            },
            "fixpoints": {},
        }
        with open(bundle_path, "wb") as handle:
            pickle.dump(empty_payload, handle)
        return folder, bundle_path
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Fixpoint bundle {bundle_path} was not generated. Ensure the mean-field pipeline completed successfully."
        )
    return folder, bundle_path


def _load_fixpoint_bundle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a fixpoint dictionary payload.")
    if "metadata" not in payload or "fixpoints" not in payload:
        raise ValueError(f"{path} is missing required fixpoint metadata.")
    return payload


def _parse_rep_from_key(key: str) -> float | None:
    if "_focus" not in key:
        return None
    prefix, _, _ = key.partition("_focus")
    try:
        return float(prefix)
    except ValueError:
        return None


def _candidate_sort_key(entry: Dict[str, Any]) -> Tuple[int, int, float, int]:
    value = float(entry.get("value", float("nan")))
    finite_flag = 0 if math.isfinite(value) else 1
    safe_value = value if math.isfinite(value) else 0.0
    return (int(entry["focus_count"]), finite_flag, safe_value, int(entry["index"]))


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
    mode = str(stability_filter or "stable").lower()
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
            rep_value = _parse_rep_from_key(str(rep_label))
            if rep_value is None or abs(rep_value - target_rep) > 1e-9:
                continue
            for idx, (fp_value, fixpoint) in enumerate(sorted(rep_entries.items(), key=lambda item: float(item[0]))):
                rates = fixpoint.get("rates")
                if rates is None:
                    continue
                values = np.asarray(rates, dtype=float).ravel()
                if values.size != expected_length:
                    raise ValueError(
                        f"Fixpoint {rep_label} lists {values.size} populations, but the network expects {expected_length}."
                    )
                stability = str(fixpoint.get("stability", "") or "").lower() or "unknown"
                if mode == "stable" and stability != "stable":
                    continue
                if mode == "unstable" and stability == "stable":
                    continue
                try:
                    value = float(fp_value)
                except (TypeError, ValueError):
                    value = float("nan")
                selection.append(
                    {
                        "id": f"{rep_label}_{idx}",
                        "focus_count": focus_count,
                        "index": idx,
                        "value": value,
                        "rates": values.tolist(),
                        "stability": stability,
                        "rep_label": rep_label,
                    }
                )
    if not selection:
        focus_msg = f"focus_counts {sorted(allowed)}" if allowed else "all focus_counts"
        raise ValueError(f"No fixpoints matched {focus_msg} with stability filter '{mode}'.")
    selection.sort(key=_candidate_sort_key)
    return selection


def _candidate_max_excit_rate(candidate: Dict[str, Any], excitatory_count: int) -> float:
    rates = np.asarray(candidate.get("rates", []), dtype=float)
    if rates.size < excitatory_count:
        return float("nan")
    excit = rates[:excitatory_count]
    if excit.size == 0:
        return float("nan")
    return float(np.nanmax(excit))


def _deduplicate_candidates_by_focus(
    candidates: Sequence[Dict[str, Any]],
    excitatory_count: int,
) -> List[Dict[str, Any]]:
    best_per_focus: Dict[int, Dict[str, Any]] = {}
    for entry in candidates:
        focus = int(entry.get("focus_count", 0) or 0)
        max_rate = _candidate_max_excit_rate(entry, excitatory_count)
        if not math.isfinite(max_rate):
            continue
        existing = best_per_focus.get(focus)
        if existing is None or max_rate > existing.get("max_excitatory_rate", -float("inf")):
            cloned = dict(entry)
            cloned["max_excitatory_rate"] = max_rate
            best_per_focus[focus] = cloned
    reduced = list(best_per_focus.values())
    reduced.sort(key=_candidate_sort_key)
    return reduced


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


def _resolve_fixpoint_candidates(
    parameter: Dict[str, Any],
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
) -> List[Dict[str, Any]]:
    bundle = _load_fixpoint_bundle(bundle_path)
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    r_eplus = parameter.get("R_Eplus")
    if r_eplus is None:
        raise ValueError("Parameter 'R_Eplus' must be set before selecting fixed points.")
    raw = _load_fixpoint_candidates(bundle, focus_counts, stability_filter, 2 * q_value, float(r_eplus))
    reduced = _deduplicate_candidates_by_focus(raw, q_value)
    if not reduced:
        raise ValueError("No fixpoint candidates matched the requested filters.")
    return reduced


def _has_any_fixpoint_candidates(
    parameter: Dict[str, Any],
    bundle_path: str,
    focus_counts: Sequence[int],
) -> bool:
    bundle = _load_fixpoint_bundle(bundle_path)
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        return False
    r_eplus = parameter.get("R_Eplus")
    if r_eplus is None:
        return False
    try:
        candidates = _load_fixpoint_candidates(bundle, focus_counts, "any", 2 * q_value, float(r_eplus))
    except ValueError:
        return False
    return bool(candidates)


def _uniform_init_candidate(parameter: Dict[str, Any], value: float = 0.1) -> Dict[str, Any]:
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    rng = np.random.default_rng()
    low = max(0.0, float(value) - 0.2)
    high = min(1.0, float(value) + 0.2)
    rates = rng.uniform(low=low, high=high, size=2 * q_value).astype(float)
    return {
        "id": f"uniform_{float(value):.3f}",
        "focus_count": 0,
        "index": 0,
        "value": float("nan"),
        "rates": rates.tolist(),
        "stability": "mf_fallback",
        "rep_label": "uniform_init",
        "source": "uniform_fallback",
    }


def _binary_output_folder(parameter: Dict[str, Any], binary_cfg: Dict[str, Any]) -> str:
    folder = ensure_output_folder(parameter, tag=sim_tag_from_cfg(_filtered_parameter_for_tag(parameter)))
    binary_tag = sim_tag_from_cfg({"parameter": dict(parameter), "binary": dict(binary_cfg)})
    binary_dir = os.path.join(folder, "binary", binary_tag)
    os.makedirs(binary_dir, exist_ok=True)
    return binary_dir


def _load_trace_payload(trace_path: str) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace {trace_path} does not exist.")
    with np.load(trace_path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{trace_path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
        times = np.asarray(data["times"], dtype=float) if "times" in data else np.arange(rates.shape[0], dtype=float)
        states = (
            np.asarray(data["neuron_states"], dtype=np.uint8)
            if "neuron_states" in data
            else np.zeros((0, 0), dtype=np.uint8)
        )
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_interval = int(np.asarray(data.get("neuron_state_interval", 0)).item())
        state_times = (
            np.asarray(data["neuron_state_times"], dtype=np.int64)
            if "neuron_state_times" in data
            else np.zeros((0,), dtype=np.int64)
        )
        subthreshold = (
            np.asarray(data["subthreshold_fields"], dtype=np.float32)
            if "subthreshold_fields" in data
            else np.zeros((0, 0), dtype=np.float32)
        )
        state_updates = np.asarray(data["state_updates"], dtype=np.uint16) if "state_updates" in data else None
        state_deltas = np.asarray(data["state_deltas"], dtype=np.int8) if "state_deltas" in data else None
        initial_state = np.asarray(data["initial_state"], dtype=np.uint8) if "initial_state" in data else None
        spike_times = np.asarray(data["spike_times"], dtype=float).ravel() if "spike_times" in data else np.zeros(0, dtype=float)
        spike_ids = np.asarray(data["spike_ids"], dtype=np.int64).ravel() if "spike_ids" in data else np.zeros(0, dtype=np.int64)
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "sample_interval": sample_interval,
        "state_interval": state_interval,
        "state_times": state_times,
        "subthreshold_fields": subthreshold,
        "state_updates": state_updates,
        "state_deltas": state_deltas,
        "initial_state": initial_state,
        "spike_times": spike_times,
        "spike_ids": spike_ids,
    }


def _instantiate_replay_network(parameter: Dict[str, Any], binary_cfg: Dict[str, Any]) -> ClusteredEI_network:
    seed = binary_cfg.get("seed")
    if seed is None:
        raise ValueError("binary.seed must be defined to reconstruct connectivity-dependent fields.")
    np.random.seed(int(seed))
    weight_dtype = np.float64 if str(binary_cfg.get("weight_dtype", "float32")).lower() == "float64" else np.float32
    network = ClusteredEI_network(parameter)
    network.initialize(
        weight_mode=str(binary_cfg.get("weight_mode", "auto")),
        ram_budget_gb=float(binary_cfg.get("ram_budget_gb", 12.0) or 12.0),
        weight_dtype=weight_dtype,
    )
    return network


def _fields_from_states(network: ClusteredEI_network, states: np.ndarray, *, chunk_size: int = 8) -> np.ndarray:
    if states.ndim != 2 or states.size == 0:
        return np.zeros((0, network.N), dtype=np.float32)
    chunk = max(1, int(chunk_size))
    outputs: List[np.ndarray] = []
    for start in range(0, states.shape[0], chunk):
        end = min(states.shape[0], start + chunk)
        block = states[start:end].astype(np.float32, copy=False).T
        if network.weight_mode == "dense" and network.weights_dense is not None:
            fields = network.weights_dense @ block
        elif network.weights_csc is not None:
            fields = network.weights_csc @ block
        else:
            raise RuntimeError("BinaryNetwork weights were not initialized correctly.")
        outputs.append(np.asarray(fields.T, dtype=np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, network.N), dtype=np.float32)


def _assembly_membership(parameter: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    n_e = int(parameter.get("N_E", 0) or 0)
    n_i = int(parameter.get("N_I", 0) or 0)
    if n_e % q_value != 0 or n_i % q_value != 0:
        raise ValueError(f"N_E ({n_e}) and N_I ({n_i}) must be divisible by Q ({q_value}).")
    excit_size = n_e // q_value
    inhib_size = n_i // q_value
    assembly_names = [f"E{idx + 1}" for idx in range(q_value)] + [f"I{idx + 1}" for idx in range(q_value)]
    membership: List[int] = []
    for idx in range(q_value):
        membership.extend([idx] * excit_size)
    for idx in range(q_value):
        membership.extend([q_value + idx] * inhib_size)
    return np.asarray(membership, dtype=np.int64), assembly_names


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
    indices_by_assembly = {assembly: base_indices[filtered_ids == assembly] for assembly in unique_ids}
    pairs: List[Tuple[int, int]] = []
    if within:
        per_bucket = max(1, max_pairs // max(1, len(unique_ids)))
        for indices in indices_by_assembly.values():
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
    unique = np.unique(np.asarray(indices, dtype=np.int64))
    subset = data[:, unique].astype(np.float32, copy=True)
    means = subset.mean(axis=0)
    stds = subset.std(axis=0)
    valid = stds > 0
    valid_indices = np.nonzero(valid)[0]
    if valid_indices.size:
        subset[:, valid_indices] = (subset[:, valid_indices] - means[valid_indices]) / stds[valid_indices]
    mapping = {int(unique[idx]): int(idx) for idx in range(unique.size)}
    return subset, mapping, valid


def _centered_columns(data: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    if not indices:
        return np.zeros((data.shape[0], 0), dtype=np.float32), {}
    unique = np.unique(np.asarray(indices, dtype=np.int64))
    subset = data[:, unique].astype(np.float32, copy=True)
    subset -= subset.mean(axis=0)
    mapping = {int(unique[idx]): int(idx) for idx in range(unique.size)}
    return subset, mapping


def _compute_pairwise_correlations(data: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
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
        corr_values.append(float(np.mean(subset[:, first] * subset[:, second])))
    return np.asarray(corr_values, dtype=np.float32)


def _compute_pairwise_covariances(data: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    if not pairs or data.size == 0:
        return np.zeros((0,), dtype=np.float32)
    flat_indices = [idx for pair in pairs for idx in pair]
    subset, mapping = _centered_columns(data, flat_indices)
    if subset.size == 0:
        return np.zeros((0,), dtype=np.float32)
    cov_values: List[float] = []
    for i, j in pairs:
        first = mapping.get(i)
        second = mapping.get(j)
        if first is None or second is None:
            continue
        cov_values.append(float(np.mean(subset[:, first] * subset[:, second])))
    return np.asarray(cov_values, dtype=np.float32)


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
        zeros = np.zeros((len(assembly_names),), dtype=np.float32)
        return {
            "mu_emp": zeros.copy(),
            "var_total": zeros.copy(),
            "var_temporal": zeros.copy(),
            "var_quenched": zeros.copy(),
        }
    per_neuron_mean = values.mean(axis=0)
    per_neuron_var = values.var(axis=0)
    mu_emp: List[float] = []
    var_total: List[float] = []
    var_temporal: List[float] = []
    var_quenched: List[float] = []
    for idx, _ in enumerate(assembly_names):
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
    return np.asarray(candidate.get("rates", []), dtype=np.float32)


def _sampled_states_from_payload(payload: Dict[str, Any], *, stride: int) -> np.ndarray:
    states = np.asarray(payload.get("states"), dtype=np.uint8)
    if states.ndim == 2 and states.size:
        base_states = states
    else:
        updates = payload.get("state_updates")
        deltas = payload.get("state_deltas")
        initial_state = payload.get("initial_state")
        if updates is None or deltas is None or initial_state is None:
            return np.zeros((0, 0), dtype=np.uint8)
        # The saved diff logs are already decimated to the recording interval.
        # Reconstruct every stored column here and only apply the analysis
        # stride below; otherwise the trace collapses to a single sample.
        base_states = ClusteredEI_network.reconstruct_states_from_diff_logs(
            initial_state,
            updates,
            deltas,
            sample_interval=1,
        )
    step = max(1, int(stride))
    return base_states[::step] if step > 1 else base_states


def _sampled_fields_from_trace(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    stride: int,
) -> np.ndarray:
    fields = np.asarray(payload.get("subthreshold_fields"), dtype=np.float32)
    if fields.ndim == 2 and fields.size:
        step = max(1, int(stride))
        return fields[::step] if step > 1 else fields
    states = _sampled_states_from_payload(payload, stride=stride)
    if states.ndim != 2 or states.size == 0:
        neuron_count = int(parameter.get("N_E", 0) or 0) + int(parameter.get("N_I", 0) or 0)
        return np.zeros((0, neuron_count), dtype=np.float32)
    network = _instantiate_replay_network(parameter, binary_cfg)
    return _fields_from_states(network, states)


def _analyze_trace(
    trace_path: str,
    init_index: int,
    candidate: Dict[str, Any],
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    analysis_dir: str,
    *,
    stride: int,
    max_pairs: int,
) -> Dict[str, Any]:
    payload = _load_trace_payload(trace_path)
    assembly_ids, assembly_names = _assembly_membership(parameter)
    neuron_count = assembly_ids.size
    states_ds = _sampled_states_from_payload(payload, stride=stride).astype(np.float32, copy=False)
    fields_ds = _sampled_fields_from_trace(parameter, binary_cfg, payload, stride=stride).astype(np.float32, copy=False)
    rng = np.random.default_rng(init_index + 17)
    if states_ds.shape[1] != neuron_count:
        padded = np.zeros((states_ds.shape[0], neuron_count), dtype=np.float32)
        padded[:, : min(states_ds.shape[1], neuron_count)] = states_ds[:, : min(states_ds.shape[1], neuron_count)]
        states_ds = padded
    if fields_ds.shape[1] != neuron_count:
        padded = np.zeros((fields_ds.shape[0], neuron_count), dtype=np.float32)
        padded[:, : min(fields_ds.shape[1], neuron_count)] = fields_ds[:, : min(fields_ds.shape[1], neuron_count)]
        fields_ds = padded
    q_value = int(parameter.get("Q", len(assembly_names) // 2) or 0)
    excit_mask = assembly_ids < q_value
    inhib_mask = assembly_ids >= q_value
    correlations: Dict[str, Any] = {
        "state_excit_within": np.zeros((0,), dtype=np.float32),
        "state_excit_between": np.zeros((0,), dtype=np.float32),
        "state_inhib_within": np.zeros((0,), dtype=np.float32),
        "state_inhib_between": np.zeros((0,), dtype=np.float32),
        "field_excit_within": np.zeros((0,), dtype=np.float32),
        "field_excit_between": np.zeros((0,), dtype=np.float32),
        "field_inhib_within": np.zeros((0,), dtype=np.float32),
        "field_inhib_between": np.zeros((0,), dtype=np.float32),
        "field_excit_within_cov": np.zeros((0,), dtype=np.float32),
        "field_excit_between_cov": np.zeros((0,), dtype=np.float32),
        "field_inhib_within_cov": np.zeros((0,), dtype=np.float32),
        "field_inhib_between_cov": np.zeros((0,), dtype=np.float32),
    }
    if states_ds.size:
        within_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=excit_mask)
        between_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=excit_mask)
        inhib_within = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=inhib_mask)
        inhib_between = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=inhib_mask)
        correlations["state_excit_within"] = _compute_pairwise_correlations(states_ds, within_pairs)
        correlations["state_excit_between"] = _compute_pairwise_correlations(states_ds, between_pairs)
        correlations["state_inhib_within"] = _compute_pairwise_correlations(states_ds, inhib_within)
        correlations["state_inhib_between"] = _compute_pairwise_correlations(states_ds, inhib_between)
    if fields_ds.size:
        excit_within_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=excit_mask)
        excit_between_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=excit_mask)
        inhib_within_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=inhib_mask)
        inhib_between_pairs = _pair_index_sample(assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=inhib_mask)
        correlations["field_excit_within"] = _compute_pairwise_correlations(fields_ds, excit_within_pairs)
        correlations["field_excit_between"] = _compute_pairwise_correlations(fields_ds, excit_between_pairs)
        correlations["field_inhib_within"] = _compute_pairwise_correlations(fields_ds, inhib_within_pairs)
        correlations["field_inhib_between"] = _compute_pairwise_correlations(fields_ds, inhib_between_pairs)
        correlations["field_excit_within_cov"] = _compute_pairwise_covariances(fields_ds, excit_within_pairs)
        correlations["field_excit_between_cov"] = _compute_pairwise_covariances(fields_ds, excit_between_pairs)
        correlations["field_inhib_within_cov"] = _compute_pairwise_covariances(fields_ds, inhib_within_pairs)
        correlations["field_inhib_between_cov"] = _compute_pairwise_covariances(fields_ds, inhib_between_pairs)
    target_for_variance = fields_ds if fields_ds.size else states_ds
    variance_stats = _compute_variance_decomposition(target_for_variance, assembly_ids, assembly_names)
    cluster_indices = list(range(len(assembly_names)))
    state_cluster_mean, state_cluster_median = _cluster_pair_correlation_stats(
        states_ds, assembly_ids, cluster_indices, rng=rng, max_pairs=max_pairs
    )
    field_cluster_mean, field_cluster_median = _cluster_pair_correlation_stats(
        fields_ds, assembly_ids, cluster_indices, rng=rng, max_pairs=max_pairs
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
        **{key: value for key, value in correlations.items()},
        state_cluster_mean=state_cluster_mean.astype(np.float32, copy=False),
        state_cluster_median=state_cluster_median.astype(np.float32, copy=False),
        field_cluster_mean=field_cluster_mean.astype(np.float32, copy=False),
        field_cluster_median=field_cluster_median.astype(np.float32, copy=False),
        candidate_prediction=prediction,
    )
    return {
        **variance_stats,
        **correlations,
        "state_cluster_mean": state_cluster_mean,
        "state_cluster_median": state_cluster_median,
        "field_cluster_mean": field_cluster_mean,
        "field_cluster_median": field_cluster_median,
        "analysis_path": per_init_path,
    }


def _load_analysis_file(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["analysis_path"] = path
    return payload


def _network_summary_path(analysis_dir: str) -> str:
    return os.path.join(analysis_dir, NETWORK_SUMMARY_FILE)


def _normalize_network_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for measure in MEASURES:
        normalized[measure] = {}
        source = summary.get(measure, {})
        for category in CATEGORIES:
            value = source.get(category, float("nan"))
            normalized[measure][category] = float(value)
    return normalized


def _fisher_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    clipped = np.clip(finite, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS)
    return float(np.mean(np.arctanh(clipped)))


def _summarize_init_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {
        measure: {category: float("nan") for category in CATEGORIES}
        for measure in MEASURES
    }
    for (measure, category), key in MEASURE_KEY_MAP.items():
        values = payload.get(key)
        summary[measure][category] = _fisher_mean(values if values is not None else np.zeros(0, dtype=float))
    return summary


def _merge_init_summaries(entries: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {
        measure: {category: float("nan") for category in CATEGORIES}
        for measure in MEASURES
    }
    if not entries:
        return merged
    for measure in MEASURES:
        for category in CATEGORIES:
            values = [entry[measure][category] for entry in entries if np.isfinite(entry[measure][category])]
            merged[measure][category] = float(np.mean(values)) if values else float("nan")
    return merged


def _save_network_summary(analysis_dir: str, summary: Dict[str, Dict[str, float]]) -> str:
    payload: Dict[str, Dict[str, float | None]] = {}
    normalized = _normalize_network_summary(summary)
    for measure, categories in normalized.items():
        payload[measure] = {}
        for category, value in categories.items():
            payload[measure][category] = float(value) if np.isfinite(value) else None
    path = _network_summary_path(analysis_dir)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _load_network_summary(analysis_dir: str) -> Dict[str, Dict[str, float]]:
    path = _network_summary_path(analysis_dir)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary: Dict[str, Dict[str, float]] = {}
    for measure in MEASURES:
        summary[measure] = {}
        categories = payload.get(measure, {})
        for category in CATEGORIES:
            raw_value = categories.get(category)
            summary[measure][category] = float("nan") if raw_value is None else float(raw_value)
    return summary


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
        "field_excit_within_cov",
        "field_excit_between_cov",
        "field_inhib_within_cov",
        "field_inhib_between_cov",
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
        analysis_paths.append(str(analysis.get("analysis_path", "")))
        trace_paths.append(str(entry.get("trace_path", "")))
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
        summary[key] = np.nanmean(np.stack(values), axis=0) if values else np.zeros((0, 0), dtype=np.float32)
    summary["analysis_paths"] = np.asarray(analysis_paths, dtype=object)
    summary["trace_paths"] = np.asarray(trace_paths, dtype=object)
    summary["candidates"] = np.asarray(candidate_info, dtype=object)
    return summary


def _network_summary_from_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    compact_entries: List[Dict[str, Dict[str, float]]] = []
    for entry in results:
        if not entry.get("success"):
            continue
        compact = entry.get("compact_summary")
        if compact is None:
            analysis = entry.get("analysis") or {}
            compact = _summarize_init_payload(analysis)
        compact_entries.append(_normalize_network_summary(compact))
    return _merge_init_summaries(compact_entries)


def _load_saved_summary(analysis_dir: str) -> Dict[str, Any]:
    path = os.path.join(analysis_dir, "pooled_summary.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _save_summary(analysis_dir: str, summary: Dict[str, Any]) -> str:
    path = os.path.join(analysis_dir, "pooled_summary.npz")
    np.savez_compressed(path, **summary)
    return path


def _save_metadata(path: str, payload: Dict[str, Any]) -> None:
    write_yaml_config(payload, path)


def _prepare_metadata(
    analysis_dir: str,
    *,
    base_output: str,
    fixpoints_path: str,
    focus_counts: Sequence[int] | None,
    stability_filter: str,
    seed_network: int,
    binary_cfg: Dict[str, Any],
    target_rep: float | None,
) -> Dict[str, Any]:
    metadata_path = os.path.join(analysis_dir, "metadata.yaml")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            import yaml
            metadata = yaml.safe_load(handle) or {}
    metadata["base_output_name"] = base_output
    metadata["mode"] = "binary_network_multi_init"
    metadata["focus_counts"] = list(focus_counts) if focus_counts is not None else None
    metadata["fixpoints_file"] = os.path.abspath(fixpoints_path)
    metadata["stability_filter"] = stability_filter
    metadata["network_seed"] = int(seed_network)
    metadata["sample_interval"] = int(binary_cfg.get("sample_interval", 1) or 1)
    metadata["warmup_steps"] = int(binary_cfg.get("warmup_steps", 0) or 0)
    metadata["record_steps"] = int(binary_cfg.get("simulation_steps", 0) or 0)
    metadata["base_seed"] = int(seed_network)
    metadata["R_Eplus"] = float(target_rep) if target_rep is not None else None
    _save_metadata(metadata_path, metadata)
    return metadata


def _prepare_multi_init_tasks(spec: MultiInitCorrelationSpec) -> tuple[str, List[InitTask]]:
    binary_cfg = dict(spec.binary_cfg)
    binary_cfg["seed"] = int(spec.seed_network)
    target_rep = spec.parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError("Parameter 'R_Eplus' must be set before running the correlation workflow.")
    fallback_to_uniform = False
    try:
        candidates = _resolve_fixpoint_candidates(spec.parameter, spec.bundle_path, spec.focus_counts, spec.stability_filter)
        picks = _select_fixpoints(candidates, seed=int(spec.seed_inits), count=max(0, int(spec.n_inits)))
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and _has_any_fixpoint_candidates(spec.parameter, spec.bundle_path, spec.focus_counts):
            raise
        fallback_to_uniform = True
        if spec.verbose:
            print(
                "No mean-field fixpoints available for "
                f"R_Eplus={float(target_rep):.4f}; falling back to uniform random init m=0.5."
            )
        base_candidate = _uniform_init_candidate(spec.parameter, value=0.5)
        picks = []
        for idx in range(max(0, int(spec.n_inits))):
            candidate = dict(base_candidate)
            candidate["id"] = f"{base_candidate['id']}_init{idx:04d}"
            candidate["index"] = idx
            candidate["population_init_seed"] = int(spec.seed_inits) + idx
            picks.append(candidate)
    if not picks:
        raise RuntimeError("No initialization tasks were prepared for the correlation workflow.")
    binary_dir = _binary_output_folder(spec.parameter, binary_cfg)
    analysis_dir = os.path.join(binary_dir, CORR_ANALYSIS_SUBDIR)
    os.makedirs(analysis_dir, exist_ok=True)
    _prepare_metadata(
        analysis_dir,
        base_output=str(binary_cfg.get("output_name", "activity_trace")),
        fixpoints_path=spec.bundle_path,
        focus_counts=spec.focus_counts,
        stability_filter=spec.stability_filter,
        seed_network=int(spec.seed_network),
        binary_cfg=binary_cfg,
        target_rep=float(target_rep),
    )
    tasks: List[InitTask] = []
    for idx, candidate in enumerate(picks):
        label = f"{binary_cfg.get('output_name', 'activity_trace')}_init{idx:04d}"
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        analysis_path = os.path.join(analysis_dir, f"analysis_init{idx:04d}.npz")
        needs_simulation = (not spec.analysis_only) and (spec.overwrite_simulation or not os.path.exists(trace_path))
        needs_analysis = spec.overwrite_analysis or not os.path.exists(analysis_path)
        tasks.append(
            InitTask(
                index=idx,
                group_key=str(spec.key),
                group_label=str(spec.label),
                candidate=candidate,
                trace_path=trace_path,
                analysis_dir=analysis_dir,
                parameter=dict(spec.parameter),
                binary_cfg=dict(binary_cfg),
                stride=max(1, int(spec.stride_analysis or 1)),
                max_pairs=max(1, int(spec.max_pairs or DEFAULT_MAX_PAIRS)),
                needs_simulation=needs_simulation,
                needs_analysis=needs_analysis,
                population_init_seed=(
                    int(candidate["population_init_seed"])
                    if fallback_to_uniform and candidate.get("population_init_seed") is not None
                    else None
                ),
            )
        )
    return analysis_dir, tasks


def _process_task(task: InitTask) -> Dict[str, Any]:
    logs: List[str] = []
    success = True
    error: str | None = None
    try:
        if task.needs_simulation:
            label = os.path.splitext(os.path.basename(task.trace_path))[0]
            if task.candidate.get("source") == "uniform_fallback":
                logs.append(
                    f"Simulating init {task.index}: fallback uniform init p=0.1 "
                    f"(seed {task.population_init_seed})."
                )
            else:
                logs.append(
                    f"Simulating init {task.index}: fixpoint {task.candidate.get('id')} "
                    f"(focus {task.candidate.get('focus_count')}, stability {task.candidate.get('stability')})."
                )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_binary_simulation(
                    task.parameter,
                    task.binary_cfg,
                    output_name=label,
                    population_rate_inits=task.candidate["rates"],
                    population_init_seed=task.population_init_seed,
                )
            trace_path = str(result["trace_path"])
        else:
            trace_path = task.trace_path
        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"Trace {trace_path} missing for init {task.index}.")
        if task.needs_analysis:
            analysis = _analyze_trace(
                trace_path,
                task.index,
                task.candidate,
                task.parameter,
                task.binary_cfg,
                task.analysis_dir,
                stride=task.stride,
                max_pairs=task.max_pairs,
            )
        else:
            analysis_path = os.path.join(task.analysis_dir, f"analysis_init{task.index:04d}.npz")
            if not os.path.exists(analysis_path):
                raise FileNotFoundError(f"Analysis file {analysis_path} missing for init {task.index}.")
            analysis = _load_analysis_file(analysis_path)
        compact_summary = _summarize_init_payload(analysis)
    except Exception as exc:
        success = False
        error = str(exc)
        logs.append(f"Error: {exc}")
        trace_path = task.trace_path
        analysis = {}
        compact_summary = None
    return {
        "index": task.index,
        "group_key": task.group_key,
        "group_label": task.group_label,
        "candidate": task.candidate,
        "trace_path": trace_path,
        "analysis": analysis,
        "compact_summary": compact_summary,
        "success": success,
        "error": error,
        "logs": logs,
    }


def _execute_tasks(tasks: Sequence[InitTask], jobs: int) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    warm_numba_caches()
    if jobs <= 1:
        return [_process_task(task) for task in tasks]
    max_workers = min(int(jobs), len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_process_task, task): task for task in tasks}
        return [future.result() for future in concurrent.futures.as_completed(future_map)]


def _execute_tasks_with_progress(
    tasks: Sequence[InitTask],
    jobs: int,
    progress_callback: Callable[[Dict[str, Any], int, int], None] | None = None,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    warm_numba_caches()
    total = len(tasks)
    completed = 0
    results: List[Dict[str, Any]] = []
    if jobs <= 1:
        for task in tasks:
            entry = _process_task(task)
            results.append(entry)
            completed += 1
            if progress_callback is not None:
                progress_callback(entry, completed, total)
        return results
    max_workers = min(int(jobs), len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_process_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            entry = future.result()
            results.append(entry)
            completed += 1
            if progress_callback is not None:
                progress_callback(entry, completed, total)
    return results


def _log_task_results(results: Sequence[Dict[str, Any]]) -> None:
    for entry in results:
        idx = entry.get("index")
        label = str(entry.get("group_label", "task"))
        for line in entry.get("logs", []):
            print(f"[{label} init {idx:04d}] {line}")
        if not entry.get("success", False):
            print(f"[{label} init {idx:04d}] Failed: {entry.get('error')}")


def resolve_binary_trace_from_fixpoint(
    parameter: Dict[str, Any],
    *,
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
    binary_cfg: Dict[str, Any],
    init_seed: int,
    init_index: int,
    n_inits: int,
    analysis_only: bool = False,
    overwrite_simulation: bool = False,
) -> Dict[str, Any]:
    candidates = _resolve_fixpoint_candidates(parameter, bundle_path, focus_counts, stability_filter)
    picks = _select_fixpoints(candidates, seed=int(init_seed), count=max(1, int(n_inits)))
    if init_index < 0 or init_index >= len(picks):
        raise ValueError(f"Requested init_index={init_index} but only {len(picks)} initializations were selected.")
    candidate = picks[init_index]
    binary_dir = _binary_output_folder(parameter, binary_cfg)
    trace_label = f"{binary_cfg.get('output_name', 'activity_trace')}_init{init_index:04d}"
    trace_path = os.path.join(binary_dir, f"{trace_label}.npz")
    if os.path.exists(trace_path) and not overwrite_simulation:
        return _load_trace_payload(trace_path)
    if analysis_only and not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace {trace_path} does not exist. Re-run without --analysis-only.")
    if overwrite_simulation or not os.path.exists(trace_path):
        result = run_binary_simulation(
            parameter,
            binary_cfg,
            output_name=trace_label,
            population_rate_inits=candidate["rates"],
            population_init_seed=(
                int(candidate["population_init_seed"]) if candidate.get("population_init_seed") is not None else None
            ),
        )
        trace_path = str(result["trace_path"])
    return _load_trace_payload(trace_path)


def run_multi_init_correlation(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
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
    verbose: bool = True,
) -> CorrelationRunResult:
    analysis_dir, tasks = _prepare_multi_init_tasks(
        MultiInitCorrelationSpec(
            key=str(binary_cfg.get("output_name", "activity_trace")),
            label=str(binary_cfg.get("output_name", "activity_trace")),
            parameter=dict(parameter),
            binary_cfg=dict(binary_cfg),
            bundle_path=bundle_path,
            focus_counts=tuple(focus_counts),
            stability_filter=stability_filter,
            n_inits=int(n_inits),
            seed_inits=int(seed_inits),
            seed_network=int(seed_network),
            stride_analysis=max(1, int(stride_analysis or 1)),
            max_pairs=max(1, int(max_pairs or DEFAULT_MAX_PAIRS)),
            analysis_only=analysis_only,
            overwrite_simulation=overwrite_simulation,
            overwrite_analysis=overwrite_analysis,
            verbose=verbose,
        )
    )
    results = _execute_tasks(tasks, max(1, int(jobs)))
    if verbose:
        _log_task_results(results)
    summary = _collect_results(results)
    _save_summary(analysis_dir, summary)
    network_summary = _network_summary_from_results(results)
    _save_network_summary(analysis_dir, network_summary)
    return CorrelationRunResult(
        summary=summary,
        analysis_dir=analysis_dir,
        trace_paths=summary.get("trace_paths", []),
        network_summary=network_summary,
    )


def run_multi_init_correlation_batch(
    specs: Sequence[MultiInitCorrelationSpec],
    *,
    jobs: int,
    progress_callback: Callable[[Dict[str, Any], int, int], None] | None = None,
    verbose: bool = True,
) -> Dict[str, CorrelationRunResult]:
    if not specs:
        return {}
    prepared: List[tuple[MultiInitCorrelationSpec, str, List[InitTask], Dict[str, Dict[str, float]] | None]] = []
    all_tasks: List[InitTask] = []
    for spec in specs:
        analysis_dir, tasks = _prepare_multi_init_tasks(spec)
        cached_network_summary = None
        if tasks and all((not task.needs_simulation) and (not task.needs_analysis) for task in tasks):
            try:
                cached_network_summary = _load_network_summary(analysis_dir)
            except FileNotFoundError:
                cached_network_summary = None
        prepared.append((spec, analysis_dir, tasks, cached_network_summary))
        if cached_network_summary is None:
            all_tasks.extend(tasks)
    results = _execute_tasks_with_progress(all_tasks, max(1, int(jobs)), progress_callback=progress_callback)
    if verbose:
        _log_task_results(results)
    grouped: Dict[str, List[Dict[str, Any]]] = {str(spec.key): [] for spec, _, _, _ in prepared}
    for entry in results:
        grouped.setdefault(str(entry.get("group_key", "")), []).append(entry)
    output: Dict[str, CorrelationRunResult] = {}
    for spec, analysis_dir, _, cached_network_summary in prepared:
        group_results = grouped.get(str(spec.key), [])
        if cached_network_summary is not None:
            try:
                summary = _load_saved_summary(analysis_dir)
            except FileNotFoundError:
                summary = _collect_results(group_results)
                _save_summary(analysis_dir, summary)
            network_summary = cached_network_summary
        else:
            summary = _collect_results(group_results)
            _save_summary(analysis_dir, summary)
            network_summary = _network_summary_from_results(group_results)
            _save_network_summary(analysis_dir, network_summary)
        output[str(spec.key)] = CorrelationRunResult(
            summary=summary,
            analysis_dir=analysis_dir,
            trace_paths=summary.get("trace_paths", []),
            network_summary=network_summary,
        )
    return output


def mean_connectivity(parameter: Dict[str, Any]) -> float:
    n_e = float(parameter.get("N_E", 0.0) or 0.0)
    n_i = float(parameter.get("N_I", 0.0) or 0.0)
    p_ee = float(parameter.get("p0_ee", 0.0) or 0.0)
    p_ei = float(parameter.get("p0_ei", 0.0) or 0.0)
    p_ie = float(parameter.get("p0_ie", 0.0) or 0.0)
    p_ii = float(parameter.get("p0_ii", 0.0) or 0.0)
    total = (n_e + n_i) ** 2
    if total <= 0:
        return 0.0
    numerator = (n_e ** 2) * p_ee + (n_e * n_i) * (p_ei + p_ie) + (n_i ** 2) * p_ii
    return numerator / total


def scale_connectivity(parameter: Dict[str, Any], target_connectivity: float) -> Dict[str, Any]:
    base_conn = mean_connectivity(parameter)
    if base_conn <= 0:
        return dict(parameter)
    scale = float(target_connectivity) / base_conn
    updated = dict(parameter)
    for key in ("p0_ee", "p0_ei", "p0_ie", "p0_ii"):
        value = float(updated.get(key, 0.0) or 0.0) * scale
        updated[key] = max(0.0, min(1.0, value))
    return updated
