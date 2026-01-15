from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from binary_pipeline import ensure_binary_behavior_defaults
from MeanField.rate_system import ensure_output_folder
from sim_config import deep_update, parse_overrides, sim_tag_from_cfg, write_yaml_config

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    yaml = None
    YAML_ERROR = exc
else:  # pragma: no cover - optional dependency
    YAML_ERROR = None

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
LEGACY_CODE_ROOT = os.path.join(REPO_ROOT, "legacy", "Rost", "code")
LEGACY_BINET_ROOT = os.path.join(LEGACY_CODE_ROOT, "BiNet", "src")
for legacy_path in (LEGACY_CODE_ROOT, LEGACY_BINET_ROOT):
    if os.path.isdir(legacy_path) and legacy_path not in sys.path:
        sys.path.append(legacy_path)
try:  # pragma: no cover - legacy dependency
    from sim_cluster_dynamics import simulate as legacy_simulate
except ModuleNotFoundError as exc:  # pragma: no cover - legacy dependency
    raise ModuleNotFoundError(
        "Could not import legacy/Rost/code/sim_cluster_dynamics.py. "
        "Build the legacy extensions before running this script."
    ) from exc

LEGACY_MODE_LABEL = "legacy_fixpoint_initialization"
MULTI_MODE_LABEL = "legacy_single_network_multi_init"
LEGACY_ANALYSIS_SUBDIR = "max_rates_distribution_fp_init_legacy"
CORR_ANALYSIS_SUBDIR = "single_network_multi_init_corr_var"
ACTIVE_GAP_TOLERANCE = 1e-4
DEFAULT_MAX_PAIRS = 200_000


def _load_yaml_file(path: str) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to read configuration files. Install it via 'pip install pyyaml'."
        ) from YAML_ERROR
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_params_from_folder(folder: str) -> Dict[str, Any]:
    params_path = os.path.join(folder, "params.yaml")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"{folder} does not contain params.yaml.")
    return _load_yaml_file(params_path)


def _apply_overrides(parameter: Dict[str, Any], overrides: Sequence[str] | None) -> Dict[str, Any]:
    if not overrides:
        return dict(parameter)
    updates = parse_overrides(overrides)
    return deep_update(parameter, updates)


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


def _find_fixpoint_bundle_for_folder(folder: str, hint: str | None = None) -> str:
    if hint:
        path = os.path.abspath(hint)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fixpoint file {path} does not exist.")
        return path
    folder = os.path.abspath(folder)
    folder_path = Path(folder)
    for parent in [folder_path] + list(folder_path.parents):
        for candidate in parent.glob("all_fixpoints_*.pkl"):
            try:
                bundle = _load_fixpoint_bundle(str(candidate))
            except Exception:
                continue
            source_folder = bundle.get("metadata", {}).get("source_folder")
            if source_folder and os.path.abspath(str(source_folder)) == folder:
                return str(candidate)
    raise FileNotFoundError(
        f"No all_fixpoints_*.pkl file references source folder {folder}. "
        "Use --fixpoints to specify the path explicitly."
    )


def _resolve_simulation_source(
    source: str,
    *,
    fixpoint_hint: str | None = None,
    overrides: Sequence[str] | None = None,
) -> Tuple[Dict[str, Any], str | None, str]:
    source_path = os.path.abspath(source)
    folder_override: str | None = None
    if os.path.isfile(source_path) and source_path.endswith(".pkl"):
        bundle = _load_fixpoint_bundle(source_path)
        metadata = bundle.get("metadata", {})
        parameter = metadata.get("analysis_parameter")
        if not isinstance(parameter, dict):
            raise ValueError(f"{source_path} does not define analysis parameters.")
        folder_override = metadata.get("source_folder")
        fixpoint_path = source_path
    elif os.path.isdir(source_path):
        parameter = _load_params_from_folder(source_path)
        folder_override = source_path
        fixpoint_path = _find_fixpoint_bundle_for_folder(source_path, fixpoint_hint)
    else:
        raise FileNotFoundError(f"{source_path} is neither a folder nor an all_fixpoints_*.pkl file.")
    resolved_parameter = _apply_overrides(parameter, overrides or [])
    return resolved_parameter, folder_override, fixpoint_path


def _resolve_binary_config(parameter: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(parameter.get("binary", {}))
    cfg["warmup_steps"] = args.warmup_steps if getattr(args, "warmup_steps", None) is not None else cfg.get("warmup_steps", 5000)
    cfg["simulation_steps"] = (
        args.simulation_steps if getattr(args, "simulation_steps", None) is not None else cfg.get("simulation_steps", 20000)
    )
    cfg["sample_interval"] = (
        args.sample_interval if getattr(args, "sample_interval", None) is not None else cfg.get("sample_interval", 10)
    )
    cfg["batch_size"] = getattr(args, "batch_size", None) if getattr(args, "batch_size", None) is not None else cfg.get("batch_size", 1)
    cfg["seed"] = args.seed if getattr(args, "seed", None) is not None else cfg.get("seed")
    cfg["output_name"] = getattr(args, "output_name", None) or cfg.get("output_name", "activity_trace")
    queue_cfg = cfg.get("update_queue")
    if queue_cfg is not None and not isinstance(queue_cfg, dict):
        raise ValueError("binary.update_queue must be a mapping when provided.")
    return ensure_binary_behavior_defaults(cfg)


def _filtered_parameter_for_tag(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    for key in ("R_Eplus", "focus_count", "focus_counts"):
        filtered.pop(key, None)
    return filtered


def _prepare_output_folders(parameter: Dict[str, Any], *, base_folder: str | None = None) -> Tuple[str, str, str]:
    filtered = _filtered_parameter_for_tag(parameter)
    if base_folder:
        folder = os.path.abspath(base_folder)
        os.makedirs(folder, exist_ok=True)
    else:
        tag = sim_tag_from_cfg(filtered)
        folder = ensure_output_folder(parameter, tag=tag)
    params_path = os.path.join(folder, "params.yaml")
    if not os.path.exists(params_path):
        write_yaml_config(filtered, params_path)
    binary_dir = os.path.join(folder, "binary")
    os.makedirs(binary_dir, exist_ok=True)
    analysis_dir = os.path.join(binary_dir, "max_rates_distribution")
    os.makedirs(analysis_dir, exist_ok=True)
    return folder, binary_dir, analysis_dir


def _format_seed_label(base_name: str, seed: int) -> str:
    trimmed = base_name.strip() or "activity_trace"
    return f"{trimmed}_seed{seed:06d}"


def _load_metadata(path: str) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to read metadata. Install it via 'pip install pyyaml'."
        ) from YAML_ERROR
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save_metadata(path: str, payload: Dict[str, Any]) -> None:
    write_yaml_config(payload, path)


def _load_trace_payload(trace_path: str) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace {trace_path} does not exist.")
    with np.load(trace_path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{trace_path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
        times = np.asarray(data.get("times"), dtype=float) if "times" in data else np.arange(rates.shape[0])
        states = np.asarray(data.get("neuron_states"), dtype=np.uint8) if "neuron_states" in data else np.zeros((0, 0), dtype=np.uint8)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_interval = int(np.asarray(data.get("neuron_state_interval", 0)).item())
        state_times = np.asarray(data.get("neuron_state_times"), dtype=np.int64) if "neuron_state_times" in data else np.zeros((0,), dtype=np.int64)
        subthreshold = (
            np.asarray(data.get("subthreshold_fields"), dtype=np.float32)
            if "subthreshold_fields" in data
            else np.zeros((0, 0), dtype=np.float32)
        )
        state_updates = np.asarray(data.get("state_updates"), dtype=np.uint16) if "state_updates" in data else None
        state_deltas = np.asarray(data.get("state_deltas"), dtype=np.int8) if "state_deltas" in data else None
        initial_state = np.asarray(data.get("initial_state"), dtype=np.uint8) if "initial_state" in data else None
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
    }


def _load_trace_rates(trace_path: str) -> Tuple[np.ndarray, List[str]]:
    payload = _load_trace_payload(trace_path)
    return payload["rates"], payload["names"]


def _compute_maxima_from_trace(trace_path: str, bin_samples: int) -> Tuple[List[float], int]:
    rates, names = _load_trace_rates(trace_path)
    if rates.ndim != 2 or rates.shape[0] == 0:
        raise ValueError(f"{trace_path} contains an empty rates array.")
    excit_indices = [idx for idx, name in enumerate(names) if str(name).startswith("E")]
    if not excit_indices:
        raise ValueError(f"{trace_path} does not include any excitatory populations.")
    excit_rates = rates[:, excit_indices]
    bin_samples = max(1, int(bin_samples))
    maxima: List[float] = []
    start = 0
    samples = excit_rates.shape[0]
    while start < samples:
        end = min(samples, start + bin_samples)
        chunk = excit_rates[start:end]
        if chunk.size == 0:
            break
        avg = chunk.mean(axis=0)
        maxima.append(float(avg.max()))
        start = end
    return maxima, len(excit_indices)


def _save_maxima_file(
    path: str,
    maxima: Sequence[float],
    seed: int,
    bin_size: int,
    trace_path: str,
    excitatory_clusters: int,
) -> None:
    payload = {
        "maxima": np.asarray(maxima, dtype=float),
        "seed": np.array(int(seed), dtype=np.int64),
        "bin_size": np.array(int(bin_size), dtype=np.int64),
        "trace_file": np.array(str(trace_path)),
        "excitatory_clusters": np.array(int(excitatory_clusters), dtype=np.int64),
    }
    np.savez_compressed(path, **payload)


def _load_maxima_file(path: str, expected_bin: int) -> Tuple[List[float], int] | Tuple[None, None]:
    try:
        with np.load(path, allow_pickle=True) as data:
            stored_bin = int(np.asarray(data["bin_size"]).item())
            maxima = [float(val) for val in np.asarray(data["maxima"], dtype=float).ravel()]
            excit = int(np.asarray(data["excitatory_clusters"]).item())
    except Exception:
        return None, None
    if stored_bin != expected_bin:
        return None, None
    return maxima, excit


def _write_trace_summary(
    trace_path: str,
    *,
    neuron_count: int,
    warmup_steps: int,
    simulation_steps: int,
    sample_interval: int,
    state_stride: int,
) -> None:
    summary = {
        "neurons": int(neuron_count),
        "warmup_steps": int(warmup_steps),
        "simulation_steps": int(simulation_steps),
        "sample_interval": int(sample_interval),
        "state_stride": int(state_stride),
        "state_chunks": {"enabled": False, "files": []},
    }
    summary_path = os.path.splitext(trace_path)[0] + "_summary.yaml"
    write_yaml_config(summary, summary_path)


def _prepare_multi_init_folder(parameter: Dict[str, Any], base_folder: str | None) -> Tuple[str, str, str]:
    folder, binary_dir, _ = _prepare_output_folders(parameter, base_folder=base_folder)
    analysis_dir = os.path.join(binary_dir, CORR_ANALYSIS_SUBDIR)
    os.makedirs(analysis_dir, exist_ok=True)
    return folder, binary_dir, analysis_dir


def _resolve_binary_cfg(parameter: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    args.warmup_steps = getattr(args, "warmup_sweeps", None)
    args.simulation_steps = getattr(args, "record_sweeps", None)
    args.sample_interval = getattr(args, "stride_sweeps", None)
    if not hasattr(args, "seed") or getattr(args, "seed", None) is None:
        args.seed = getattr(args, "seed_network", None)
    cfg = _resolve_binary_config(parameter, args)
    if getattr(args, "kappa", None) is not None:
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
        for _, indices in indices_by_assembly.items():
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


def _centered_columns(data: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    if not indices:
        return np.zeros((data.shape[0], 0), dtype=np.float32), {}
    unique = np.unique(np.asarray(indices, dtype=np.int64))
    subset = data[:, unique].astype(np.float32, copy=True)
    means = subset.mean(axis=0)
    subset -= means
    mapping = {int(unique[idx]): int(idx) for idx in range(unique.size)}
    return subset, mapping


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


def _compute_pairwise_covariances(
    data: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
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
        vec = subset[:, first] * subset[:, second]
        cov = float(np.mean(vec))
        cov_values.append(cov)
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
    payload = _load_trace_payload(trace_path)
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
        "field_excit_within_cov": np.zeros((0,), dtype=np.float32),
        "field_excit_between_cov": np.zeros((0,), dtype=np.float32),
        "field_inhib_within_cov": np.zeros((0,), dtype=np.float32),
        "field_inhib_between_cov": np.zeros((0,), dtype=np.float32),
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
        excit_within_pairs = _pair_index_sample(
            assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=excit_mask
        )
        excit_between_pairs = _pair_index_sample(
            assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=excit_mask
        )
        correlations["field_excit_within"] = _compute_pairwise_correlations(fields_ds, excit_within_pairs)
        correlations["field_excit_between"] = _compute_pairwise_correlations(fields_ds, excit_between_pairs)
        correlations["field_excit_within_cov"] = _compute_pairwise_covariances(fields_ds, excit_within_pairs)
        correlations["field_excit_between_cov"] = _compute_pairwise_covariances(fields_ds, excit_between_pairs)
        inhib_within_pairs = _pair_index_sample(
            assembly_ids, max_pairs=max_pairs, within=True, rng=rng, allowed_mask=inhib_mask
        )
        inhib_between_pairs = _pair_index_sample(
            assembly_ids, max_pairs=max_pairs, within=False, rng=rng, allowed_mask=inhib_mask
        )
        correlations["field_inhib_within"] = _compute_pairwise_correlations(fields_ds, inhib_within_pairs)
        correlations["field_inhib_between"] = _compute_pairwise_correlations(fields_ds, inhib_between_pairs)
        correlations["field_inhib_within_cov"] = _compute_pairwise_covariances(fields_ds, inhib_within_pairs)
        correlations["field_inhib_between_cov"] = _compute_pairwise_covariances(fields_ds, inhib_between_pairs)
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
            trace_path = run_legacy_binary_simulation(
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
    target_rep: float | None,
) -> Dict[str, Any]:
    metadata_path = os.path.join(analysis_dir, "metadata.yaml")
    metadata = _load_metadata(metadata_path) if os.path.exists(metadata_path) else {}
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
    if stored_mode not in {None, MULTI_MODE_LABEL}:
        raise ValueError(
            f"{analysis_dir} already contains mode '{stored_mode}'. Create a new folder for this workflow."
        )
    if stored_mode != MULTI_MODE_LABEL:
        metadata["mode"] = MULTI_MODE_LABEL
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
    stored_rep = metadata.get("R_Eplus")
    if target_rep is not None:
        rep_value = float(target_rep)
        if stored_rep is not None:
            try:
                stored_value = float(stored_rep)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metadata R_Eplus for {analysis_dir} is invalid: {stored_rep}") from exc
            if abs(stored_value - rep_value) > 1e-9:
                raise ValueError(
                    f"Analysis folder already recorded R_Eplus={stored_value}, requested {rep_value} incompatible."
                )
        if stored_rep is None:
            metadata["R_Eplus"] = rep_value
            changed = True
    elif stored_rep is not None:
        try:
            metadata["R_Eplus"] = float(stored_rep)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metadata R_Eplus for {analysis_dir} is invalid: {stored_rep}") from exc
    if changed:
        _save_metadata(metadata_path, metadata)
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
        needs_simulation = (not getattr(args, "analysis_only", False)) and (
            getattr(args, "overwrite_simulation", False) or not os.path.exists(trace_path)
        )
        needs_analysis = getattr(args, "overwrite_analysis", False) or not os.path.exists(analysis_path)
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
                stride=max(1, int(getattr(args, "stride_sweeps_analysis", 1) or 1)),
                max_pairs=max(1, int(getattr(args, "max_pairs", DEFAULT_MAX_PAIRS) or DEFAULT_MAX_PAIRS)),
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
    max_workers = min(int(jobs), len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_process_task, task): task for task in tasks}
        results = []
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
        return results




def _build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
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
    return parser


def _prepare_max_rate_folder(parameter: Dict[str, Any], base_folder: str | None) -> Tuple[str, str, str]:
    folder, binary_dir, _ = _prepare_output_folders(parameter, base_folder=base_folder)
    analysis_dir = os.path.join(binary_dir, LEGACY_ANALYSIS_SUBDIR)
    os.makedirs(analysis_dir, exist_ok=True)
    return folder, binary_dir, analysis_dir


def _shard_seed_list(seeds: Sequence[int], job_count: int, job_index: int) -> List[int]:
    job_count = int(job_count)
    if job_count <= 0:
        raise ValueError("--job-count must be positive.")
    job_index = int(job_index)
    if job_index < 0 or job_index >= job_count:
        raise ValueError("--job-index must satisfy 0 <= job_index < job_count.")
    if job_count == 1:
        return list(seeds)
    return [seed for idx, seed in enumerate(seeds) if idx % job_count == job_index]


def _normalize_focus_list(values: Sequence[int] | None) -> List[int] | None:
    if not values:
        return None
    return sorted({int(value) for value in values})


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
                        f"Fixpoint {rep_label} entry {rep_value} lists {values.size} populations, "
                        f"but the network expects {expected_length}."
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
        focus_msg = f"focus_counts {sorted(allowed)}" if allowed else "all focus_counts"
        raise ValueError(
            f"No fixpoints matched {focus_msg} with stability filter '{stability_filter}'."
        )
    selection.sort(key=_candidate_sort_key)
    return selection


def _candidate_for_seed(seed_value: int, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Fixpoint candidate list is empty.")
    rng_seed = int(seed_value) % (2**32)
    rng = np.random.default_rng(rng_seed)
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


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


def _active_cluster_count_gap(values: np.ndarray, tolerance: float) -> int:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0
    arr_sorted = np.sort(arr)[::-1]
    max_value = float(arr_sorted[0])
    if max_value <= tolerance:
        return 0
    augmented = np.concatenate([arr_sorted, [0.0]])
    gaps = augmented[:-1] - augmented[1:]
    max_gap = float(gaps.max(initial=-float("inf")))
    if max_gap <= tolerance:
        return int(arr_sorted.size)
    return max(1, min(arr_sorted.size, int(np.argmax(gaps) + 1)))


def _active_cluster_count_sign(values: np.ndarray, tolerance: float) -> int:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0
    mean_val = float(np.mean(arr))
    centered = arr - mean_val
    threshold = max(0.0, float(tolerance))
    if threshold > 0.0:
        active_mask = centered > threshold
    else:
        active_mask = centered > 0.0
    count = int(np.count_nonzero(active_mask))
    return max(0, min(arr.size, count))


def _active_cluster_count_from_means(
    values: Sequence[float],
    *,
    method: str = "gap",
    tolerance: float = ACTIVE_GAP_TOLERANCE,
) -> int:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0
    mode = (method or "gap").lower()
    if mode == "sign":
        return _active_cluster_count_sign(arr, tolerance)
    if mode != "gap":
        raise ValueError(f"Unknown active-count method '{method}'.")
    return _active_cluster_count_gap(arr, tolerance)


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


def _focus_payload_from_candidates(
    candidates: Sequence[Dict[str, Any]],
    excitatory_count: int,
) -> Dict[int, Dict[str, List[float]]]:
    payload: Dict[int, Dict[str, List[float]]] = {}
    for entry in candidates:
        focus = int(entry.get("focus_count", 0) or 0)
        max_rate = entry.get("max_excitatory_rate")
        if max_rate is None or not math.isfinite(float(max_rate)):
            max_rate = _candidate_max_excit_rate(entry, excitatory_count)
        if not math.isfinite(float(max_rate)):
            continue
        stability = str(entry.get("stability", "") or "").lower()
        bucket = payload.setdefault(focus, {"stable": [], "unstable": []})
        key = "stable" if stability == "stable" else "unstable"
        bucket[key] = [float(max_rate)]
    return payload


def _focus_expectations_from_payload(focus_rates: Dict[int, Dict[str, List[float]]]) -> Dict[int, float]:
    expectations: Dict[int, float] = {}
    for focus, payload in focus_rates.items():
        stable_vals = payload.get("stable", [])
        unstable_vals = payload.get("unstable", [])
        selected = stable_vals if stable_vals else unstable_vals
        if not selected:
            continue
        expectations[int(focus)] = float(np.mean(selected))
    return expectations


def _build_focus_color_map(plt_module, focus_rates: Dict[int, Any]) -> Dict[int, Any]:
    if not focus_rates:
        return {}
    focus_keys = sorted(int(key) for key in focus_rates)
    colors = plt_module.cm.tab10(np.linspace(0, 1, max(1, len(focus_keys))))
    return {focus: colors[idx % len(colors)] for idx, focus in enumerate(focus_keys)}


def _legacy_cluster_sizes(parameter: Dict[str, Any]) -> Tuple[int, int]:
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive for legacy simulations.")
    N_E = int(parameter.get("N_E", 0) or 0)
    N_I = int(parameter.get("N_I", 0) or 0)
    if N_E % Q_value != 0 or N_I % Q_value != 0:
        raise ValueError(
            f"Legacy binary simulations require N_E ({N_E}) and N_I ({N_I}) to be divisible by Q ({Q_value})."
        )
    return N_E // Q_value, N_I // Q_value


def build_population_rate_vector(parameter: Dict[str, Any], template: Any) -> List[float]:
    """Expand user-provided initial rates to a per-cluster vector of length 2 * Q."""
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    default_value = 0.1

    def _uniform(value: float | None) -> np.ndarray:
        val = default_value if value is None else float(value)
        return np.full(Q_value, np.clip(val, 0.0, 1.0), dtype=float)

    excit = inhib = None
    if template is None:
        excit = inhib = _uniform(None)
    elif isinstance(template, dict):
        excit = _uniform(template.get("excitatory"))
        inhib = _uniform(template.get("inhibitory"))
    elif isinstance(template, (list, tuple, np.ndarray)):
        arr = np.asarray(template, dtype=float).ravel()
        if arr.size == 2 * Q_value:
            return arr.tolist()
        if arr.size == 2:
            excit = _uniform(arr[0])
            inhib = _uniform(arr[1])
        elif arr.size == 1:
            excit = inhib = _uniform(arr[0])
        else:
            raise ValueError(
                f"population_rate_init sequence must have length 1, 2, or 2*Q ({2*Q_value}), got {arr.size}."
            )
    else:
        excit = inhib = _uniform(template)
    if excit is None or inhib is None:
        raise ValueError("Could not infer population_rate_init values.")
    return np.concatenate([excit, inhib]).tolist()


def _initial_state_vector(parameter: Dict[str, Any], rates: Sequence[float], *, permutation_seed: int | None = None) -> np.ndarray:
    Q_value = int(parameter.get("Q", 0) or 0)
    values = np.asarray(rates, dtype=float).ravel()
    if values.size != 2 * Q_value:
        raise ValueError(f"population_rate_inits must list {2 * Q_value} entries, got {values.size}.")
    excit_size, inhib_size = _legacy_cluster_sizes(parameter)
    excit_probs = np.clip(values[:Q_value], 0.0, 1.0)
    inhib_probs = np.clip(values[Q_value:], 0.0, 1.0)
    if Q_value > 1 and permutation_seed is not None:
        rng = np.random.default_rng(int(permutation_seed))
        perm = rng.permutation(Q_value)
        excit_probs = excit_probs[perm]
        inhib_probs = inhib_probs[perm]
    excit_vector = np.repeat(excit_probs, excit_size)
    inhib_vector = np.repeat(inhib_probs, inhib_size)
    return np.concatenate([excit_vector, inhib_vector]).astype(np.float32)


def _build_legacy_parameters(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    seed: int,
    init_rates: Sequence[float],
    *,
    capture_spikes: bool = False,
    cluster_seed: int | None = None,
) -> Dict[str, Any]:
    required = ("N_E", "N_I", "Q", "V_th", "g", "p0_ee", "p0_ie", "p0_ei", "p0_ii", "tau_e")
    for key in required:
        if key not in parameter:
            raise ValueError(f"Parameter '{key}' is required for legacy simulations.")
    N_E = int(parameter["N_E"])
    N_I = int(parameter["N_I"])
    V_th = float(parameter["V_th"])
    g_value = float(parameter["g"])
    tau_e = float(parameter["tau_e"])
    m_x = float(parameter.get("m_X", 0.0) or 0.0)
    p0_ee = float(parameter["p0_ee"])
    p0_ie = float(parameter["p0_ie"])
    p0_ei = float(parameter["p0_ei"])
    p0_ii = float(parameter["p0_ii"])
    Q_value = int(parameter["Q"])
    R_Eplus = parameter.get("R_Eplus")
    if R_Eplus is None:
        raise ValueError("Parameter 'R_Eplus' must be set for legacy simulations.")
    R_Eplus = float(R_Eplus)
    R_j = float(parameter.get("R_j", 0.0) or 0.0)
    jep = R_Eplus
    jip = 1.0 + R_j * (jep - 1.0)
    jplus = np.array([[jep, jip], [jip, jip]], dtype=float)
    permutation_seed = seed if cluster_seed is None else int(seed) + int(cluster_seed)
    init_vector = _initial_state_vector(parameter, init_rates, permutation_seed=permutation_seed)
    warmup_steps = int(binary_cfg.get("warmup_steps", 0) or 0)
    simulation_steps = int(binary_cfg.get("simulation_steps", 0) or 0)
    total_steps = warmup_steps + simulation_steps
    if total_steps <= 0:
        total_steps = 1
    update_ratios = [1, 2]
    n_updates = 1
    total_updates = sum(count * ratio for count, ratio in zip((N_E, N_I), update_ratios))
    tau_scale = total_updates / float(n_updates * update_ratios[0])
    input_length = total_steps / tau_scale if tau_scale > 0 else float(total_steps)
    ps_matrix = np.array([[p0_ee, p0_ei], [p0_ie, p0_ii]], dtype=float)
    thresholds = np.array([V_th, V_th], dtype=float)
    connection_type = parameter.get("connection_type", "bernoulli")
    kappa = float(parameter.get("kappa", 0.0) or 0.0)
    baseline_drive = math.sqrt(max(p0_ee, 0.0) * max(N_E, 1))
    jxs = np.array([baseline_drive, 0.8 * baseline_drive], dtype=float)
    legacy_params: Dict[str, Any] = {
        "randseed": int(seed),
        "simulation_type": "new",
        "connection_type": connection_type,
        "kappa": kappa,
        "spec_func": "EI_jplus",
        "spec_args": {
            "Q": Q_value,
            "jplus": jplus,
        },
        "network_args": {
            "Ns": [N_E, N_I],
            "ps": ps_matrix,
            "Ts": thresholds,
            "g": g_value,
            "update_mode": "random",
            "n_updates": n_updates,
            "delta_T": 0.0,
            "delta_j": None,
            "update_ratios": update_ratios,
        },
        "effective_tau": tau_e,
        "stim_clusters": 0,
        "init": init_vector,
        "input_length": input_length,
        "trials": 1,
        "stim_start": 0.0,
        "stim_end": 0.0,
        "stim_level": 0.0,
        "mx": m_x,
        "jxs": jxs,
        "smooth_rates": None,
        "downsample": None,
    }
    legacy_params["state_record_stride"] = int(total_updates)
    legacy_params["return_mean_cluster_rates"] = True
    legacy_params["return_max_cluster_rates"] = False
    legacy_params["return_cluster_rates_and_spiketimes"] = False
    return legacy_params


def run_legacy_binary_simulation(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    binary_dir: str,
    output_label: str,
    population_rate_inits: Sequence[float],
    *,
    seed: int,
    capture_spikes: bool = False,
    capture_state_dynamics: bool = False,
    cluster_seed: int | None = None,
) -> str:
    legacy_params = _build_legacy_parameters(
        parameter,
        binary_cfg,
        seed,
        population_rate_inits,
        capture_spikes=capture_spikes,
        cluster_seed=cluster_seed,
    )
    if capture_spikes:
        legacy_params["return_cluster_rates_and_spiketimes"] = True
    state_stride = int(legacy_params.get("state_record_stride", 1) or 1)
    if capture_state_dynamics:
        if capture_spikes:
            raise ValueError("Cannot capture spikes and state dynamics simultaneously.")
        legacy_params["return_state_dynamics"] = True
    warmup_steps = int(binary_cfg.get("warmup_steps", 0) or 0)
    simulation_steps = int(binary_cfg.get("simulation_steps", 0) or 0)
    raw_output = legacy_simulate(legacy_params)
    if not isinstance(raw_output, dict):
        raise RuntimeError("Legacy simulation did not return a structured payload.")
    rates_array = np.asarray(raw_output.get("cluster_rates"), dtype=float)
    if rates_array.size == 0:
        raise ValueError("Legacy simulation did not return cluster rate traces.")
    spike_payload = None
    neuron_states = np.zeros((0, 0), dtype=np.uint8)
    state_times = np.zeros((0,), dtype=np.int64)
    subthreshold_fields = np.zeros((0, 0), dtype=np.float32)
    state_updates = None
    state_deltas = None
    initial_state = None
    trimmed_spike_times = None
    trimmed_spike_ids = None
    trimmed_spike_trials = None
    if capture_spikes:
        spike_payload = np.asarray(raw_output.get("spike_times"), dtype=float)
        if spike_payload.size == 0:
            raise RuntimeError("Legacy simulation did not provide spike times.")
        updates_raw = raw_output.get("state_updates")
        deltas_raw = raw_output.get("state_deltas")
        init_raw = raw_output.get("initial_state")
        if updates_raw is not None and deltas_raw is not None:
            state_updates = np.asarray(updates_raw, dtype=np.uint16)
            state_deltas = np.asarray(deltas_raw, dtype=np.int8)
            initial_state = np.asarray(init_raw, dtype=np.uint8) if init_raw is not None else None
        if spike_payload.ndim == 2 and spike_payload.shape[0] >= 3:
            spike_times = np.asarray(spike_payload[0], dtype=float)
            spike_trials = np.asarray(spike_payload[1], dtype=float)
            spike_ids = np.asarray(spike_payload[2], dtype=float)
            mask = np.ones(spike_times.shape, dtype=bool)
            if warmup_steps > 0:
                mask &= spike_times >= warmup_steps
            if simulation_steps > 0:
                mask &= spike_times < (warmup_steps + simulation_steps)
            spike_times = spike_times[mask] - float(warmup_steps)
            spike_ids = spike_ids[mask].astype(np.int64, copy=False)
            spike_trials = spike_trials[mask].astype(np.int16, copy=False)
            trimmed_spike_times = spike_times.astype(np.float32, copy=False)
            trimmed_spike_ids = spike_ids
            trimmed_spike_trials = spike_trials
        else:
            trimmed_spike_times = np.zeros(0, dtype=np.float32)
            trimmed_spike_ids = np.zeros(0, dtype=np.int64)
            trimmed_spike_trials = np.zeros(0, dtype=np.int16)
    if capture_state_dynamics:
        sampled_states = np.asarray(raw_output.get("sampled_states"), dtype=float)
        sampled_fields = np.asarray(raw_output.get("sampled_fields"), dtype=float)
        state_indices = np.asarray(raw_output.get("state_indices"), dtype=np.int64).ravel()
        if sampled_states.size == 0 or sampled_fields.size == 0:
            raise RuntimeError("Legacy simulation did not provide sampled state dynamics.")
        if sampled_states.ndim == 3:
            sampled_states = sampled_states[0]
        if sampled_fields.ndim == 3:
            sampled_fields = sampled_fields[0]
        neuron_states = sampled_states.T.astype(np.uint8, copy=False)
        subthreshold_fields = sampled_fields.T.astype(np.float32, copy=False)
        state_times = state_indices.astype(np.int64) + 1
    if rates_array.ndim == 3:
        # trials x clusters x time
        cluster_rates = rates_array[0]
    elif rates_array.ndim == 2:
        cluster_rates = rates_array
    else:
        raise ValueError("Legacy simulation returned data with unexpected shape.")
    if cluster_rates.ndim != 2:
        raise ValueError("Legacy simulation did not provide cluster rate traces.")
    rates = np.asarray(cluster_rates, dtype=float).T
    if rates.size == 0:
        raise ValueError("Legacy simulation produced an empty rate trace.")
    if warmup_steps > 0:
        if warmup_steps >= rates.shape[0]:
            raise ValueError("Warmup interval exceeds legacy trace length.")
        rates = rates[warmup_steps:]
    if simulation_steps > 0 and rates.shape[0] > simulation_steps:
        rates = rates[:simulation_steps]
    sample_interval = max(1, int(binary_cfg.get("sample_interval", 1) or 1))
    if sample_interval > 1:
        rates = rates[::sample_interval]
    if rates.size == 0:
        raise ValueError("Legacy trace is empty after applying warmup/sample_interval.")
    times = np.arange(rates.shape[0], dtype=np.int64) * sample_interval
    Q_value = int(parameter.get("Q", rates.shape[1] // 2) or 0)
    names: List[str]
    expected_pops = 2 * Q_value if Q_value > 0 else rates.shape[1]
    if expected_pops == rates.shape[1]:
        names = [f"E{idx + 1}" for idx in range(Q_value)] + [f"I{idx + 1}" for idx in range(Q_value)]
    else:
        names = [f"C{idx + 1}" for idx in range(rates.shape[1])]
    if state_updates is not None and state_updates.size:
        total_cols = state_updates.shape[1]
        start_idx = min(max(warmup_steps, 0), total_cols)
        if simulation_steps > 0:
            end_idx = min(start_idx + simulation_steps, total_cols)
        else:
            end_idx = total_cols
        state_updates = state_updates[:, start_idx:end_idx]
        if state_deltas is not None and state_deltas.size:
            state_deltas = state_deltas[:, start_idx:end_idx]
        if sample_interval > 1:
            state_updates = state_updates[:, ::sample_interval]
            if state_deltas is not None and state_deltas.size:
                state_deltas = state_deltas[:, ::sample_interval]
    os.makedirs(binary_dir, exist_ok=True)
    trace_path = os.path.join(binary_dir, f"{output_label}.npz")
    payload = {
        "rates": rates,
        "names": np.asarray(names, dtype=object),
        "times": times.astype(np.float32),
        "sample_interval": np.array(int(sample_interval), dtype=np.int64),
        "warmup_steps": np.array(int(warmup_steps), dtype=np.int64),
        "simulation_steps": np.array(int(simulation_steps), dtype=np.int64),
        "neuron_states": neuron_states,
        "neuron_state_interval": np.array(int(sample_interval), dtype=np.int64),
        "neuron_state_times": state_times.astype(np.int64),
        "subthreshold_fields": subthreshold_fields,
    }
    if state_updates is not None and state_updates.size:
        payload["state_updates"] = state_updates
        if state_deltas is not None and state_deltas.size:
            payload["state_deltas"] = state_deltas
        if initial_state is not None and initial_state.size:
            payload["initial_state"] = initial_state
    if trimmed_spike_times is not None:
        payload["spike_times"] = trimmed_spike_times
        if trimmed_spike_ids is not None:
            payload["spike_ids"] = trimmed_spike_ids
        if trimmed_spike_trials is not None:
            payload["spike_trials"] = trimmed_spike_trials
    np.savez_compressed(trace_path, **payload)
    total_neurons = int(parameter.get("N_E", 0) or 0) + int(parameter.get("N_I", 0) or 0)
    _write_trace_summary(
        trace_path,
        neuron_count=total_neurons,
        warmup_steps=warmup_steps,
        simulation_steps=simulation_steps,
        sample_interval=sample_interval,
        state_stride=sample_interval,
    )
    return trace_path


def _process_seed_task(task: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(task["seed"])
    trace_path = str(task["trace_path"])
    maxima_path = str(task["maxima_path"])
    result: Dict[str, Any] = {
        "seed": seed,
        "trace_path": trace_path,
        "trace_generated": False,
        "states_recorded": False,
        "maxima": None,
        "excitatory_clusters": None,
        "logs": [],
        "error": None,
    }
    try:
        trace_exists = os.path.exists(trace_path)
        if task.get("needs_simulation"):
            run_cfg = dict(task["binary_cfg"])
            run_cfg["seed"] = seed
            focus = task.get("candidate_focus")
            stability = task.get("candidate_stability")
            focus_desc = f"focus {focus}" if focus is not None else "focus ?"
            stability_desc = stability if stability else "unknown stability"
            result["logs"].append(
                f"Simulating legacy binary network for seed {seed} using fixpoint {task.get('candidate_id', 'unknown')} "
                f"({focus_desc}, {stability_desc})."
            )
            trace_path = run_legacy_binary_simulation(
                task["parameter"],
                run_cfg,
                task["binary_dir"],
                task["label"],
                task["init_rates"],
                seed=seed,
                capture_spikes=bool(task.get("capture_spikes")),
            )
            result["trace_path"] = trace_path
            result["trace_generated"] = True
            if task.get("capture_spikes"):
                result["states_recorded"] = True
            trace_exists = True
        if not trace_exists:
            result["logs"].append(f"Skipping seed {seed}: trace {trace_path} is missing.")
            return result
        maxima = None
        excit_count: int | None = None
        if (not task.get("overwrite_analysis")) and os.path.exists(maxima_path):
            cached, cached_excit = _load_maxima_file(maxima_path, int(task["bin_size"]))
            if cached is not None:
                maxima = cached
                excit_count = cached_excit
            else:
                result["logs"].append(
                    f"Recomputing maxima for seed {seed}: bin size mismatch or corrupt cache."
                )
        if maxima is None:
            maxima, excit_count = _compute_maxima_from_trace(trace_path, int(task["bin_size"]))
            _save_maxima_file(maxima_path, maxima, seed, int(task["bin_size"]), trace_path, int(excit_count))
        result["maxima"] = maxima
        result["excitatory_clusters"] = int(excit_count) if excit_count is not None else None
    except Exception as exc:  # pragma: no cover - worker safety
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def _emit_worker_logs(result: Dict[str, Any]) -> None:
    for message in result.get("logs") or []:
        print(message)
    error = result.get("error")
    if error:
        print(f"Seed {result.get('seed')}: {error}")


def _debug_main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug helper: run a short legacy binary simulation using uniform initial rates."
    )
    parser.add_argument(
        "source",
        help="Path to a params.yaml folder or an all_fixpoints_*.pkl bundle.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="legacy_debug_trace",
        help="Name of the generated trace (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the legacy simulator (default: %(default)s).",
    )
    parser.add_argument(
        "--init-rate",
        type=float,
        default=0.1,
        help="Uniform firing-rate initializer applied to all clusters (default: %(default)s).",
    )
    parser.add_argument(
        "--capture-states",
        action="store_true",
        help="Record neuron states for debugging (may increase runtime).",
    )
    args = parser.parse_args()
    parameter, folder_hint, _ = _resolve_simulation_source(args.source)
    binary_cfg = ensure_binary_behavior_defaults(parameter.get("binary"))
    binary_cfg["output_name"] = args.output_name
    rates = build_population_rate_vector(parameter, args.init_rate)
    folder, binary_dir, _ = _prepare_output_folders(parameter, base_folder=folder_hint)
    trace_path = run_legacy_binary_simulation(
        parameter,
        binary_cfg,
        binary_dir,
        args.output_name,
        rates,
        seed=int(args.seed),
        capture_state_dynamics=bool(args.capture_states),
    )
    print(f"Legacy trace stored at {trace_path}")


if __name__ == "__main__":
    _debug_main()
