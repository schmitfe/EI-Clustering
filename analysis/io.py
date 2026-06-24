from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from sim_config import sim_tag_from_cfg

from .preprocessing import bin_spikes_by_cluster, compute_cluster_rates, validate_analysis_input
from .types import AnalysisInput


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_parameter_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    parameter = payload.get("parameter")
    if isinstance(parameter, dict):
        return parameter
    return payload


def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _infer_dt(times: np.ndarray, fallback: Optional[float]) -> float:
    if fallback is not None:
        return float(fallback)
    arr = np.asarray(times, dtype=float).ravel()
    if arr.size >= 2:
        diffs = np.diff(arr)
        positive = diffs[diffs > 0]
        if positive.size:
            return float(np.median(positive))
    return 1.0


def _binary_cluster_names(parameter: Dict[str, Any], fallback_names: Optional[np.ndarray]) -> list[str]:
    if fallback_names is not None:
        return [str(name) for name in fallback_names.tolist()]
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value > 0:
        return [f"E{idx + 1}" for idx in range(q_value)] + [f"I{idx + 1}" for idx in range(q_value)]
    return []


def _spiking_cluster_mapping(net_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    n_exc = int(net_dict.get("N_E", 0) or 0)
    n_inh = int(net_dict.get("N_I", 0) or 0)
    n_clusters = int(net_dict.get("n_clusters", 0) or 0)
    if n_clusters <= 0:
        raise ValueError("spiking.net.n_clusters must be positive.")
    if n_exc % n_clusters != 0 or n_inh % n_clusters != 0:
        raise ValueError("Spiking neuron counts must be divisible by n_clusters.")
    mapping = np.zeros(n_exc + n_inh, dtype=np.int64)
    exc_size = n_exc // n_clusters
    inh_size = n_inh // n_clusters
    for cluster in range(n_clusters):
        mapping[cluster * exc_size : (cluster + 1) * exc_size] = cluster
    for cluster in range(n_clusters):
        start = n_exc + cluster * inh_size
        stop = n_exc + (cluster + 1) * inh_size
        mapping[start:stop] = n_clusters + cluster
    cluster_sizes = np.array([exc_size] * n_clusters + [inh_size] * n_clusters, dtype=float)
    names = [f"E{idx + 1}" for idx in range(n_clusters)] + [f"I{idx + 1}" for idx in range(n_clusters)]
    return mapping, cluster_sizes, names


def analysis_input_from_binary_trace(
    trace_path: str | Path,
    *,
    parameter: Optional[Dict[str, Any]] = None,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> AnalysisInput:
    path = Path(trace_path)
    cfg = dict(analysis_cfg or {})
    with np.load(path, allow_pickle=True) as payload:
        rates = np.asarray(payload["rates"], dtype=float)
        times = np.asarray(payload["times"], dtype=float) if "times" in payload else np.arange(rates.shape[0], dtype=float)
        names = payload["names"] if "names" in payload else None
        population_names = payload["population_names"] if "population_names" in payload else names
        population_cell_types = payload["population_cell_types"] if "population_cell_types" in payload else None
        population_cluster_indices = payload["population_cluster_indices"] if "population_cluster_indices" in payload else None
    dt_override = cfg.get("dt", cfg.get("bin_size"))
    dt = _infer_dt(times, None if dt_override is None else float(dt_override))
    X_binary = rates.astype(np.uint8) if np.all(np.isin(rates, [0.0, 1.0])) else None
    result = AnalysisInput(
        X_rate=rates,
        X_binary=X_binary,
        dt=dt,
        source_type="binary",
        cluster_ids=list(range(rates.shape[1])),
        cluster_names=_binary_cluster_names(parameter or {}, population_names),
        cluster_cell_types=None
        if population_cell_types is None
        else [str(value) for value in np.asarray(population_cell_types, dtype=object).tolist()],
        cluster_group_ids=None
        if population_cluster_indices is None
        else [int(value) for value in np.asarray(population_cluster_indices, dtype=int).tolist()],
        metadata={
            "trace_path": str(path.resolve()),
            "times_start": float(times[0]) if times.size else 0.0,
            "times_stop": float(times[-1]) if times.size else 0.0,
        },
        config=cfg,
    )
    return validate_analysis_input(result)


def analysis_input_from_spiking_payload(
    payload: Dict[str, Any],
    *,
    parameter: Optional[Dict[str, Any]] = None,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> AnalysisInput:
    cfg = dict(analysis_cfg or {})
    sim_dict = dict(_unwrap_object(payload.get("sim_dict")) or _unwrap_object(payload.get("params")) or {})
    net_dict = dict(_unwrap_object(payload.get("net_dict")) or {})
    if parameter is not None:
        spiking_cfg = parameter.get("spiking") or {}
        if not sim_dict:
            sim_dict = dict(spiking_cfg.get("sim") or {})
        if not net_dict:
            net_dict = dict(spiking_cfg.get("net") or {})
        net_dict.setdefault("N_E", parameter.get("N_E"))
        net_dict.setdefault("N_I", parameter.get("N_I"))
        net_dict.setdefault("n_clusters", parameter.get("Q"))
    if "spiketimes" in payload:
        spikes = np.asarray(payload["spiketimes"], dtype=float)
        spike_times = spikes[0]
        spike_ids = spikes[1].astype(np.int64)
    else:
        spike_times = np.asarray(payload.get("spike_times", np.zeros(0)), dtype=float).ravel()
        spike_ids = np.asarray(payload.get("spike_ids", np.zeros(0)), dtype=np.int64).ravel()
    dt = float(cfg.get("dt") or cfg.get("bin_size") or sim_dict.get("dt") or 1.0)
    t_stop = float(sim_dict.get("simtime") or (spike_times.max() if spike_times.size else dt))
    mapping, cluster_sizes, cluster_names = _spiking_cluster_mapping(net_dict)
    X_counts = bin_spikes_by_cluster(
        spike_times,
        spike_ids,
        mapping,
        dt=dt,
        t_start=0.0,
        t_stop=t_stop,
        n_clusters=len(cluster_names),
    )
    X_rate = compute_cluster_rates(X_counts, dt=dt / 1000.0 if dt > 1e-9 else 1.0, cluster_sizes=cluster_sizes)
    X_binary = X_counts.astype(np.uint8) if np.all(np.isin(X_counts, [0, 1])) else None
    result = AnalysisInput(
        X_counts=X_counts,
        X_rate=X_rate,
        X_binary=X_binary,
        dt=dt,
        source_type="snn",
        cluster_ids=list(range(len(cluster_names))),
        cluster_names=cluster_names,
        cluster_cell_types=["E"] * (len(cluster_names) // 2) + ["I"] * (len(cluster_names) // 2),
        cluster_group_ids=list(range(len(cluster_names) // 2)) + list(range(len(cluster_names) // 2)),
        metadata={
            "simtime": t_stop,
            "spike_count": int(spike_times.size),
            "n_clusters": len(cluster_names),
        },
        config=cfg,
    )
    return validate_analysis_input(result)


def _detect_npz_source(path: Path) -> str:
    with np.load(path, allow_pickle=True) as payload:
        if "spiketimes" in payload or ("spike_times" in payload and "spike_ids" in payload):
            return "snn"
        if "rates" in payload:
            return "binary"
    raise ValueError(f"Could not determine source type from {path}.")


def _discover_npz(folder: Path) -> Path:
    candidates = sorted(
        path
        for path in folder.glob("*.npz")
        if "analysis" not in path.name and "summary" not in path.name and "pooled" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"No simulation .npz files found in {folder}.")
    return candidates[0]


def load_analysis_input(
    source: str | Path,
    *,
    source_type: str = "auto",
    parameter: Optional[Dict[str, Any]] = None,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> AnalysisInput:
    path = Path(source)
    cfg = dict(analysis_cfg or {})
    if path.is_dir():
        params_path = path / "params.yaml"
        resolved_parameter = parameter if parameter is not None else (_normalize_parameter_payload(_load_yaml(params_path)) if params_path.exists() else {})
        npz_path = _discover_npz(path)
    else:
        resolved_parameter = dict(parameter or {})
        params_path = path.parent / "params.yaml"
        if not resolved_parameter and params_path.exists():
            resolved_parameter = _normalize_parameter_payload(_load_yaml(params_path))
        npz_path = path
    resolved_source = _detect_npz_source(npz_path) if source_type == "auto" else str(source_type)
    if resolved_source == "binary":
        return analysis_input_from_binary_trace(npz_path, parameter=resolved_parameter, analysis_cfg=cfg)
    if resolved_source == "snn":
        with np.load(npz_path, allow_pickle=True) as payload:
            content = {key: payload[key] for key in payload.files}
        return analysis_input_from_spiking_payload(content, parameter=resolved_parameter, analysis_cfg=cfg)
    raise ValueError(f"Unsupported source_type '{resolved_source}'.")


def resolve_analysis_output_dir(base_dir: str | Path, analysis_cfg: Dict[str, Any]) -> Path:
    tag = sim_tag_from_cfg({"analysis": analysis_cfg})
    output_dir = Path(base_dir) / "analysis" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
