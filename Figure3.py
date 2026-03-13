#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np  # noqa: E402

from figure_cli import add_v_sweep_arguments, parse_int_values, resolve_float_values, resolve_v_sweep
import pipelines.mean_field as ei_pipeline  # noqa: E402
from MeanField.rate_system import ensure_output_folder  # noqa: E402
from pipelines.binary import ensure_binary_behavior_defaults, finalize_binary_config, run_binary_simulation  # noqa: E402
from plotting import (
    BinaryStateSource,
    FontCfg,
    RasterLabels,
    add_panel_label,
    draw_listed_colorbar,
    plot_binary_raster,
    plot_spike_raster,
    style_axes,
    _prepare_line_color_map,
)  # noqa: E402
from sim_config import add_override_arguments, deep_update, load_from_args, parse_overrides, sim_tag_from_cfg  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 3 columns by running the EI pipeline, performing BinaryNetwork simulations, "
            "and visualizing raster/rate traces with MF cluster predictions."
        )
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--r-eplus",
        type=str,
        nargs="+",
        default=None,
        help="Explicit R_Eplus values or range expressions (e.g., 6 8 or 1:20:0.5).",
    )
    add_v_sweep_arguments(parser)
    parser.add_argument("--retry-step", type=float, help="Optional retry increment for solver restarts.")
    parser.add_argument(
        "--delta-rep-mf",
        type=float,
        default=0.025,
        help="R_Eplus sampling half-width for MF fallback (default: %(default)s).",
    )
    parser.add_argument(
        "--rep-retry-mf",
        type=int,
        default=10,
        help="Resample attempts for MF fallback when no fixpoints are found (default: %(default)s).",
    )
    parser.add_argument(
        "--rep-rng-seed",
        type=int,
        help="Seed for MF resample RNG (default: random).",
    )
    parser.add_argument(
        "--erf-jobs",
        type=int,
        default=1,
        help="Number of workers for the ERF stage (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for the BinaryNetwork simulations (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite-erf",
        action="store_true",
        help="Re-run the ERF stage even if matching files exist.",
    )
    parser.add_argument(
        "--focus-counts",
        type=str,
        nargs="+",
        help="Focus counts or integer ranges to include (e.g., 1 2 4 or 1:4:1). Defaults to all values from 1..Q.",
    )
    parser.add_argument(
        "--stability-filter",
        choices=("stable", "unstable", "any"),
        default="stable",
        help="Select fixpoints of the given stability (default: %(default)s).",
    )
    parser.add_argument(
        "--column-override",
        action="append",
        default=[],
        metavar="label:path=value",
        help="Column-specific override using the column label (e.g., a:kappa=0).",
    )
    parser.add_argument(
        "--column-title",
        action="append",
        default=[],
        metavar="label:title",
        help="Override the column title text (e.g., a:R_j=1.1).",
    )
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size.")
    parser.add_argument("--seed", type=int, help="Base seed for the BinaryNetwork simulations.")
    parser.add_argument("--output-name", type=str, help="Custom prefix for saved binary traces.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new BinaryNetwork simulations and reuse existing traces.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if matching traces exist.",
    )
    parser.add_argument(
        "--raster-duration",
        type=float,
        help="Restrict the raster plot to this duration (same units as the trace time axis).",
    )
    parser.add_argument(
        "--raster-stride",
        type=int,
        default=1,
        help="Plot every Nth neuron in the raster (default: %(default)s).",
    )
    parser.add_argument(
        "--rates-duration",
        type=float,
        help="Restrict the rate plot to this duration (same units as the trace time axis).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="Figures/Figure3",
        help="Prefix for the saved figure files (default: %(default)s.{png,pdf}).",
    )
    return parser.parse_args()


def _resolve_r_eplus_list(args: argparse.Namespace, parameter: Dict[str, Any]) -> List[float]:
    values = resolve_float_values(args.r_eplus, option_name="--r-eplus")
    if values:
        return list(values)
    base = parameter.get("R_Eplus")
    if base is None:
        raise ValueError("Provide at least one R_Eplus value via the config or --r-eplus.")
    return [float(base)]


def _load_trace_payload(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace {path} does not exist.")
    with np.load(path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
        times = np.asarray(data.get("times"), dtype=float) if "times" in data else np.arange(rates.shape[0])
        states = np.asarray(data.get("neuron_states"), dtype=np.uint8) if "neuron_states" in data else np.zeros((0, 0), dtype=np.uint8)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_updates = np.asarray(data["state_updates"], dtype=np.uint16) if "state_updates" in data else None
        state_deltas = np.asarray(data["state_deltas"], dtype=np.int8) if "state_deltas" in data else None
        initial_state = np.asarray(data["initial_state"], dtype=np.uint8) if "initial_state" in data else None
        spike_times = np.asarray(data["spike_times"], dtype=float).ravel() if "spike_times" in data else np.zeros(0, dtype=float)
        spike_ids = np.asarray(data["spike_ids"], dtype=np.int64).ravel() if "spike_ids" in data else np.zeros(0, dtype=np.int64)
        spike_trials = np.asarray(data["spike_trials"], dtype=np.int16).ravel() if "spike_trials" in data else np.zeros(0, dtype=np.int16)
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "state_updates": state_updates,
        "state_deltas": state_deltas,
        "initial_state": initial_state,
        "sample_interval": sample_interval,
        "spike_times": spike_times,
        "spike_ids": spike_ids,
        "spike_trials": spike_trials,
    }


@dataclass(frozen=True)
class FocusMarker:
    focus: int
    value: float
    stable: bool


def _collect_focus_markers(focus_rates: Dict[int, Dict[str, List[float]]]) -> List[FocusMarker]:
    markers: List[FocusMarker] = []
    for focus in sorted(int(key) for key in focus_rates.keys()):
        payload = focus_rates.get(int(focus), {})
        stable_values = [float(val) for val in (payload.get("stable") or []) if np.isfinite(val)]
        unstable_values = [float(val) for val in (payload.get("unstable") or []) if np.isfinite(val)]
        if stable_values:
            markers.append(FocusMarker(focus=int(focus), value=max(stable_values), stable=True))
        if unstable_values:
            markers.append(FocusMarker(focus=int(focus), value=max(unstable_values), stable=False))
    return markers


@dataclass(frozen=True)
class ColumnSpec:
    label: str
    overrides: Sequence[str] = ()
    title: str | None = None


@dataclass(frozen=True)
class ColumnContext:
    spec: ColumnSpec
    parameter: Dict[str, Any]
    title: str
    focus_counts: Sequence[int]
    focus_markers: List[FocusMarker]
    folder: str
    bundle_path: str
    binary_cfg: Dict[str, Any]
    seed: int
    trace_path: str


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


COLUMN_SPECS: Sequence[ColumnSpec] = (
    ColumnSpec(label="a", overrides=("kappa=0",)),
    ColumnSpec(label="b", overrides=("kappa=0.5",)),
    ColumnSpec(label="c", overrides=("kappa=1",)),
)


def _normalize_label(label: str) -> str:
    if not label or not label.strip():
        raise ValueError("Column label must not be empty.")
    return label.strip().lower()


def _parse_column_override_entries(entries: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Column override '{raw}' is missing ':' between label and override.")
        label, override = raw.split(":", 1)
        override = override.strip()
        if "=" not in override:
            raise ValueError(f"Column override '{raw}' must include '=' in the override expression.")
        key = _normalize_label(label)
        mapping.setdefault(key, []).append(override)
    return mapping


def _parse_column_title_entries(entries: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Column title '{raw}' is missing ':' between label and title.")
        label, title = raw.split(":", 1)
        key = _normalize_label(label)
        mapping[key] = title.strip()
    return mapping


def _validate_column_keys(mapping: Dict[str, object], known: set[str], context: str) -> None:
    if not mapping:
        return
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"Unknown column label(s) for {context}: {', '.join(unknown)}.")


def _format_time_ticks(start: float, end: float, count: int = 4) -> Tuple[np.ndarray, List[str]]:
    if count <= 1 or end <= start:
        values = np.array([start, end], dtype=float)
    else:
        values = np.linspace(start, end, count, dtype=float)
    labels: List[str] = []
    for value in values:
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            labels.append(f"{rounded:d}")
        else:
            labels.append(f"{value:.0f}")
    return values, labels


def _time_axis_scale_from_taus(parameter: Dict[str, Any]) -> Tuple[float, str]:
    tau_e = float(parameter.get("tau_e", math.nan))
    tau_i = float(parameter.get("tau_i", math.nan))
    n_e = int(parameter.get("N_E", 0) or 0)
    n_i = int(parameter.get("N_I", 0) or 0)
    if not np.isfinite(tau_e) or not np.isfinite(tau_i) or tau_e <= 0.0 or tau_i <= 0.0:
        return 1.0, "Time [s]"
    max_tau = max(tau_e, tau_i)
    min_tau = min(tau_e, tau_i)
    n_max = n_e if tau_e >= tau_i else n_i
    n_min = n_i if tau_e >= tau_i else n_e
    expected_updates = float(n_max) + (max_tau / min_tau) * float(n_min)
    if not np.isfinite(expected_updates) or expected_updates <= 0.0 or max_tau <= 0.0:
        return 1.0, "Time [s]"
    max_tau_seconds = max_tau / 1000.0
    if max_tau_seconds <= 0.0 or not np.isfinite(max_tau_seconds):
        return 1.0, "Time [s]"
    return float(expected_updates / max_tau_seconds), "Time [s]"


def _format_kappa_value(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _resolve_focus_counts(parameter: Dict[str, Any], explicit: Sequence[int] | None = None) -> List[int]:
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    if explicit:
        values = sorted({max(1, int(val)) for val in explicit})
        if values:
            return values
    return list(range(1, q_value + 1))


def _resolve_binary_config(parameter: Dict[str, Any], overrides: BinaryRunSettings) -> Dict[str, Any]:
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
    return finalize_binary_config(parameter, ensure_binary_behavior_defaults(cfg))


def _filtered_parameter_for_tag(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    filtered.pop("R_Eplus", None)
    filtered.pop("focus_count", None)
    filtered.pop("focus_counts", None)
    return filtered


def _compute_fixpoint_bundle_path(parameter: Dict[str, Any]) -> str:
    filtered = _filtered_parameter_for_tag(parameter)
    tag = sim_tag_from_cfg(filtered)
    kappa = float(parameter.get("kappa", 0.0) or 0.0)
    conn = str(parameter.get("connection_type", "bernoulli")).lower().replace(" ", "_")
    encoded_kappa = f"{kappa:.2f}".replace(".", "_")
    r_j = parameter.get("R_j", 0.0)
    return os.path.join("data", f"all_fixpoints_{conn}_kappa{encoded_kappa}_Rj{r_j}_{tag}.pkl")


def _ensure_fixpoint_bundle(
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
        folder = ensure_output_folder(param, tag=sim_tag_from_cfg(_filtered_parameter_for_tag(param)))
    ei_pipeline.run_analysis(folder, param, focus_counts, plot_erfs=sweep_cfg.plot_erfs)
    bundle_path = _compute_fixpoint_bundle_path(param)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Fixpoint bundle {bundle_path} was not generated. Ensure ei_pipeline completed successfully."
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
        raise ValueError(f"No fixpoints matched {focus_msg} with stability filter '{stability_filter}'.")
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


def _candidate_for_seed(seed_value: int, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Fixpoint candidate list is empty.")
    rng_seed = int(seed_value) % (2**32)
    rng = np.random.default_rng(rng_seed)
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


def _format_seed_label(base_name: str, seed: int) -> str:
    trimmed = base_name.strip() or "activity_trace"
    return f"{trimmed}_seed{seed:06d}"


def _expected_binary_trace_path(parameter: Dict[str, Any], binary_cfg: Dict[str, Any], output_name: str) -> str:
    folder = ensure_output_folder(parameter, tag=sim_tag_from_cfg(_filtered_parameter_for_tag(parameter)))
    binary_tag = sim_tag_from_cfg({"parameter": parameter, "binary": dict(binary_cfg)})
    binary_dir = os.path.join(folder, "binary", binary_tag)
    return os.path.join(binary_dir, f"{output_name}.npz")


def _apply_overrides(parameter: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    if not overrides:
        return dict(parameter)
    return deep_update(parameter, parse_overrides(overrides))


def _build_column_parameter(
    base_parameter: Dict[str, Any],
    spec: ColumnSpec,
    *,
    column_override_map: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    overrides: List[str] = list(spec.overrides)
    overrides.extend(column_override_map.get(_normalize_label(spec.label), []))
    return _apply_overrides(base_parameter, overrides)


def _resolve_column_title(
    parameter: Dict[str, Any],
    spec: ColumnSpec,
    *,
    title_map: Dict[str, str],
) -> str:
    label_key = _normalize_label(spec.label)
    custom = title_map.get(label_key)
    if custom is not None and custom.strip():
        return custom
    if spec.title:
        return spec.title
    kappa_value = float(parameter.get("kappa", 0.0) or 0.0)
    rj_value = float(parameter.get("R_j", 0.0) or 0.0)

    return rf"$\kappa={_format_kappa_value(kappa_value)}\quad R_j={_format_kappa_value(rj_value)}$"


def _prepare_focus_markers(
    parameter: Dict[str, Any],
    bundle_path: str,
    focus_counts: Sequence[int],
    stability_filter: str,
) -> Tuple[List[FocusMarker], Sequence[Dict[str, Any]]]:
    bundle = _load_fixpoint_bundle(bundle_path)
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    r_eplus = parameter.get("R_Eplus")
    if r_eplus is None:
        raise ValueError("Parameter 'R_Eplus' must be set before selecting MF fixpoints.")
    raw_candidates = _load_fixpoint_candidates(
        bundle,
        focus_counts,
        stability_filter,
        2 * q_value,
        float(r_eplus),
    )
    candidates = _deduplicate_candidates_by_focus(raw_candidates, q_value)
    if not candidates:
        raise ValueError("No fixpoint candidates matched the requested filters.")
    focus_rates = _focus_payload_from_candidates(candidates, q_value)
    return _collect_focus_markers(focus_rates), candidates


def _is_missing_fixpoints_error(exc: Exception) -> bool:
    return "no fixpoint" in str(exc).lower()


def _prepare_focus_markers_with_retry(
    parameter: Dict[str, Any],
    focus_counts: Sequence[int],
    stability_filter: str,
    sweep_cfg: PipelineSweepSettings,
    *,
    delta_rep: float,
    rep_retry: int,
    rng_seed: int | None,
    column_label: str,
) -> Tuple[List[FocusMarker], Sequence[Dict[str, Any]], str, str]:
    target_rep = parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError("Parameter 'R_Eplus' must be set before selecting MF fixpoints.")
    target_rep = float(target_rep)
    folder, bundle_path = _ensure_fixpoint_bundle(
        parameter,
        focus_counts,
        [target_rep],
        sweep_cfg,
    )
    base_exc: ValueError | None = None
    try:
        focus_markers, candidates = _prepare_focus_markers(
            parameter,
            bundle_path,
            focus_counts,
            stability_filter,
        )
        return focus_markers, candidates, folder, bundle_path
    except ValueError as exc:
        if not _is_missing_fixpoints_error(exc):
            raise
        base_exc = exc
    if rep_retry <= 0 or delta_rep <= 0:
        if base_exc is not None:
            raise base_exc
        raise
    print(
        f"No MF fixpoints for R_Eplus={target_rep:.4f} (column {column_label}); "
        f"trying {rep_retry} resamples within +/-{delta_rep:.4f}."
    )
    rng = np.random.default_rng(rng_seed)
    for attempt in range(int(rep_retry)):
        sample_rep = float(rng.uniform(target_rep - delta_rep, target_rep + delta_rep))
        sample_param = dict(parameter)
        sample_param["R_Eplus"] = sample_rep
        folder, bundle_path = _ensure_fixpoint_bundle(
            sample_param,
            focus_counts,
            [sample_rep],
            sweep_cfg,
        )
        try:
            focus_markers, candidates = _prepare_focus_markers(
                sample_param,
                bundle_path,
                focus_counts,
                stability_filter,
            )
        except ValueError as exc:
            if not _is_missing_fixpoints_error(exc):
                raise
            continue
        print(
            f"MF resample {attempt + 1}/{rep_retry} found fixpoints at "
            f"R_Eplus={sample_rep:.4f} (column {column_label})."
        )
        return focus_markers, candidates, folder, bundle_path
    raise ValueError(
        f"No MF fixpoints found after {rep_retry} resamples within +/-{delta_rep:.4f} "
        f"of R_Eplus={target_rep:.4f}."
    )


def _build_trace_task(
    parameter: Dict[str, Any],
    binary_cfg: Dict[str, Any],
    *,
    candidates: Sequence[Dict[str, Any]],
    seed: int,
    analysis_only: bool,
    overwrite_simulation: bool,
) -> Tuple[str, Dict[str, Any] | None]:
    base_output = str(binary_cfg.get("output_name", "activity_trace"))
    label = _format_seed_label(base_output, seed)
    task_binary_cfg = dict(binary_cfg)
    task_binary_cfg["seed"] = int(seed)
    trace_path = _expected_binary_trace_path(parameter, task_binary_cfg, label)
    needs_simulation = (not analysis_only) and (overwrite_simulation or not os.path.exists(trace_path))
    if not needs_simulation:
        return trace_path, None
    candidate = _candidate_for_seed(seed, candidates)
    init_rates = tuple(float(value) for value in candidate["rates"])
    task = {
        "parameter": parameter,
        "binary_cfg": task_binary_cfg,
        "label": label,
        "init_rates": init_rates,
        "seed": int(seed),
        "trace_path": trace_path,
        "overwrite_simulation": bool(overwrite_simulation),
    }
    return trace_path, task


def _simulate_binary_task(task: Dict[str, Any]) -> str:
    trace_path = str(task["trace_path"])
    if (not task.get("overwrite_simulation")) and os.path.exists(trace_path):
        return trace_path
    seed = int(task["seed"])
    label = str(task["label"])
    print(f"Simulating BinaryNetwork for seed {seed} (column {label}).")
    result = run_binary_simulation(
        task["parameter"],
        task["binary_cfg"],
        output_name=label,
        population_rate_inits=task["init_rates"],
    )
    return str(result["trace_path"])


def _marker_data_step(
    ax: plt.Axes,
    span: float,
    orientation: str,
    *,
    marker_size: float,
    fraction: float = 2.0 / 3.0,
) -> tuple[float, float]:
    span = max(float(span), 1e-9)
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
    except Exception:
        bbox = ax.get_window_extent()
    extent = bbox.width if orientation == "x" else bbox.height
    extent = max(extent, 1e-9)
    data_per_pixel = span / extent
    pixels_per_point = fig.dpi / 72.0 if fig.dpi else 1.0
    data_per_point = data_per_pixel * pixels_per_point
    diameter_points = 2.0 * math.sqrt(marker_size / math.pi)
    diameter_data = diameter_points * data_per_point
    step_data = diameter_data * fraction
    return step_data, diameter_data


def _compute_linear_offsets(
    values: Sequence[float],
    focus_keys: Sequence[int],
    *,
    step: float,
    threshold: float = 0.025,
) -> Tuple[List[float], List[List[int]]]:
    count = len(values)
    offsets = [0.0] * count
    clusters: List[List[int]] = []
    if count == 0:
        return offsets, clusters
    if count == 1 or step <= 0.0:
        clusters = [[idx] for idx in range(count)]
        return offsets, clusters
    order = sorted(range(count), key=lambda idx: values[idx])
    i = 0
    while i < count:
        base_idx = order[i]
        cluster = [base_idx]
        j = i + 1
        while j < count and abs(values[order[j]] - values[base_idx]) <= threshold:
            cluster.append(order[j])
            j += 1
        cluster_sorted = sorted(cluster, key=lambda idx: focus_keys[idx], reverse=True)
        clusters.append(cluster_sorted)
        for pos, idx in enumerate(cluster_sorted):
            offsets[idx] = -step * pos
        i = j
    return offsets, clusters


def _plot_example_traces(
    ax_raster: plt.Axes,
    ax_rates: plt.Axes,
    payload: Dict[str, Any] | None,
    *,
    parameter: Dict[str, Any],
    raster_duration: float | None,
    rates_duration: float | None,
    raster_stride: int,
    font_cfg: FontCfg,
    focus_markers: Sequence[FocusMarker],
    color_map: Dict[int, str],
) -> None:
    if payload is None:
        for ax, label in ((ax_raster, "raster"), (ax_rates, "rate traces")):
            ax.text(0.5, 0.5, f"No {label} available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    excit_neurons = int(parameter.get("N_E", 0) or 0)
    states_raw = payload.get("states")
    if states_raw is None:
        states_arr = np.zeros((0, 0), dtype=np.uint8)
    else:
        states_arr = np.asarray(states_raw)
    sample_interval = max(1, int(payload.get("sample_interval", 1) or 1))
    raster_window = (0.0, float(raster_duration)) if raster_duration is not None and raster_duration > 0 else None
    stride = max(1, int(raster_stride))
    labels = RasterLabels(
        show=True,
        excitatory="Exc.",
        inhibitory="Inh.",
        location="left",
        kwargs={
            "fontsize": font_cfg.tick,
            "rotation": 90,
            "ha": "right",
            "va": "center",
        },
    )
    total_neurons = excit_neurons + int(parameter.get("N_I", 0) or 0)
    updates_raw = payload.get("state_updates")
    deltas_raw = payload.get("state_deltas")
    init_state = payload.get("initial_state")
    if init_state is not None and not np.asarray(init_state).size:
        init_state = None
    if updates_raw is not None and deltas_raw is not None and np.asarray(updates_raw).size:
        state_source = BinaryStateSource.from_diff_logs(
            updates_raw,
            deltas_raw,
            neuron_count=total_neurons,
            initial_state=init_state,
        )
    else:
        state_source = BinaryStateSource.from_array(states_arr)
    existing = {id(text) for text in ax_raster.texts}
    times_raw = payload.get("times")
    time_axis = np.asarray(times_raw, dtype=float) if times_raw is not None else np.empty(0, dtype=float)
    rates_raw = payload.get("rates")
    rates = np.asarray(rates_raw) if rates_raw is not None else np.zeros((0, 0), dtype=float)
    names = payload.get("names") or []
    state_steps = int(states_arr.shape[0])
    rate_steps = int(rates.shape[0])
    total_steps = max(state_steps, rate_steps)
    finite_times = time_axis[np.isfinite(time_axis)]
    if finite_times.size:
        window_start = float(finite_times.min())
        window_end = float(finite_times.max())
    else:
        window_start = 0.0
        window_end = float(total_steps * sample_interval)
    if window_end <= window_start:
        window_end = window_start + float(sample_interval)
    scale_start = window_start
    scale_end = window_end
    if raster_duration is not None and raster_duration > 0:
        scale_start = 0.0
        scale_end = float(raster_duration)
    if rates_duration is not None and rates_duration > 0:
        scale_start = 0.0
        scale_end = float(rates_duration)
    if scale_end <= scale_start:
        scale_end = scale_start + float(sample_interval)
    time_scale, time_label = _time_axis_scale_from_taus(parameter)
    safe_scale = time_scale if time_scale > 0 else 1.0
    plot_window = raster_window if raster_window is not None else (window_start, window_end)
    spike_times = np.asarray(payload.get("spike_times"), dtype=float)
    spike_ids = np.asarray(payload.get("spike_ids"), dtype=np.int64)
    n_inh = max(total_neurons - excit_neurons, 0)
    if spike_times.size and spike_ids.size:
        scaled_spike_times = spike_times / safe_scale
        if plot_window is not None:
            t_start = plot_window[0] / safe_scale
            t_end = plot_window[1] / safe_scale
        else:
            t_start = None
            t_end = None
        plot_spike_raster(
            ax=ax_raster,
            spike_times_ms=scaled_spike_times,
            spike_ids=spike_ids,
            n_exc=excit_neurons,
            n_inh=n_inh,
            stride=stride,
            t_start=t_start,
            t_end=t_end,
            marker=".",
            marker_size=1.5,
            labels=labels,
        )
    else:
        plot_binary_raster(
            ax=ax_raster,
            state_source=state_source,
            sample_interval=sample_interval,
            n_exc=excit_neurons,
            total_neurons=total_neurons,
            window=plot_window,
            time_scale=time_scale,
            stride=stride,
            labels=labels,
            marker=".",
            marker_size=2.0,
            empty_text="No neuron state samples",
        )
    for text in ax_raster.texts:
        if id(text) not in existing:
            label = text.get_text().strip().lower()
            if label.startswith("inh"):
                text.set_color("#8B0000")
            elif label.startswith("exc"):
                text.set_color("black")
    ax_raster.set_title("")
    ax_raster.set_xlabel("")
    ax_raster.set_ylabel("")
    ax_raster.tick_params(axis="x", labelbottom=False)
    ax_raster.tick_params(axis="y", left=False, labelleft=False)
    _plot_grayscale_rates(
        ax_rates,
        time_axis,
        rates,
        names,
        time_scale=time_scale,
        sample_interval=sample_interval,
    )
    scaled_start = scale_start / safe_scale
    scaled_end = scale_end / safe_scale
    if not np.isfinite(scaled_start):
        scaled_start = 0.0
    if not np.isfinite(scaled_end):
        scaled_end = scaled_start + 1.0
    if scaled_end <= scaled_start:
        scaled_end = scaled_start + 1.0
    ax_raster.set_xlim(scaled_start, scaled_end)
    ax_rates.set_xlim(scaled_start, scaled_end)
    ax_rates.set_ylabel(r"$m_c$")
    ax_rates.set_xlabel(time_label)
    ax_rates.xaxis.set_major_locator(MaxNLocator(integer=True))
    _plot_fixpoint_overlays(
        ax_rates,
        focus_markers,
        color_map=color_map,
        x_start=scaled_start,
        x_end=scaled_end,
    )


def _plot_fixpoint_overlays(
    ax: plt.Axes,
    markers: Sequence[FocusMarker],
    *,
    color_map: Dict[int, str],
    x_start: float,
    x_end: float,
) -> None:
    if not markers:
        return
    if not np.isfinite(x_start) or not np.isfinite(x_end):
        return
    if x_end <= x_start:
        x_end = x_start + 1.0
    span = x_end - x_start
    step_x, diameter_x = _marker_data_step(ax, span, "x", marker_size=25.0)
    marker_x = x_end - max(diameter_x / 2.0, 0.0)
    fallback_color = "#444444"
    valid_entries: List[Tuple[int, float, int]] = []
    for idx, marker in enumerate(markers):
        value = float(marker.value)
        if not np.isfinite(value):
            continue
        valid_entries.append((idx, value, marker.focus))
    if not valid_entries:
        return
    values = [entry[1] for entry in valid_entries]
    focuses = [entry[2] for entry in valid_entries]
    offsets, clusters = _compute_linear_offsets(values, focuses, step=step_x, threshold=0.025)
    aligned_levels = list(values)
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        leftmost_idx = min(cluster, key=lambda idx: offsets[idx])
        ref_value = values[leftmost_idx]
        for member in cluster:
            aligned_levels[member] = ref_value
    for (entry, offset) in zip(valid_entries, offsets):
        idx, value, _ = entry
        marker = markers[idx]
        color = color_map.get(marker.focus, fallback_color)
        linestyle = "-" if marker.stable else "--"
        value = aligned_levels[idx]
        ax.hlines(
            value,
            x_start,
            x_end,
            colors=color,
            linewidth=0.8,
            linestyles=linestyle,
            alpha=0.5,
            zorder=4.0,
        )
        facecolor = color if marker.stable else "white"
        ax.scatter(
            marker_x + offset,
            value,
            marker="o",
            s=25.0,
            facecolors=facecolor,
            edgecolors=color,
            linewidths=0.8,
            zorder=5.0,
        )


def _plot_grayscale_rates(
    ax: plt.Axes,
    times: np.ndarray,
    rates: np.ndarray,
    names: Sequence[str],
    *,
    time_scale: float,
    sample_interval: int,
) -> None:
    if rates.ndim != 2 or rates.shape[0] == 0:
        ax.text(0.5, 0.5, "No excitatory spikes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    excit_indices = [idx for idx, name in enumerate(names) if str(name).startswith("E")]
    if not excit_indices:
        ax.text(0.5, 0.5, "No excitatory populations", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    excit_rates = rates[:, excit_indices]
    if times.size:
        time_axis = np.asarray(times, dtype=float)
        valid = np.isfinite(time_axis)
        if time_axis.size != excit_rates.shape[0] or not np.all(valid):
            time_axis = np.arange(excit_rates.shape[0], dtype=float) * float(sample_interval)
        else:
            time_axis = time_axis.astype(float, copy=False)
    else:
        time_axis = np.arange(excit_rates.shape[0], dtype=float) * float(sample_interval)
    safe_scale = time_scale if time_scale > 0 else 1.0
    scaled_time_axis = time_axis / safe_scale
    cmap = plt.get_cmap("Greys")
    if excit_rates.shape[1] <= 1:
        shades = [0.55]
    else:
        shades = np.linspace(0.25, 0.85, excit_rates.shape[1])
    for idx, shade in enumerate(shades):
        ax.plot(scaled_time_axis, excit_rates[:, idx], color=cmap(shade), linewidth=1.2)
    finite_scaled = scaled_time_axis[np.isfinite(scaled_time_axis)]
    if finite_scaled.size:
        ax.set_xlim(float(finite_scaled.min()), float(finite_scaled.max()))
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 1.0])


def _save_figure(fig: plt.Figure, output_prefix: str, r_value: float) -> None:
    encoded_r = f"{r_value:.2f}".replace(".", "_")
    base = Path(output_prefix)#.with_name(f"{Path(output_prefix).name}_REplus{encoded_r}")
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path, dpi=600)
    print(f"Stored Figure 3 panel at {png_path} and {pdf_path}")


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    r_eplus_values = _resolve_r_eplus_list(args, base_parameter)
    parsed_focus_counts = parse_int_values(args.focus_counts, option_name="--focus-counts")
    column_override_map = _parse_column_override_entries(args.column_override)
    column_title_map = _parse_column_title_entries(args.column_title)
    known_labels = {_normalize_label(spec.label) for spec in COLUMN_SPECS}
    _validate_column_keys(column_override_map, known_labels, "column overrides")
    _validate_column_keys(column_title_map, known_labels, "column titles")
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    v_start, v_stop, v_steps = resolve_v_sweep(args)
    sweep_cfg = PipelineSweepSettings(
        v_start=v_start,
        v_end=v_stop,
        v_steps=v_steps,
        retry_step=args.retry_step,
        jobs=max(1, int(args.erf_jobs or 1)),
        overwrite_simulation=bool(args.overwrite_erf),
        plot_erfs=False,
    )
    binary_overrides = BinaryRunSettings(
        warmup_steps=args.warmup_steps,
        simulation_steps=args.simulation_steps,
        sample_interval=args.sample_interval,
        batch_size=args.batch_size,
        seed=args.seed,
        output_name=args.output_name,
    )
    for r_value in r_eplus_values:
        print(f"=== Figure 3 workflow for R_Eplus = {r_value:.4f} ===")
        base_param = deepcopy(base_parameter)
        base_param["R_Eplus"] = float(r_value)
        column_contexts: List[ColumnContext] = []
        simulation_tasks: List[Dict[str, Any]] = []
        focus_union: set[int] = set()
        for idx, spec in enumerate(COLUMN_SPECS):
            column_param = _build_column_parameter(
                base_param,
                spec,
                column_override_map=column_override_map,
            )
            title_text = _resolve_column_title(column_param, spec, title_map=column_title_map)
            focus_counts = _resolve_focus_counts(column_param, parsed_focus_counts)
            focus_union.update(int(value) for value in focus_counts)
            column_param["R_Eplus"] = float(column_param.get("R_Eplus", r_value) or r_value)
            binary_cfg = _resolve_binary_config(column_param, binary_overrides)
            seed = int(binary_cfg.get("seed", 0) or 0)
            focus_markers, candidates, folder, bundle_path = _prepare_focus_markers_with_retry(
                column_param,
                focus_counts,
                args.stability_filter,
                sweep_cfg,
                delta_rep=float(args.delta_rep_mf),
                rep_retry=int(args.rep_retry_mf),
                rng_seed=args.rep_rng_seed,
                column_label=str(spec.label),
            )
            trace_path, task = _build_trace_task(
                column_param,
                binary_cfg,
                candidates=candidates,
                seed=seed,
                analysis_only=args.analysis_only,
                overwrite_simulation=args.overwrite_simulation,
            )
            if task:
                simulation_tasks.append(task)
            column_contexts.append(
                ColumnContext(
                    spec=spec,
                    parameter=column_param,
                    title=title_text,
                    focus_counts=focus_counts,
                    focus_markers=focus_markers,
                    folder=folder,
                    bundle_path=bundle_path,
                    binary_cfg=binary_cfg,
                    seed=seed,
                    trace_path=trace_path,
                )
            )
        if not focus_union:
            raise ValueError("No focus counts available for plotting.")
        color_map, colorbar_entries = _prepare_line_color_map(
            sorted(focus_union),
            colormap="viridis_r",
        )
        if simulation_tasks:
            job_count = max(1, int(args.jobs or 1))
            if job_count > 1 and len(simulation_tasks) > 1:
                max_workers = min(job_count, len(simulation_tasks))
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                    future_map = {
                        pool.submit(_simulate_binary_task, task): task for task in simulation_tasks
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        task = future_map[future]
                        try:
                            future.result()
                        except Exception as exc:
                            seed = task.get("seed", "?")
                            raise RuntimeError(
                                f"BinaryNetwork simulation failed for seed {seed}: {exc}"
                            ) from exc
            else:
                for task in simulation_tasks:
                    _simulate_binary_task(task)
        n_cols = len(column_contexts)
        fig = plt.figure(figsize=(4.2 * n_cols + 1.4, 5))
        outer = fig.add_gridspec(
            1,
            n_cols + 2,
            width_ratios=[1.0] * n_cols + 2*[0.05],
            wspace=0.28,
            left=0.06,
            right=0.99,
            top=0.92,
            bottom=0.12,
        )
        colorbar_ax = fig.add_subplot(outer[0, -2])
        rate_axes: List[plt.Axes] = []
        for idx, context in enumerate(column_contexts):
            param_copy = context.parameter
            try:
                payload = _load_trace_payload(context.trace_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load trace for column {context.spec.label}: {exc}"
                ) from exc
            column_grid = outer[0, idx].subgridspec(2, 1, height_ratios=[1.0, 0.6], hspace=0.08)
            ax_raster = fig.add_subplot(column_grid[0, 0])
            if rate_axes:
                ax_rates = fig.add_subplot(column_grid[1, 0], sharex=ax_raster, sharey=rate_axes[0])
            else:
                ax_rates = fig.add_subplot(column_grid[1, 0], sharex=ax_raster)
            _plot_example_traces(
                ax_raster,
                ax_rates,
                payload,
                parameter=param_copy,
                raster_duration=args.raster_duration,
                rates_duration=args.rates_duration,
                raster_stride=args.raster_stride,
                font_cfg=font_cfg,
                focus_markers=context.focus_markers,
                color_map=color_map,
            )
            ax_raster.set_title(context.title, fontsize=font_cfg.title)
            style_axes(ax_raster, font_cfg, set_xlabel=False, set_ylabel=False)
            style_axes(ax_rates, font_cfg)
            panel_prefix = context.spec.label.strip()
            add_panel_label(ax_raster, f"{panel_prefix}1", font_cfg, x=-0.15, y=1.04)
            add_panel_label(ax_rates, f"{panel_prefix}2", font_cfg, x=-0.15, y=1.06)
            rate_axes.append(ax_rates)
        if rate_axes:
            for idx, ax in enumerate(rate_axes):
                ax.set_ylim(0.0, 1.05)
                ax.set_yticks([0.0, 1.0])
                if idx == 0:
                    ax.set_ylabel(r"$m_c$")
                    ax.set_yticklabels(["0", "1"])
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
                ax.set_xlim(0,5)
        colorbar = draw_listed_colorbar(
            fig,
            colorbar_ax,
            colorbar_entries,
            font_cfg=font_cfg,
            label="MF prediction: # active clusters",
            height_fraction=0.8,
            width_fraction=1.0,
            use_parent_axis=False,
        )
        colorbar_ax.get_xaxis().set_visible(False)
        colorbar_ax.get_yaxis().set_visible(False)
        if colorbar is not None:
            cbar_ax = colorbar.ax
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=5,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    label="stable",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label="unstable",
                ),
            ]
            cbar_ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(2., -0.05),
                frameon=False,
                fontsize=font_cfg.tick,
                borderpad=0.0,
                handletextpad=-0.4,
                labelspacing=0.2,
            )
            pos = cbar_ax.get_position()
            cbar_ax.set_axes_locator(None)  # key line
            cbar_ax.set_position([pos.x0 - 0.025, pos.y0 + 0.07, pos.width, pos.height])
        _save_figure(fig, args.output_prefix, r_value)
        plt.close(fig)


if __name__ == "__main__":
    main()
