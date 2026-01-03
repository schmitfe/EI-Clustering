from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import maximum_rates_distribution as base
from sim_config import write_yaml_config

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
LEGACY_ANALYSIS_SUBDIR = "max_rates_distribution_fp_init_legacy"
ACTIVE_TAIL_FRACTION = 0.1
ACTIVE_GAP_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate clustered binary EI networks initialized from fixed points using "
            "the legacy Rost implementation and build the distribution of maximum excitatory cluster rates."
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
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps from the config.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps from the config.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval from the config.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size from the config.")
    parser.add_argument("--seed", type=int, help="Base random seed for numpy (each replica offsets this value).")
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for saved traces (defaults to binary.output_name or 'activity_trace').",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20,
        help="Total number of network instances to consider (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (N_jobs) used to parallelize per-seed work (default: %(default)s).",
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=1,
        help="Total number of distributed jobs when running via SLURM (default: %(default)s).",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Zero-based index of this job among the --job-count partitions (default: %(default)s).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=50,
        help="Samples per time bin when averaging firing rates (default: %(default)s).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count for pooled maxima (default: %(default)s).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new simulations and only reuse existing traces/maxima.",
    )
    parser.add_argument(
        "--overwrite-simulation",
        action="store_true",
        help="Re-run simulations even if a matching trace already exists.",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="Recompute per-run maxima even if cached data are available.",
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
        "--active-count-method",
        choices=("gap", "sign"),
        default="gap",
        help="Strategy for counting active clusters from rates (default: %(default)s).",
    )
    return parser.parse_args()


def _prepare_analysis_folder(parameter: Dict[str, Any], base_folder: str | None) -> Tuple[str, str, str]:
    folder, binary_dir, _ = base._prepare_output_folders(parameter, base_folder=base_folder)
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


def _maxima_by_active_cluster(
    active_records: Sequence[Dict[str, Any]],
    seed_results: Dict[int, Dict[str, Any]],
) -> Dict[int, List[float]]:
    maxima_by_cluster: Dict[int, List[float]] = {}
    for record in active_records:
        seed = int(record.get("seed", -1))
        final_active = record.get("final_active_clusters")
        if final_active is None:
            continue
        per_seed = seed_results.get(seed)
        if not per_seed:
            continue
        maxima = per_seed.get("maxima")
        if maxima is None:
            continue
        values = np.asarray(maxima, dtype=float).ravel()
        if values.size == 0:
            continue
        maxima_by_cluster.setdefault(int(final_active), []).extend(values.tolist())
    return maxima_by_cluster


def _plot_active_cluster_histograms(
    analysis_dir: str,
    maxima_by_cluster: Dict[int, Sequence[float]],
    bin_count: int,
    focus_expectations: Dict[int, float],
    plt_module,
    *,
    focus_colors: Dict[int, Any],
) -> Tuple[str | None, str | None]:
    if not maxima_by_cluster:
        return None, None
    cluster_keys = sorted(maxima_by_cluster)
    if not cluster_keys:
        return None, None
    cols = min(3, max(1, len(cluster_keys)))
    rows = int(math.ceil(len(cluster_keys) / cols))
    fig, axes = plt_module.subplots(
        rows,
        cols,
        figsize=(6 * cols, 4.5 * rows),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.reshape(-1)
    bin_count = max(1, int(bin_count))
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1, endpoint=True)
    fallback_colors = plt_module.cm.tab10(np.linspace(0, 1, max(1, len(cluster_keys))))
    counts_payload: Dict[int, np.ndarray] = {}
    for idx, cluster_count in enumerate(cluster_keys):
        ax = axes_flat[idx]
        values = np.asarray(maxima_by_cluster[cluster_count], dtype=float)
        color = focus_colors.get(int(cluster_count))
        if color is None:
            color = fallback_colors[idx % len(fallback_colors)]
        hist, _ = np.histogram(values, bins=bin_edges)
        counts_payload[int(cluster_count)] = hist
        mean_val = float(np.mean(values)) if values.size else None
        median_val = float(np.median(values)) if values.size else None
        ax.hist(
            values,
            bins=bin_edges,
            color=color,
            alpha=0.8,
            label=f"{cluster_count} active clusters",
        )
        expectation = focus_expectations.get(int(cluster_count))
        if expectation is not None:
            ax.axvline(
                expectation,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"Mean-field expectation ({expectation:.3f})",
            )
        if mean_val is not None:
            ax.axvline(
                mean_val,
                color=color,
                linestyle="-.",
                linewidth=1.2,
                label=f"Empirical mean ({mean_val:.3f})",
            )
        if median_val is not None:
            ax.axvline(
                median_val,
                color=color,
                linestyle=":",
                linewidth=1.2,
                label=f"Empirical median ({median_val:.3f})",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Mean firing rate")
        ax.set_ylabel("Bin count")
        ax.set_title(f"{cluster_count} active clusters (samples: {len(values)})")
        ax.legend()
    for ax in axes_flat[len(cluster_keys):]:
        ax.set_visible(False)
    fig.tight_layout()
    hist_path = os.path.join(analysis_dir, "active_cluster_histograms.png")
    fig.savefig(hist_path, dpi=200)
    plt_module.close(fig)
    counts_path = os.path.join(analysis_dir, "active_cluster_hist_counts.npz")
    payload = {"bin_edges": bin_edges}
    payload.update({f"clusters_{count}": counts for count, counts in counts_payload.items()})
    np.savez_compressed(counts_path, **payload)
    return hist_path, counts_path


def _tail_means_from_trace(
    trace_path: str,
    window_fraction: float,
) -> Tuple[np.ndarray, List[str], int]:
    rates, names = base._load_trace_rates(trace_path)
    if rates.ndim != 2 or rates.shape[0] == 0:
        raise ValueError(f"{trace_path} contains an empty rates array.")
    excit_indices = [idx for idx, name in enumerate(names) if str(name).startswith("E")]
    if not excit_indices:
        raise ValueError(f"{trace_path} does not include excitatory populations.")
    excit_rates = rates[:, excit_indices]
    samples = excit_rates.shape[0]
    window = max(1, int(math.ceil(samples * window_fraction)))
    tail = excit_rates[-window:]
    means = tail.mean(axis=0)
    excit_names = [names[idx] for idx in excit_indices]
    return means, excit_names, window


def _build_spike_state_array(
    spike_payload: np.ndarray | None,
    neuron_count: int,
    warmup_steps: int,
    simulation_steps: int,
    sample_interval: int,
) -> np.ndarray:
    if spike_payload is None or spike_payload.size == 0 or neuron_count <= 0:
        return np.zeros((0, max(neuron_count, 0)), dtype=np.uint8)
    times = np.asarray(spike_payload[0], dtype=np.int64).ravel()
    units = np.asarray(spike_payload[2], dtype=np.int64).ravel()
    max_time = int(times.max()) if times.size else -1
    total_steps = max(warmup_steps + simulation_steps, max_time + 1)
    total_steps = max(total_steps, 0)
    states = np.zeros((total_steps, neuron_count), dtype=np.uint8)
    valid = (
        (units >= 0)
        & (units < neuron_count)
        & (times >= 0)
        & (times < total_steps)
    )
    if valid.any():
        states[times[valid], units[valid]] = 1
    if warmup_steps > 0 and warmup_steps < states.shape[0]:
        states = states[warmup_steps:]
    if simulation_steps > 0 and states.shape[0] > simulation_steps:
        states = states[:simulation_steps]
    sample_interval = max(1, int(sample_interval))
    if sample_interval > 1 and states.shape[0]:
        states = states[::sample_interval]
    return states


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
    if capture_spikes:
        legacy_params["return_mean_cluster_rates"] = False
        legacy_params["return_max_cluster_rates"] = False
        legacy_params["return_cluster_rates_and_spiketimes"] = True
    else:
        legacy_params["return_mean_cluster_rates"] = True
        legacy_params["return_max_cluster_rates"] = False
        legacy_params["return_cluster_rates_and_spiketimes"] = False
    return legacy_params


def _run_legacy_binary_simulation(
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
    state_stride = int(legacy_params.get("state_record_stride", 1) or 1)
    if capture_state_dynamics:
        if capture_spikes:
            raise ValueError("Cannot capture spikes and state dynamics simultaneously.")
        legacy_params["return_mean_cluster_rates"] = False
        legacy_params["return_max_cluster_rates"] = False
        legacy_params["return_cluster_rates_and_spiketimes"] = False
        legacy_params["return_state_dynamics"] = True
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
    if capture_spikes:
        spike_payload = np.asarray(raw_output.get("spike_times"), dtype=float)
        if spike_payload.size == 0:
            raise RuntimeError("Legacy simulation did not provide spike times.")
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
    warmup_steps = int(binary_cfg.get("warmup_steps", 0) or 0)
    simulation_steps = int(binary_cfg.get("simulation_steps", 0) or 0)
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
    if not capture_state_dynamics and capture_spikes and spike_payload is not None:
        total_neurons = int(parameter.get("N_E", 0) or 0) + int(parameter.get("N_I", 0) or 0)
        neuron_states = _build_spike_state_array(
            spike_payload,
            total_neurons,
            warmup_steps,
            simulation_steps,
            sample_interval,
        )
    trace_path = os.path.join(binary_dir, f"{output_label}.npz")
    np.savez_compressed(
        trace_path,
        rates=rates,
        names=np.asarray(names, dtype=object),
        times=times.astype(np.float32),
        sample_interval=np.array(int(sample_interval), dtype=np.int64),
        neuron_states=neuron_states,
        neuron_state_interval=np.array(int(state_stride), dtype=np.int64),
        neuron_state_times=state_times.astype(np.int64),
        subthreshold_fields=subthreshold_fields,
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
            trace_path = _run_legacy_binary_simulation(
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
            cached, cached_excit = base._load_maxima_file(maxima_path, int(task["bin_size"]))
            if cached is not None:
                maxima = cached
                excit_count = cached_excit
            else:
                result["logs"].append(
                    f"Recomputing maxima for seed {seed}: bin size mismatch or corrupt cache."
                )
        if maxima is None:
            maxima, excit_count = base._compute_maxima_from_trace(trace_path, int(task["bin_size"]))
            base._save_maxima_file(maxima_path, maxima, seed, int(task["bin_size"]), trace_path, int(excit_count))
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


def main() -> None:
    args = parse_args()
    parameter, folder_hint, fixpoints_path = base._resolve_simulation_source(
        args.source,
        fixpoint_hint=args.fixpoints,
        overrides=args.overwrite,
    )
    target_rep = parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError(
            "Parameter set must define R_Eplus to align fixpoints with the simulated network."
        )
    target_rep = float(target_rep)
    bundle = base._load_fixpoint_bundle(fixpoints_path)
    binary_cfg = base._resolve_binary_config(parameter, args)
    bin_size = max(1, int(args.bin_size))
    bins = max(1, int(args.bins))
    total_simulations = max(0, int(args.simulations))
    active_count_method = str(args.active_count_method or "gap").lower()
    folder, binary_dir, analysis_dir = _prepare_analysis_folder(parameter, folder_hint)
    metadata_path = os.path.join(analysis_dir, "metadata.yaml")
    metadata = base._load_metadata(metadata_path) if os.path.exists(metadata_path) else {}
    metadata_changed = False
    base_output = binary_cfg.get("output_name", "activity_trace")
    existing_base = metadata.get("base_output_name")
    if existing_base and existing_base != base_output:
        raise ValueError(
            f"Analysis folder {analysis_dir} already stores data for output '{existing_base}'. "
            f"Requested base name '{base_output}' would mix incompatible runs."
        )
    if existing_base is None:
        metadata["base_output_name"] = base_output
        metadata_changed = True
    if metadata.get("bin_size") != bin_size:
        metadata["bin_size"] = bin_size
        metadata_changed = True
    base_seed = binary_cfg.get("seed")
    if base_seed is None:
        base_seed = metadata.get("base_seed", 0)
    if metadata.get("base_seed") != base_seed:
        metadata["base_seed"] = base_seed
        metadata_changed = True
    mode = metadata.get("mode")
    if mode not in {None, LEGACY_MODE_LABEL}:
        raise ValueError(
            f"Analysis folder {analysis_dir} already stores results for mode '{metadata['mode']}'. "
            "Create a new folder for the legacy fixpoint workflow."
        )
    if mode != LEGACY_MODE_LABEL:
        metadata["mode"] = LEGACY_MODE_LABEL
        metadata_changed = True
    stored_fixpoints = metadata.get("fixpoints_file")
    if stored_fixpoints:
        if os.path.abspath(str(stored_fixpoints)) != os.path.abspath(fixpoints_path):
            raise ValueError(
                f"Analysis folder {analysis_dir} already references {stored_fixpoints}. "
                f"Requested {fixpoints_path} would mix incompatible runs."
            )
    else:
        metadata["fixpoints_file"] = os.path.abspath(fixpoints_path)
        metadata_changed = True
    focus_filter = _normalize_focus_list(args.focus_counts)
    stored_focus = _normalize_focus_list(metadata.get("focus_counts"))
    if focus_filter is None:
        focus_filter = stored_focus
    elif stored_focus is not None and stored_focus != focus_filter:
        raise ValueError(
            f"Analysis folder already constrained focus_counts to {stored_focus}. "
            f"Requested {focus_filter} would mix incompatible runs."
        )
    elif stored_focus is None:
        metadata["focus_counts"] = focus_filter
        metadata_changed = True
    stability_filter = args.stability_filter.lower()
    stored_stability = str(metadata.get("stability_filter") or "").lower() or None
    if stored_stability is None:
        metadata["stability_filter"] = stability_filter
        metadata_changed = True
    elif stored_stability != stability_filter:
        raise ValueError(
            f"Analysis folder already uses stability filter '{stored_stability}'. "
            f"Requested '{stability_filter}' would mix incompatible runs."
        )
    else:
        stability_filter = stored_stability
    metadata["fixpoints_file"] = os.path.abspath(fixpoints_path)
    stored_active_method = str(metadata.get("active_count_method") or "").lower() or None
    if stored_active_method is None:
        metadata["active_count_method"] = active_count_method
        metadata_changed = True
    elif stored_active_method != active_count_method:
        raise ValueError(
            f"Analysis folder already uses active-count method '{stored_active_method}'. "
            f"Requested '{active_count_method}' would mix incompatible runs."
        )
    if metadata_changed:
        base._save_metadata(metadata_path, metadata)
        metadata_changed = False
    excitatory_clusters = metadata.get("excitatory_clusters")
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive to determine population counts.")
    pop_vector_length = 2 * Q_value
    raw_candidates = _load_fixpoint_candidates(bundle, focus_filter, stability_filter, pop_vector_length, target_rep)
    candidates = _deduplicate_candidates_by_focus(raw_candidates, Q_value)
    seeds = [int(base_seed) + idx for idx in range(total_simulations)]
    worker_jobs = int(args.jobs)
    if worker_jobs <= 0:
        raise ValueError("--jobs must be a positive integer.")
    job_count = int(args.job_count)
    if job_count <= 0:
        raise ValueError("--job-count must be a positive integer.")
    job_index = int(args.job_index)
    assigned_seeds = _shard_seed_list(seeds, job_count, job_index)
    if not assigned_seeds:
        print(
            f"Job index {job_index} / {job_count} has no seeds to process "
            f"(requested simulations: {len(seeds)})."
        )
        return
    pooled_entries: List[float] = []
    seen_seeds: List[int] = []
    assignment_cache: Dict[int, Dict[str, Any]] = {}
    focus_reference = os.path.abspath(fixpoints_path)
    example_trace_path: str | None = None
    example_seed: int | None = None
    example_states_available = False
    confusion_matrix = np.zeros((Q_value + 1, Q_value + 1), dtype=np.int64)
    active_cluster_records: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    capture_assigned = False
    active_count_kwargs = {
        "method": active_count_method,
        "tolerance": ACTIVE_GAP_TOLERANCE,
    }
    for seed in assigned_seeds:
        label = base._format_seed_label(base_output, seed)
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        maxima_path = os.path.join(analysis_dir, f"{label}_maxima.npz")
        candidate = _candidate_for_seed(seed, candidates)
        assignment_cache[int(seed)] = candidate
        trace_exists = os.path.exists(trace_path)
        needs_simulation = (not args.analysis_only) and (args.overwrite_simulation or not trace_exists)
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
                "binary_cfg": binary_cfg,
                "binary_dir": binary_dir,
                "bin_size": bin_size,
                "needs_simulation": needs_simulation,
                "capture_spikes": capture_spikes,
                "overwrite_analysis": bool(args.overwrite_analysis),
                "candidate_id": candidate.get("id"),
                "candidate_focus": candidate.get("focus_count"),
                "candidate_stability": candidate.get("stability"),
                "init_rates": tuple(float(value) for value in candidate["rates"]),
            }
        )
    seed_results: Dict[int, Dict[str, Any]] = {}

    def _record_result(result: Dict[str, Any]) -> None:
        if not result:
            return
        seed_results[int(result.get("seed"))] = result
        _emit_worker_logs(result)

    if tasks:
        if worker_jobs == 1:
            for task in tasks:
                _record_result(_process_seed_task(task))
        else:
            max_workers = min(worker_jobs, len(tasks))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_seed = {pool.submit(_process_seed_task, task): task["seed"] for task in tasks}
                for future in concurrent.futures.as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - worker crash
                        print(f"Seed {seed}: worker crashed with {exc.__class__.__name__}: {exc}")
                        continue
                    _record_result(result)
    processed_all_seeds = len(assigned_seeds) == len(seeds)
    for task in tasks:
        seed = task["seed"]
        result = seed_results.get(seed)
        if not result:
            continue
        maxima = result.get("maxima")
        if maxima is None:
            continue
        pooled_entries.extend(list(maxima))
        seen_seeds.append(seed)
        excit_count = result.get("excitatory_clusters")
        if excitatory_clusters is None and excit_count is not None:
            excitatory_clusters = excit_count
            metadata["excitatory_clusters"] = excit_count
            metadata_changed = True
        if result.get("states_recorded"):
            example_states_available = True
        trace_path = result.get("trace_path") or task["trace_path"]
        if os.path.exists(trace_path):
            if example_trace_path is None:
                example_trace_path = trace_path
                example_seed = seed
            elif not example_states_available:
                try:
                    payload = base._load_trace_payload(trace_path)
                    if payload.get("states") is not None and payload["states"].size:
                        example_trace_path = trace_path
                        example_seed = seed
                        example_states_available = True
                except Exception:
                    pass
        candidate = assignment_cache.get(seed)
        if candidate is None:
            candidate = _candidate_for_seed(seed, candidates)
            assignment_cache[int(seed)] = candidate
        focus_active = int(candidate.get("focus_count", 0) or 0)
        initial_active_estimate = _active_cluster_count_from_means(
            np.asarray(candidate["rates"][:Q_value], dtype=float),
            **active_count_kwargs,
        )
        tail_means = None
        tail_window = 0
        excit_names: List[str] = []
        try:
            tail_means, excit_names, tail_window = _tail_means_from_trace(trace_path, ACTIVE_TAIL_FRACTION)
        except Exception as exc:
            print(f"Could not compute active cluster statistics for seed {seed}: {exc}")
        if tail_means is not None:
            final_active = _active_cluster_count_from_means(tail_means, **active_count_kwargs)
            init_idx = max(0, min(Q_value, int(focus_active)))
            final_idx = max(0, min(Q_value, int(final_active)))
            confusion_matrix[init_idx, final_idx] += 1
            record = {
                "seed": int(seed),
                "fixpoint_id": candidate["id"],
                "initial_focus_count": init_idx,
                "initial_active_clusters_estimate": int(initial_active_estimate),
                "final_active_clusters": final_idx,
                "tail_window_samples": int(tail_window),
                "tail_means": {name: float(value) for name, value in zip(excit_names, tail_means)},
            }
            active_cluster_records.append(record)
            print(
                f"Seed {seed}: initial focus {init_idx} (estimate {initial_active_estimate:.2f}), "
                f"final active clusters {final_idx} "
                f"(tail window {tail_window} samples)."
            )
    if metadata_changed:
        base._save_metadata(metadata_path, metadata)
    if not pooled_entries:
        print("No maxima were collected. Ensure simulations ran successfully.")
        return
    if not processed_all_seeds:
        print(
            f"Processed {len(assigned_seeds)} seeds out of {len(seeds)} total in shard {job_index + 1}/{job_count}. "
            "Skipping pooled summary; re-run once with --analysis-only (without sharding) after all jobs finish to "
            "build the combined histogram."
        )
        return
    pooled_array = np.asarray(pooled_entries, dtype=float)
    pooled_path = os.path.join(analysis_dir, "pooled_maxima.npz")
    np.savez_compressed(
        pooled_path,
        maxima=pooled_array,
        bin_size=np.array(bin_size, dtype=np.int64),
        seeds=np.asarray(seen_seeds, dtype=np.int64),
    )
    focus_rates: Dict[int, Dict[str, List[float]]] = {}
    if excitatory_clusters:
        focus_rates = _focus_payload_from_candidates(candidates, excitatory_clusters)
    example_payload: Dict[str, Any] | None = None
    if example_trace_path:
        try:
            example_payload = base._load_trace_payload(example_trace_path)
        except Exception as exc:  # pragma: no cover - plotting helper
            print(f"Warning: could not load example trace {example_trace_path}: {exc}")
            example_payload = None
    focus_expectations = _focus_expectations_from_payload(focus_rates)
    plt = base._prepare_matplotlib()
    focus_color_map = _build_focus_color_map(plt, focus_rates)
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
    ax_hist = fig.add_subplot(grid[0, :])
    ax_raster = fig.add_subplot(grid[1, 0])
    ax_rates = fig.add_subplot(grid[1, 1])
    ax_conf = fig.add_subplot(grid[1, 2])
    edges = np.linspace(0.0, 1.0, max(2, bins + 1), endpoint=True)
    counts, _, _ = ax_hist.hist(
        pooled_array,
        bins=edges,
        color="#7fb0ff",
        alpha=0.75,
        label="Max excitatory bin activity",
    )
    max_count = float(counts.max()) if counts.size else 1.0
    if focus_rates:
        marker_base = max_count * 1.05
        marker_step = max(max_count * 0.08, 0.05)
        fallback_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(focus_rates))))
        for idx, (focus_count, payload) in enumerate(sorted(focus_rates.items())):
            stable_values = payload.get("stable", [])
            unstable_values = payload.get("unstable", [])
            if not stable_values and not unstable_values:
                continue
            y_level = marker_base + idx * marker_step
            color = focus_color_map.get(int(focus_count))
            if color is None:
                color = fallback_colors[idx % len(fallback_colors)]
            if stable_values:
                ax_hist.scatter(
                    stable_values,
                    np.full(len(stable_values), y_level),
                    marker="v",
                    s=36,
                    color=color,
                    edgecolor="black",
                    linewidths=0.4,
                    label=f"Focus count {focus_count} (stable)",
                )
            if unstable_values:
                ax_hist.scatter(
                    unstable_values,
                    np.full(len(unstable_values), y_level),
                    marker="v",
                    s=36,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.0,
                    label=f"Focus count {focus_count} (unstable)",
                )
    ax_hist.set_xlabel("Mean firing rate")
    ax_hist.set_ylabel("Bin frequency")
    ax_hist.set_title("Distribution of maximum excitatory rates (legacy fixpoint initialization)")
    ax_hist.set_xlim(0.0, 1.0)
    if focus_rates:
        ax_hist.legend()
    if example_payload is not None:
        excit_neurons = int(
            parameter.get("N_E", example_payload["states"].shape[1] if example_payload["states"].ndim == 2 else 0) or 0
        )
        base._plot_onset_raster(ax_raster, example_payload["states"], example_payload["sample_interval"], excit_neurons)
        base._plot_excitatory_rates(ax_rates, example_payload["times"], example_payload["rates"], example_payload["names"])
        if example_seed is not None:
            ax_raster.set_title(ax_raster.get_title() + f" (seed {example_seed})")
            ax_rates.set_title(ax_rates.get_title() + f" (seed {example_seed})")
    else:
        ax_raster.text(0.5, 0.5, "No trace available for raster plot", ha="center", va="center", transform=ax_raster.transAxes)
        ax_raster.set_axis_off()
        ax_rates.text(0.5, 0.5, "No rate trace available", ha="center", va="center", transform=ax_rates.transAxes)
        ax_rates.set_axis_off()
    if active_cluster_records:
        im = ax_conf.imshow(confusion_matrix, origin="lower", cmap="Blues")
        ticks = np.arange(0, Q_value + 1)
        ax_conf.set_xticks(ticks)
        ax_conf.set_yticks(ticks)
        ax_conf.set_xlabel("Final active clusters")
        ax_conf.set_ylabel("Initial active clusters")
        ax_conf.set_title("Active cluster transitions")
        for i in ticks:
            for j in ticks:
                value = int(confusion_matrix[i, j])
                if value > 0:
                    color = "white" if value > confusion_matrix.max() * 0.5 else "black"
                    ax_conf.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04, label="Counts")
    else:
        ax_conf.text(0.5, 0.5, "No active-cluster statistics", ha="center", va="center", transform=ax_conf.transAxes)
        ax_conf.set_axis_off()
    fig.tight_layout()
    hist_path = os.path.join(analysis_dir, "max_rates_histogram.png")
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    active_hist_path: str | None = None
    active_counts_path: str | None = None
    maxima_by_cluster = _maxima_by_active_cluster(active_cluster_records, seed_results)
    if maxima_by_cluster:
        active_hist_path, active_counts_path = _plot_active_cluster_histograms(
            analysis_dir,
            maxima_by_cluster,
            bins,
            focus_expectations,
            plt,
            focus_colors=focus_color_map,
        )
    init_records = []
    for seed in seen_seeds:
        entry = assignment_cache.get(seed)
        if entry is None:
            entry = _candidate_for_seed(seed, candidates)
        init_records.append(
            {
                "seed": int(seed),
                "fixpoint_id": entry["id"],
                "focus_count": entry["focus_count"],
                "stability": entry["stability"],
                "value": entry.get("label"),
            }
        )
    summary = {
        "mode": LEGACY_MODE_LABEL,
        "folder": folder,
        "analysis_dir": analysis_dir,
        "pooled_maxima_file": os.path.basename(pooled_path),
        "histogram_file": os.path.basename(hist_path),
        "fixpoints_file": focus_reference,
        "focus_counts": focus_filter,
        "stability_filter": stability_filter,
        "seeds": seen_seeds,
        "pooled_samples": len(pooled_entries),
        "bin_size": bin_size,
        "init_fixpoints": init_records,
        "active_tail_fraction": ACTIVE_TAIL_FRACTION,
        "active_count_method": active_count_method,
        "active_cluster_stats": active_cluster_records,
        "confusion_matrix": confusion_matrix.tolist(),
        "active_cluster_histogram_file": os.path.basename(active_hist_path) if active_hist_path else None,
        "active_cluster_counts_file": os.path.basename(active_counts_path) if active_counts_path else None,
    }
    write_yaml_config(summary, os.path.join(analysis_dir, "analysis_summary.yaml"))
    print(f"Stored pooled maxima at {pooled_path}")
    print(f"Saved histogram to {hist_path}")
    if active_hist_path:
        print(f"Saved per-cluster histograms to {active_hist_path}")
    if active_counts_path:
        print(f"Stored per-cluster bin counts at {active_counts_path}")


if __name__ == "__main__":
    main()
