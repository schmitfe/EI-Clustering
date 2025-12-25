from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from binary_pipeline import ensure_binary_behavior_defaults, run_binary_simulation
from MeanField.rate_system import ensure_output_folder
from sim_config import deep_update, parse_overrides, sim_tag_from_cfg, write_yaml_config

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    yaml = None
    YAML_ERROR = exc
else:  # pragma: no cover - optional dependency
    YAML_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate multiple clustered binary EI networks with different seeds and "
            "build the distribution of maximum excitatory cluster rates."
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
    return parser.parse_args()


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


def _apply_overrides(parameter: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
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
    cfg["warmup_steps"] = args.warmup_steps if args.warmup_steps is not None else cfg.get("warmup_steps", 5000)
    cfg["simulation_steps"] = (
        args.simulation_steps if args.simulation_steps is not None else cfg.get("simulation_steps", 20000)
    )
    cfg["sample_interval"] = (
        args.sample_interval if args.sample_interval is not None else cfg.get("sample_interval", 10)
    )
    cfg["batch_size"] = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)
    cfg["seed"] = args.seed if args.seed is not None else cfg.get("seed")
    cfg["output_name"] = args.output_name or cfg.get("output_name", "activity_trace")
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
    return {
        "rates": rates,
        "names": names,
        "times": times,
        "states": states,
        "sample_interval": sample_interval,
    }


def _load_trace_rates(trace_path: str) -> Tuple[np.ndarray, List[str]]:
    payload = _load_trace_payload(trace_path)
    return payload["rates"], payload["names"]


def _compute_maxima_from_trace(trace_path: str, bin_samples: int) -> Tuple[List[float], int]:
    rates, names = _load_trace_rates(trace_path)
    if rates.ndim != 2 or rates.shape[0] == 0:
        raise ValueError(f"{trace_path} contains an empty rates array.")
    excit_indices = [idx for idx, name in enumerate(names) if name.startswith("E")]
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


def _extract_focus_rates(rates: Sequence[float] | None, focus_count: int, excit_count: int) -> List[float]:
    if rates is None or focus_count <= 0 or excit_count <= 0:
        return []
    focus_count = min(int(focus_count), excit_count)
    arr = np.asarray(rates, dtype=float).ravel()
    if arr.size < focus_count:
        focus_count = arr.size
    return [float(value) for value in arr[:focus_count]]


def _load_focus_fixpoints_from_bundle(
    bundle: Dict[str, Any],
    excit_count: int | None,
    target_rep: float,
) -> Dict[int, Dict[str, List[float]]]:
    if excit_count is None:
        return {}
    focus_rates: Dict[int, Dict[str, List[float]]] = {}
    fixpoints = bundle.get("fixpoints", {})
    for focus_label, entries in fixpoints.items():
        try:
            focus_count = int(focus_label)
        except (TypeError, ValueError):
            continue
        stable_values: List[float] = []
        unstable_values: List[float] = []
        for rep_label, rep_entries in entries.items():
            rep_value = _parse_rep_from_key(str(rep_label))
            if rep_value is None or abs(rep_value - target_rep) > 1e-9:
                continue
            if not isinstance(rep_entries, dict):
                continue
            for fixpoint in rep_entries.values():
                rates = _extract_focus_rates(fixpoint.get("rates"), focus_count, excit_count)
                if not rates:
                    continue
                stability = str(fixpoint.get("stability", "") or "").lower()
                if stability == "stable":
                    stable_values.extend(rates)
                else:
                    unstable_values.extend(rates)
        if stable_values or unstable_values:
            focus_rates[focus_count] = {"stable": stable_values, "unstable": unstable_values}
    return focus_rates


def _prepare_matplotlib():
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to plot histograms. Install it via 'pip install matplotlib'.") from exc
    return plt


def _plot_onset_raster(ax, states: np.ndarray, sample_interval: int, excitatory_neurons: int):
    if states.size == 0:
        ax.text(0.5, 0.5, "No neuron state samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Neuron index")
        ax.set_title("Neuron onset raster")
        return
    steps, neuron_count = states.shape
    sample_interval = max(1, int(sample_interval))
    state_int = states.astype(np.int16, copy=False)
    if steps <= 1:
        ax.text(0.5, 0.5, "Insufficient samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Neuron index")
        ax.set_title("Neuron onset raster")
        return
    transitions = np.argwhere(np.diff(state_int, axis=0) == 1)
    if transitions.size == 0:
        ax.text(0.5, 0.5, "No onset transitions recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Neuron index")
        ax.set_title("Neuron onset raster")
        return
    excitatory_limit = max(0, min(int(excitatory_neurons), neuron_count))
    times = (transitions[:, 0] + 1) * sample_interval
    neurons = transitions[:, 1]
    excit_mask = neurons < excitatory_limit
    inhib_mask = neurons >= excitatory_limit
    if excit_mask.any():
        ax.scatter(times[excit_mask], neurons[excit_mask], s=6, marker=".", color="black", label="Excitatory")
    if inhib_mask.any():
        ax.scatter(times[inhib_mask], neurons[inhib_mask], s=6, marker=".", color="#8B0000", label="Inhibitory")
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Neuron index")
    ax.set_ylim(-0.5, neuron_count - 0.5)
    if excit_mask.any() and inhib_mask.any():
        ax.legend(loc="upper right")
    ax.set_title("Neuron onset raster")


def _plot_excitatory_rates(ax, times: np.ndarray, rates: np.ndarray, names: Sequence[str]):
    if rates.size == 0 or not names:
        ax.text(0.5, 0.5, "No rates available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Activity")
        ax.set_title("Excitatory rates over time")
        return
    excit_indices = [idx for idx, name in enumerate(names) if name.startswith("E")]
    if not excit_indices:
        ax.text(0.5, 0.5, "No excitatory populations", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Activity")
        ax.set_title("Excitatory rates over time")
        return
    times = np.asarray(times, dtype=float)
    if times.size != rates.shape[0]:
        times = np.arange(rates.shape[0], dtype=float)
    for idx in excit_indices:
        ax.plot(times, rates[:, idx], linewidth=0.8, label=names[idx])
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Activity")
    ax.set_title("Excitatory rates over time")
    if len(excit_indices) <= 10:
        ax.legend(loc="upper right", fontsize=8, ncol=2)


def main() -> None:
    args = parse_args()
    parameter, folder_hint, fixpoints_path = _resolve_simulation_source(
        args.source,
        fixpoint_hint=args.fixpoints,
        overrides=args.overwrite,
    )
    target_rep = parameter.get("R_Eplus")
    if target_rep is None:
        raise ValueError(
            "Parameter set must define R_Eplus to align fixpoints with the simulated network. "
            "Use -O R_Eplus=<value> if it is missing."
        )
    target_rep = float(target_rep)
    binary_cfg = _resolve_binary_config(parameter, args)
    bin_size = max(1, int(args.bin_size))
    bins = max(1, int(args.bins))
    total_simulations = max(0, int(args.simulations))
    folder, binary_dir, analysis_dir = _prepare_output_folders(parameter, base_folder=folder_hint)
    bundle = _load_fixpoint_bundle(fixpoints_path)
    metadata_path = os.path.join(analysis_dir, "metadata.yaml")
    metadata = _load_metadata(metadata_path) if os.path.exists(metadata_path) else {}
    metadata_changed = False
    base_output = binary_cfg.get("output_name", "activity_trace")
    existing_base = metadata.get("base_output_name")
    if existing_base and existing_base != base_output:
        raise ValueError(
            f"Analysis folder {analysis_dir} already tracks output '{existing_base}'. "
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
    metadata_seed = metadata.get("base_seed")
    if metadata_seed is None or metadata_seed != base_seed:
        metadata["base_seed"] = base_seed
        metadata_changed = True
    stored_fixpoints = metadata.get("fixpoints_file")
    if stored_fixpoints:
        if os.path.abspath(str(stored_fixpoints)) != os.path.abspath(fixpoints_path):
            raise ValueError(
                f"Analysis folder {analysis_dir} already references fixpoints file {stored_fixpoints}. "
                f"Requested {fixpoints_path} would mix incompatible runs."
            )
    else:
        metadata["fixpoints_file"] = os.path.abspath(fixpoints_path)
        metadata_changed = True
    excitatory_clusters = metadata.get("excitatory_clusters")
    if metadata_changed:
        _save_metadata(metadata_path, metadata)
    seeds = [base_seed + idx for idx in range(total_simulations)]
    pooled_entries: List[float] = []
    seen_seeds: List[int] = []
    focus_reference = os.path.abspath(fixpoints_path)
    example_trace_path: str | None = None
    example_seed: int | None = None
    for seed in seeds:
        label = _format_seed_label(base_output, seed)
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        maxima_path = os.path.join(analysis_dir, f"{label}_maxima.npz")
        trace_exists = os.path.exists(trace_path)
        if not args.analysis_only and (args.overwrite_simulation or not trace_exists):
            run_cfg = dict(binary_cfg)
            run_cfg["seed"] = seed
            print(f"Simulating binary network for seed {seed}.")
            result = run_binary_simulation(parameter, run_cfg, output_name=label)
            trace_path = result["trace_path"]
            trace_exists = True
        if not trace_exists:
            print(f"Skipping seed {seed}: trace {trace_path} is missing.")
            continue
        maxima: List[float] | None = None
        excit_count: int | None = None
        if not args.overwrite_analysis and os.path.exists(maxima_path):
            cached, cached_excit = _load_maxima_file(maxima_path, bin_size)
            if cached is not None:
                maxima = cached
                excit_count = cached_excit
            else:
                print(f"Recomputing maxima for seed {seed}: bin size mismatch or corrupt cache.")
        if maxima is None:
            try:
                maxima, excit_count = _compute_maxima_from_trace(trace_path, bin_size)
            except ValueError as exc:
                print(f"Skipping seed {seed}: {exc}")
                continue
            _save_maxima_file(maxima_path, maxima, seed, bin_size, trace_path, excit_count)
        pooled_entries.extend(maxima)
        seen_seeds.append(seed)
        if excitatory_clusters is None and excit_count is not None:
            excitatory_clusters = excit_count
            metadata["excitatory_clusters"] = excit_count
            metadata_changed = True
        if example_trace_path is None and os.path.exists(trace_path):
            example_trace_path = trace_path
            example_seed = seed
    if metadata_changed:
        _save_metadata(metadata_path, metadata)
    if not pooled_entries:
        print("No maxima were collected. Ensure simulations ran successfully.")
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
        focus_rates = _load_focus_fixpoints_from_bundle(bundle, excitatory_clusters, target_rep)
    example_payload: Dict[str, Any] | None = None
    if example_trace_path:
        try:
            example_payload = _load_trace_payload(example_trace_path)
        except Exception as exc:
            print(f"Warning: could not load example trace {example_trace_path}: {exc}")
            example_payload = None
    plt = _prepare_matplotlib()
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_hist = fig.add_subplot(grid[0, :])
    ax_raster = fig.add_subplot(grid[1, 0])
    ax_rates = fig.add_subplot(grid[1, 1])
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
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(focus_rates))))
        for idx, (focus_count, payload) in enumerate(sorted(focus_rates.items())):
            stable_values = payload.get("stable", [])
            unstable_values = payload.get("unstable", [])
            if not stable_values and not unstable_values:
                continue
            y_level = marker_base + idx * marker_step
            color = colors[idx % len(colors)]
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
    ax_hist.set_title("Distribution of maximum excitatory rates")
    ax_hist.set_xlim(0.0, 1.0)
    if focus_rates:
        ax_hist.legend()
    if example_payload is not None:
        excit_neurons = int(parameter.get("N_E", example_payload["states"].shape[1] if example_payload["states"].ndim == 2 else 0) or 0)
        _plot_onset_raster(ax_raster, example_payload["states"], example_payload["sample_interval"], excit_neurons)
        _plot_excitatory_rates(ax_rates, example_payload["times"], example_payload["rates"], example_payload["names"])
        if example_seed is not None:
            ax_raster.set_title(ax_raster.get_title() + f" (seed {example_seed})")
            ax_rates.set_title(ax_rates.get_title() + f" (seed {example_seed})")
    else:
        ax_raster.text(0.5, 0.5, "No trace available for raster plot", ha="center", va="center", transform=ax_raster.transAxes)
        ax_raster.set_axis_off()
        ax_rates.text(0.5, 0.5, "No rate trace available", ha="center", va="center", transform=ax_rates.transAxes)
        ax_rates.set_axis_off()
    plt.tight_layout()
    hist_path = os.path.join(analysis_dir, "max_rates_histogram.png")
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    summary = {
        "folder": folder,
        "analysis_dir": analysis_dir,
        "pooled_maxima_file": os.path.basename(pooled_path),
        "histogram_file": os.path.basename(hist_path),
        "seeds": seen_seeds,
        "pooled_samples": len(pooled_entries),
        "bin_size": bin_size,
        "fixpoints_file": focus_reference,
    }
    write_yaml_config(summary, os.path.join(analysis_dir, "analysis_summary.yaml"))
    print(f"Stored pooled maxima at {pooled_path}")
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
