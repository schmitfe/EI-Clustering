from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from binary_pipeline import run_binary_simulation
from MeanField.rate_system import ensure_output_folder
from sim_config import add_override_arguments, load_from_args, sim_tag_from_cfg, write_yaml_config

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
    add_override_arguments(parser)
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
    parser.add_argument(
        "--fixpoints-yaml",
        type=str,
        help="Optional focus_fixpoints YAML used to mark fixed points in the histogram.",
    )
    return parser.parse_args()


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
    return cfg


def _filtered_parameter_for_tag(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    for key in ("R_Eplus", "focus_count", "focus_counts"):
        filtered.pop(key, None)
    return filtered


def _prepare_output_folders(parameter: Dict[str, Any]) -> Tuple[str, str, str]:
    filtered = _filtered_parameter_for_tag(parameter)
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


def _load_trace_rates(trace_path: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace {trace_path} does not exist.")
    with np.load(trace_path, allow_pickle=True) as data:
        if "rates" not in data or "names" not in data:
            raise ValueError(f"{trace_path} does not contain 'rates' and 'names'.")
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"]]
    return rates, names


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


def _load_focus_fixpoints(path: str, excit_count: int | None) -> Dict[int, List[float]]:
    if not path:
        return {}
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to load fixpoint references. Install it via 'pip install pyyaml'."
        ) from YAML_ERROR
    if excit_count is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fixpoint reference {path} does not exist.")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    focus_rates: Dict[int, List[float]] = {}
    entries = data.get("focus_fixpoints", {})
    for focus_label, payload in entries.items():
        try:
            focus_count = int(focus_label)
        except ValueError:
            continue
        values: List[float] = []
        for fixpoint in payload.get("fixpoints", []):
            rates = _extract_focus_rates(fixpoint.get("rates"), focus_count, excit_count)
            values.extend(rates)
        if values:
            focus_rates[focus_count] = values
    return focus_rates


def _prepare_matplotlib():
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to plot histograms. Install it via 'pip install matplotlib'.") from exc
    return plt


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    binary_cfg = _resolve_binary_config(parameter, args)
    bin_size = max(1, int(args.bin_size))
    bins = max(1, int(args.bins))
    total_simulations = max(0, int(args.simulations))
    folder, binary_dir, analysis_dir = _prepare_output_folders(parameter)
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
    if args.fixpoints_yaml:
        metadata["focus_fixpoints_file"] = os.path.abspath(args.fixpoints_yaml)
        metadata_changed = True
    excitatory_clusters = metadata.get("excitatory_clusters")
    if metadata_changed:
        _save_metadata(metadata_path, metadata)
    seeds = [base_seed + idx for idx in range(total_simulations)]
    pooled_entries: List[float] = []
    seen_seeds: List[int] = []
    focus_reference = metadata.get("focus_fixpoints_file")
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
    focus_rates = {}
    reference_path = args.fixpoints_yaml or focus_reference
    if reference_path and excitatory_clusters:
        try:
            focus_rates = _load_focus_fixpoints(reference_path, excitatory_clusters)
        except FileNotFoundError as exc:
            print(f"Fixpoint reference warning: {exc}")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise exc
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.linspace(0.0, 1.0, max(2, bins + 1), endpoint=True)
    counts, _, _ = ax.hist(
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
        for idx, (focus_count, values) in enumerate(sorted(focus_rates.items())):
            if not values:
                continue
            y_level = marker_base + idx * marker_step
            ax.scatter(
                values,
                np.full(len(values), y_level),
                marker="v",
                s=36,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidths=0.4,
                label=f"Focus count {focus_count}",
            )
    ax.set_xlabel("Mean firing rate")
    ax.set_ylabel("Bin frequency")
    ax.set_title("Distribution of maximum excitatory rates")
    ax.set_xlim(0.0, 1.0)
    if focus_rates:
        ax.legend()
    ax.tight_layout()
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
        "focus_fixpoints_file": reference_path,
    }
    write_yaml_config(summary, os.path.join(analysis_dir, "analysis_summary.yaml"))
    print(f"Stored pooled maxima at {pooled_path}")
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
