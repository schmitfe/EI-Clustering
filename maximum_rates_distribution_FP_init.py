from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import maximum_rates_distribution as base
from binary_pipeline import run_binary_simulation
from sim_config import write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate clustered binary EI networks initialized from fixed points and "
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
    return parser.parse_args()


def _prepare_analysis_folder(parameter: Dict[str, Any], base_folder: str | None) -> Tuple[str, str, str]:
    folder, binary_dir, _ = base._prepare_output_folders(parameter, base_folder=base_folder)
    analysis_dir = os.path.join(binary_dir, "max_rates_distribution_fp_init")
    os.makedirs(analysis_dir, exist_ok=True)
    return folder, binary_dir, analysis_dir


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
    if mode not in {None, "fixpoint_initialization"}:
        raise ValueError(
            f"Analysis folder {analysis_dir} already stores results for mode '{metadata['mode']}'. "
            "Create a new folder for the fixpoint-initialized workflow."
        )
    if mode != "fixpoint_initialization":
        metadata["mode"] = "fixpoint_initialization"
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
    if metadata_changed:
        base._save_metadata(metadata_path, metadata)
        metadata_changed = False
    excitatory_clusters = metadata.get("excitatory_clusters")
    Q_value = int(parameter.get("Q", 0) or 0)
    if Q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive to determine population counts.")
    pop_vector_length = 2 * Q_value
    candidates = _load_fixpoint_candidates(bundle, focus_filter, stability_filter, pop_vector_length, target_rep)
    seeds = [int(base_seed) + idx for idx in range(total_simulations)]
    pooled_entries: List[float] = []
    seen_seeds: List[int] = []
    assignment_cache: Dict[int, Dict[str, Any]] = {}
    focus_reference = os.path.abspath(fixpoints_path)
    for seed in seeds:
        label = base._format_seed_label(base_output, seed)
        trace_path = os.path.join(binary_dir, f"{label}.npz")
        maxima_path = os.path.join(analysis_dir, f"{label}_maxima.npz")
        candidate = _candidate_for_seed(seed, candidates)
        assignment_cache[int(seed)] = candidate
        trace_exists = os.path.exists(trace_path)
        if not args.analysis_only and (args.overwrite_simulation or not trace_exists):
            run_cfg = dict(binary_cfg)
            run_cfg["seed"] = seed
            print(
                f"Simulating binary network for seed {seed} using fixpoint {candidate['id']} "
                f"(focus {candidate['focus_count']}, {candidate['stability']})."
            )
            result = run_binary_simulation(
                parameter,
                run_cfg,
                output_name=label,
                population_rate_inits=candidate["rates"],
            )
            trace_path = result["trace_path"]
            trace_exists = True
        if not trace_exists:
            print(f"Skipping seed {seed}: trace {trace_path} is missing.")
            continue
        maxima: List[float] | None = None
        excit_count: int | None = None
        if not args.overwrite_analysis and os.path.exists(maxima_path):
            cached, cached_excit = base._load_maxima_file(maxima_path, bin_size)
            if cached is not None:
                maxima = cached
                excit_count = cached_excit
            else:
                print(f"Recomputing maxima for seed {seed}: bin size mismatch or corrupt cache.")
        if maxima is None:
            try:
                maxima, excit_count = base._compute_maxima_from_trace(trace_path, bin_size)
            except ValueError as exc:
                print(f"Skipping seed {seed}: {exc}")
                continue
            base._save_maxima_file(maxima_path, maxima, seed, bin_size, trace_path, excit_count)
        pooled_entries.extend(maxima)
        seen_seeds.append(seed)
        if excitatory_clusters is None and excit_count is not None:
            excitatory_clusters = excit_count
            metadata["excitatory_clusters"] = excit_count
            metadata_changed = True
    if metadata_changed:
        base._save_metadata(metadata_path, metadata)
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
        focus_rates = base._load_focus_fixpoints_from_bundle(bundle, excitatory_clusters, target_rep)
    plt = base._prepare_matplotlib()
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
        for idx, (focus_count, payload) in enumerate(sorted(focus_rates.items())):
            stable_values = payload.get("stable", [])
            unstable_values = payload.get("unstable", [])
            if not stable_values and not unstable_values:
                continue
            y_level = marker_base + idx * marker_step
            color = colors[idx % len(colors)]
            if stable_values:
                ax.scatter(
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
                ax.scatter(
                    unstable_values,
                    np.full(len(unstable_values), y_level),
                    marker="v",
                    s=36,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.0,
                    label=f"Focus count {focus_count} (unstable)",
                )
    ax.set_xlabel("Mean firing rate")
    ax.set_ylabel("Bin frequency")
    ax.set_title("Distribution of maximum excitatory rates (fixpoint initialization)")
    ax.set_xlim(0.0, 1.0)
    if focus_rates:
        ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(analysis_dir, "max_rates_histogram.png")
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
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
        "mode": "fixpoint_initialization",
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
    }
    write_yaml_config(summary, os.path.join(analysis_dir, "analysis_summary.yaml"))
    print(f"Stored pooled maxima at {pooled_path}")
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
