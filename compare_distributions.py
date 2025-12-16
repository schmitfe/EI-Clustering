from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    yaml = None
    YAML_ERROR = exc
else:
    YAML_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare binary simulation firing-rate distributions with fixed-point predictions."
    )
    parser.add_argument(
        "path",
        help="Path to a data folder (e.g. data/Bernoulli/Rj00_60/<tag>/) or an all_fixpoints_*.pkl file.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count (default: %(default)s).")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=50,
        help="Samples per time bin when averaging firing rates (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for histograms (defaults to <folder>/binary).",
    )
    return parser.parse_args()


def _load_yaml(path: str) -> Dict:
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to read configuration files. Install it via 'pip install pyyaml'."
        ) from YAML_ERROR
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_all_fixpoints(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".pkl")


def _compute_excitatory_bins(rates: np.ndarray, names: Sequence[str], bin_samples: int) -> tuple[List[float], int]:
    excit_indices = [idx for idx, name in enumerate(names) if str(name).startswith("E")]
    if not excit_indices:
        return [], 0
    excit_rates = rates[:, excit_indices]
    bin_samples = max(1, int(bin_samples))
    sample_count = excit_rates.shape[0]
    binned: List[float] = []
    start = 0
    while start < sample_count:
        end = min(sample_count, start + bin_samples)
        chunk = excit_rates[start:end]
        avg = chunk.mean(axis=0)
        binned.extend(float(value) for value in avg)
        start = end
    return binned, len(excit_indices)


def _extract_focus_rates(rates: Sequence[float] | None, focus_count: int, excit_count: int) -> List[float]:
    if rates is None:
        return []
    focus_count = max(0, min(int(focus_count), int(excit_count)))
    if focus_count == 0:
        return []
    arr = np.asarray(rates, dtype=float).ravel()
    if arr.size < focus_count:
        focus_count = arr.size
    return [float(val) for val in arr[:focus_count]]


def _load_focus_from_yaml(path: str, excit_count: int) -> Dict[int, List[float]]:
    if not os.path.exists(path):
        return {}
    data = _load_yaml(path)
    focus_rates: Dict[int, List[float]] = defaultdict(list)
    focus_fixpoints = data.get("focus_fixpoints", {})
    for raw_focus, entry in focus_fixpoints.items():
        focus_count = int(raw_focus)
        for fixpoint in entry.get("fixpoints", []):
            rates = _extract_focus_rates(fixpoint.get("rates"), focus_count, excit_count)
            focus_rates[focus_count].extend(rates)
    return focus_rates


def main() -> None:
    args = parse_args()
    if _is_all_fixpoints(args.path):
        with open(args.path, "rb") as handle:
            fixpoint_payload = pickle.load(handle)
        metadata = fixpoint_payload.get("metadata", {})
        source_folder = metadata.get("source_folder")
        if not source_folder or not os.path.isdir(source_folder):
            raise FileNotFoundError(
                "Unable to locate the source_folder described in the fixpoint file. "
                "Pass a data folder path instead."
            )
        base_dir = source_folder
        parameter = metadata.get("analysis_parameter", {})
    else:
        base_dir = os.path.abspath(args.path)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"{base_dir} is not a directory or the path is invalid.")
        parameter = _load_yaml(os.path.join(base_dir, "params.yaml"))
    binary_dir = os.path.join(base_dir, "binary")
    npz_files = sorted(glob.glob(os.path.join(binary_dir, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"No binary *.npz traces were found in {binary_dir}.")
    output_dir = args.output_dir or binary_dir
    os.makedirs(output_dir, exist_ok=True)
    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "rates" not in data or "names" not in data:
                    print(f"Skipping {npz_path}: missing 'rates' or 'names'.")
                    continue
                rates = np.asarray(data["rates"], dtype=float)
                names = [str(name) for name in data["names"]]
        except Exception as exc:  # pragma: no cover - best-effort robustness
            print(f"Skipping {npz_path}: failed to load ({exc}).")
            continue
        if rates.ndim != 2 or rates.shape[0] == 0:
            print(f"Skipping {npz_path}: empty rates array.")
            continue
        excit_rates, excit_count = _compute_excitatory_bins(rates, names, args.bin_size)
        if not excit_rates or excit_count == 0:
            print(f"Skipping {npz_path}: no excitatory populations detected.")
            continue
        yaml_path = os.path.splitext(npz_path)[0] + "_fixpoints.yaml"
        focus_rates = _load_focus_from_yaml(yaml_path, excit_count)
        plt.figure(figsize=(8, 5))
        bins=np.linspace(0,1, max(1, args.bins), endpoint=True)
        counts, _, _ = plt.hist(
            excit_rates,
            bins=bins,
            color="#7fb0ff",
            alpha=0.7,
            label="Binary excitatory bins",
        )
        max_count = float(counts.max()) if counts.size else 1.0
        marker_base = max_count * 1.05
        marker_step = max(max_count * 0.08, 0.05)
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(focus_rates))))
        for idx, (focus_count, values) in enumerate(sorted(focus_rates.items())):
            if not values:
                continue
            y_level = marker_base + idx * marker_step
            plt.scatter(
                values,
                np.full(len(values), y_level),
                marker="v",
                s=36,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidths=0.4,
                label=f"Focus count {focus_count}",
            )
        base_name = os.path.basename(npz_path)
        plt.title(f"Firing rates for {base_name}")
        plt.xlabel("Mean firing rate")
        plt.ylabel("Bin frequency")
        plt.xlim(0,1)
        plt.legend()
        plt.tight_layout()
        output_name = os.path.splitext(base_name)[0] + "_rate_hist.png"
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved histogram for {base_name} to {output_path}")


if __name__ == "__main__":
    main()
