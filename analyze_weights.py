#!/usr/bin/env python3
from __future__ import annotations

"""Population-level analysis for exported binary-network weight matrices.

The input is the `*_weights.npz` file written by `pipelines/binary.py`.

For each post-population / pre-population block, the script computes:

- `mean_indegree`, `std_indegree`, `var_indegree`
  Statistics of the realized incoming synapse multiplicity per post neuron.
- `mean_weight`, `std_weight`, `var_weight`
  Statistics of individual realized synaptic weight quanta inside the block.
- `mean_entry_weight`, `std_entry_weight`, `var_entry_weight`
  Statistics of the accumulated weight-matrix entries, including zeros.

Rows in the output matrices correspond to post-synaptic populations and
columns correspond to pre-synaptic populations.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a dumped binary-network weight matrix on the population level. "
            "For each post-population / pre-population block, compute mean/std/var "
            "of incoming synapse multiplicity across post neurons and weight "
            "statistics across realized synapses and matrix entries."
        )
    )
    parser.add_argument(
        "weights_source",
        type=str,
        help=(
            "Either a path to the exported *_weights.npz file produced by "
            "pipelines/binary.py, or a directory containing exactly one such file."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output path for the analysis .npz. Defaults to '<weights>_analysis.npz'.",
    )
    parser.add_argument(
        "--print-matrices",
        action="store_true",
        help="Print the resulting matrices to stdout in addition to saving them.",
    )
    return parser.parse_args()


def _resolve_weights_path(source: str) -> str:
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        matches = sorted(candidate.glob("*_weights.npz"))
        if not matches:
            raise FileNotFoundError(
                f"No '*_weights.npz' file found in directory {candidate}."
            )
        if len(matches) > 1:
            match_list = ", ".join(path.name for path in matches)
            raise ValueError(
                f"Directory {candidate} contains multiple '*_weights.npz' files: {match_list}. "
                "Pass the desired file path explicitly."
            )
        return str(matches[0].resolve())
    if not candidate.exists():
        raise FileNotFoundError(f"Weight source {candidate} does not exist.")
    return str(candidate.resolve())


def _require_keys(payload: Dict[str, Any], keys: Sequence[str], path: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing required field(s): {joined}.")


def _load_npz_payload(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _as_text_list(values: np.ndarray) -> list[str]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(item) for item in arr.tolist()]


def _load_dense_from_legacy_object(payload: Dict[str, Any]) -> np.ndarray | None:
    raw = payload.get("weights")
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.dtype != object:
        return np.asarray(arr, dtype=float)
    if arr.ndim != 0:
        return None
    obj = arr.item()
    if hasattr(obj, "toarray"):
        return np.asarray(obj.toarray(), dtype=float)
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float)
    return None


def _load_weight_representation(path: str) -> Tuple[str, np.ndarray | None, Dict[str, np.ndarray]]:
    payload = _load_npz_payload(path)
    _require_keys(
        payload,
        (
            "population_names",
            "population_start_ids",
            "population_end_ids",
        ),
        path,
    )

    weight_format = str(np.asarray(payload.get("weight_format", "")).item() if "weight_format" in payload else "").lower()
    if weight_format == "dense":
        _require_keys(payload, ("weights",), path)
        dense = np.asarray(payload["weights"], dtype=float)
        return "dense", dense, payload
    if weight_format == "csr":
        _require_keys(payload, ("weights_data", "weights_indices", "weights_indptr", "weight_shape"), path)
        return "csr", None, payload

    # Backwards-compatible fallback for older dumps that stored `network.weights`
    # directly. Dense arrays are handled without special cases, sparse objects
    # are converted to dense because the legacy schema does not expose CSR fields.
    legacy_dense = _load_dense_from_legacy_object(payload)
    if legacy_dense is not None:
        return "dense", legacy_dense, payload
    raise ValueError(
        f"{path} uses an unsupported weight format. "
        "Expected dense/CSR export from pipelines/binary.py."
    )


def _population_bounds(payload: Dict[str, np.ndarray]) -> Tuple[list[str], np.ndarray, np.ndarray]:
    names = _as_text_list(payload["population_names"])
    starts = np.asarray(payload["population_start_ids"], dtype=np.int64).ravel()
    ends = np.asarray(payload["population_end_ids"], dtype=np.int64).ravel()
    if starts.size != ends.size or starts.size != len(names):
        raise ValueError("Population metadata has inconsistent lengths.")
    if np.any(ends < starts):
        raise ValueError("Population metadata contains invalid start/end indices.")
    return names, starts, ends


def _infer_weight_quantum(nonzero_weights: np.ndarray) -> float:
    """Infer the per-synapse quantum from accumulated matrix entries."""
    values = np.asarray(nonzero_weights, dtype=float).ravel()
    values = values[values != 0.0]
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0):
        return float(values[values < 0.0].max())
    return float(values[values > 0.0].min())


def _weighted_mean_var(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    total = float(weights.sum())
    if values.size == 0 or total <= 0.0:
        return 0.0, 0.0
    mean = float(np.sum(values * weights) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, var


def _empty_stat_matrices(pop_count: int) -> Dict[str, np.ndarray]:
    names = (
        "mean_indegree",
        "std_indegree",
        "var_indegree",
        "mean_occupied_indegree",
        "std_occupied_indegree",
        "var_occupied_indegree",
        "mean_weight",
        "std_weight",
        "var_weight",
        "mean_entry_weight",
        "std_entry_weight",
        "var_entry_weight",
        "mean_nonzero_entry_weight",
        "std_nonzero_entry_weight",
        "var_nonzero_entry_weight",
        "weight_quantum",
    )
    return {name: np.zeros((pop_count, pop_count), dtype=float) for name in names}


def _analyze_block(block: np.ndarray) -> Dict[str, float]:
    if block.size == 0:
        return {name: 0.0 for name in _empty_stat_matrices(1)}

    block = np.asarray(block, dtype=float)
    mask = block != 0.0
    occupied_indegrees = mask.sum(axis=1, dtype=np.int64)
    nonzero_weights = block[mask]
    quantum = _infer_weight_quantum(nonzero_weights)

    if quantum != 0.0:
        multiplicities = np.rint(block / quantum).astype(np.int64, copy=False)
        multiplicities = np.maximum(multiplicities, 0)
    else:
        multiplicities = np.zeros(block.shape, dtype=np.int64)
    indegrees = multiplicities.sum(axis=1, dtype=np.int64)

    entry_var = float(block.var())
    occupied_var = float(occupied_indegrees.var())
    indegree_var = float(indegrees.var())

    nonzero_entry_mean = 0.0
    nonzero_entry_var = 0.0
    weight_mean = 0.0
    weight_var = 0.0
    if nonzero_weights.size:
        nonzero_entry_mean = float(nonzero_weights.mean())
        nonzero_entry_var = float(nonzero_weights.var())
        nonzero_counts = multiplicities[mask]
        synaptic_weights = nonzero_weights / nonzero_counts
        weight_mean, weight_var = _weighted_mean_var(synaptic_weights, nonzero_counts)

    return {
        "mean_indegree": float(indegrees.mean()),
        "std_indegree": float(np.sqrt(indegree_var)),
        "var_indegree": indegree_var,
        "mean_occupied_indegree": float(occupied_indegrees.mean()),
        "std_occupied_indegree": float(np.sqrt(occupied_var)),
        "var_occupied_indegree": occupied_var,
        "mean_weight": weight_mean,
        "std_weight": float(np.sqrt(weight_var)),
        "var_weight": weight_var,
        "mean_entry_weight": float(block.mean()),
        "std_entry_weight": float(np.sqrt(entry_var)),
        "var_entry_weight": entry_var,
        "mean_nonzero_entry_weight": nonzero_entry_mean,
        "std_nonzero_entry_weight": float(np.sqrt(nonzero_entry_var)),
        "var_nonzero_entry_weight": nonzero_entry_var,
        "weight_quantum": quantum,
    }


def _analyze_dense_blocks(
    weights: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> Dict[str, np.ndarray]:
    pop_count = starts.size
    stats = _empty_stat_matrices(pop_count)

    for post_idx, (post_start, post_end) in enumerate(zip(starts, ends)):
        for pre_idx, (pre_start, pre_end) in enumerate(zip(starts, ends)):
            block = weights[post_start:post_end, pre_start:pre_end]
            block_stats = _analyze_block(block)
            for name, value in block_stats.items():
                stats[name][post_idx, pre_idx] = value

    return stats


def _analyze_csr_blocks(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> Dict[str, np.ndarray]:
    pop_count = starts.size
    stats = _empty_stat_matrices(pop_count)

    for post_idx, (post_start, post_end) in enumerate(zip(starts, ends)):
        row_count = int(post_end - post_start)
        if row_count <= 0:
            continue
        for pre_idx, (pre_start, pre_end) in enumerate(zip(starts, ends)):
            col_count = int(pre_end - pre_start)
            if col_count <= 0:
                continue
            block = np.zeros((row_count, col_count), dtype=float)
            for local_row, row in enumerate(range(int(post_start), int(post_end))):
                row_start = int(indptr[row])
                row_end = int(indptr[row + 1])
                row_indices = indices[row_start:row_end]
                row_data = data[row_start:row_end]
                # CSR stores all incoming weights for one post neuron in a row.
                # We therefore filter each row to the current pre-population
                # interval and accumulate per-neuron indegrees plus block weights.
                mask = (row_indices >= pre_start) & (row_indices < pre_end)
                if np.any(mask):
                    block[local_row, row_indices[mask] - pre_start] = row_data[mask]
            block_stats = _analyze_block(block)
            for name, value in block_stats.items():
                stats[name][post_idx, pre_idx] = value

    return stats


def analyze_weights(path: str) -> Dict[str, Any]:
    weight_format, dense_weights, payload = _load_weight_representation(path)
    names, starts, ends = _population_bounds(payload)

    if weight_format == "dense":
        if dense_weights is None:
            raise RuntimeError("Dense weight analysis requires a dense matrix.")
        stats = _analyze_dense_blocks(dense_weights, starts, ends)
    else:
        stats = _analyze_csr_blocks(
            data=np.asarray(payload["weights_data"], dtype=float),
            indices=np.asarray(payload["weights_indices"], dtype=np.int64),
            indptr=np.asarray(payload["weights_indptr"], dtype=np.int64),
            starts=starts,
            ends=ends,
        )

    result: Dict[str, Any] = {
        "source_weights_path": np.array(os.path.abspath(path)),
        "population_names": np.array(names),
        "population_start_ids": starts,
        "population_end_ids": ends,
        "weight_format": np.array(weight_format),
    }
    if "weight_mode" in payload:
        result["weight_mode"] = np.asarray(payload["weight_mode"])
    if "weight_shape" in payload:
        result["weight_shape"] = np.asarray(payload["weight_shape"], dtype=np.int64)
    if "population_sizes" in payload:
        result["population_sizes"] = np.asarray(payload["population_sizes"], dtype=np.int64)
    if "population_cell_types" in payload:
        result["population_cell_types"] = np.asarray(payload["population_cell_types"])
    if "population_cluster_indices" in payload:
        result["population_cluster_indices"] = np.asarray(payload["population_cluster_indices"], dtype=np.int64)
    result.update(stats)
    return result


def _default_output_path(weights_path: str) -> str:
    input_path = Path(weights_path)
    if input_path.suffix == ".npz":
        return str(input_path.with_name(f"{input_path.stem}_analysis.npz"))
    return str(input_path.with_name(f"{input_path.name}_analysis.npz"))


def _print_matrix(name: str, matrix: np.ndarray, labels: Sequence[str]) -> None:
    print(f"{name} (rows=post, cols=pre)")
    print("labels:", ", ".join(labels))
    print(np.array2string(matrix, precision=6, suppress_small=False))
    print()


def main() -> None:
    args = parse_args()
    weights_path = _resolve_weights_path(args.weights_source)
    result = analyze_weights(weights_path)
    output_path = args.output or _default_output_path(weights_path)
    np.savez_compressed(output_path, **result)

    labels = _as_text_list(np.asarray(result["population_names"]))
    print(f"Resolved weights file: {weights_path}")
    print(f"Stored weight analysis at {os.path.abspath(output_path)}")
    print(f"Population count: {len(labels)}")
    print("Matrix convention: rows are post-synaptic populations, columns are pre-synaptic populations.")
    print("Measures:")
    print("  mean_indegree/std_indegree/var_indegree: incoming synapse multiplicity per post neuron.")
    print("  mean_occupied_indegree/std_occupied_indegree/var_occupied_indegree: non-zero matrix entries per post neuron.")
    print("  mean_weight/std_weight/var_weight: individual realized synaptic weight quanta.")
    print("  mean_entry_weight/std_entry_weight/var_entry_weight: accumulated matrix entries, including zeros.")
    print("  mean_nonzero_entry_weight/std_nonzero_entry_weight/var_nonzero_entry_weight: accumulated non-zero matrix entries.")

    if args.print_matrices:
        for name in (
            "mean_indegree",
            "std_indegree",
            "var_indegree",
            "mean_occupied_indegree",
            "std_occupied_indegree",
            "var_occupied_indegree",
            "mean_weight",
            "std_weight",
            "var_weight",
            "mean_entry_weight",
            "std_entry_weight",
            "var_entry_weight",
            "mean_nonzero_entry_weight",
            "std_nonzero_entry_weight",
            "var_nonzero_entry_weight",
            "weight_quantum",
        ):
            _print_matrix(name, np.asarray(result[name], dtype=float), labels)


if __name__ == "__main__":
    main()
