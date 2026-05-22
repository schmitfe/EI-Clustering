#!/usr/bin/env python3
from __future__ import annotations

"""Population-level analysis for exported binary-network weight matrices.

The input is the `*_weights.npz` file written by `pipelines/binary.py`.

For each post-population / pre-population block, the script computes:

- `mean_indegree`, `std_indegree`
  Statistics of the number of realized incoming synapses per post neuron.
- `mean_weight`, `std_weight`
  Statistics of the realized non-zero synaptic weights inside the block.

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
            "For each post-population / pre-population block, compute mean/std indegree "
            "across post neurons and mean/std synaptic weight across realized synapses."
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


def _analyze_dense_blocks(
    weights: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> Dict[str, np.ndarray]:
    pop_count = starts.size
    mean_indegree = np.zeros((pop_count, pop_count), dtype=float)
    std_indegree = np.zeros((pop_count, pop_count), dtype=float)
    mean_weight = np.zeros((pop_count, pop_count), dtype=float)
    std_weight = np.zeros((pop_count, pop_count), dtype=float)

    for post_idx, (post_start, post_end) in enumerate(zip(starts, ends)):
        for pre_idx, (pre_start, pre_end) in enumerate(zip(starts, ends)):
            block = weights[post_start:post_end, pre_start:pre_end]
            if block.size == 0:
                continue
            mask = block != 0
            indegrees = mask.sum(axis=1, dtype=np.int64)
            mean_indegree[post_idx, pre_idx] = float(indegrees.mean())
            std_indegree[post_idx, pre_idx] = float(indegrees.std())
            block_weights = block[mask]
            if block_weights.size:
                mean_weight[post_idx, pre_idx] = float(block_weights.mean())
                std_weight[post_idx, pre_idx] = float(block_weights.std())

    return {
        "mean_indegree": mean_indegree,
        "std_indegree": std_indegree,
        "mean_weight": mean_weight,
        "std_weight": std_weight,
    }


def _analyze_csr_blocks(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> Dict[str, np.ndarray]:
    pop_count = starts.size
    mean_indegree = np.zeros((pop_count, pop_count), dtype=float)
    std_indegree = np.zeros((pop_count, pop_count), dtype=float)
    mean_weight = np.zeros((pop_count, pop_count), dtype=float)
    std_weight = np.zeros((pop_count, pop_count), dtype=float)

    for post_idx, (post_start, post_end) in enumerate(zip(starts, ends)):
        row_count = int(post_end - post_start)
        if row_count <= 0:
            continue
        for pre_idx, (pre_start, pre_end) in enumerate(zip(starts, ends)):
            indegrees = np.zeros(row_count, dtype=np.int64)
            nonzero_weights = []
            for local_row, row in enumerate(range(int(post_start), int(post_end))):
                row_start = int(indptr[row])
                row_end = int(indptr[row + 1])
                row_indices = indices[row_start:row_end]
                row_data = data[row_start:row_end]
                # CSR stores all incoming weights for one post neuron in a row.
                # We therefore filter each row to the current pre-population
                # interval and accumulate per-neuron indegrees plus block weights.
                mask = (row_indices >= pre_start) & (row_indices < pre_end)
                indegrees[local_row] = int(np.count_nonzero(mask))
                if np.any(mask):
                    nonzero_weights.append(row_data[mask])
            mean_indegree[post_idx, pre_idx] = float(indegrees.mean())
            std_indegree[post_idx, pre_idx] = float(indegrees.std())
            if nonzero_weights:
                weights_block = np.concatenate(nonzero_weights)
                mean_weight[post_idx, pre_idx] = float(weights_block.mean())
                std_weight[post_idx, pre_idx] = float(weights_block.std())

    return {
        "mean_indegree": mean_indegree,
        "std_indegree": std_indegree,
        "mean_weight": mean_weight,
        "std_weight": std_weight,
    }


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
    print("  mean_indegree/std_indegree: statistics of non-zero incoming connection counts per post neuron.")
    print("  mean_weight/std_weight: statistics of realized non-zero synaptic weights in each population block.")

    if args.print_matrices:
        _print_matrix("mean_indegree", np.asarray(result["mean_indegree"], dtype=float), labels)
        _print_matrix("std_indegree", np.asarray(result["std_indegree"], dtype=float), labels)
        _print_matrix("mean_weight", np.asarray(result["mean_weight"], dtype=float), labels)
        _print_matrix("std_weight", np.asarray(result["std_weight"], dtype=float), labels)


if __name__ == "__main__":
    main()
