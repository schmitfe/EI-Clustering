#!/usr/bin/env python3
"""Visualize one SimulationCollection npz file from plot_figure3_dwell_times.py."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw weights, population-level weight statistics, inferred state emissions, "
            "state occupancy, and dwell-time distribution for one SimulationCollection file."
        )
    )
    parser.add_argument("collection_file", help="Path to one output_dir/SimulationCollection/*.npz file.")
    parser.add_argument("--output", help="Output figure path. Defaults to '<collection>_summary.png'.")
    parser.add_argument(
        "--weight-display-size",
        type=int,
        default=900,
        help="Maximum side length for the downsampled raw weight matrix display.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _scalar(value: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _state_sort_key(label: str) -> tuple[int, tuple[int, ...], str]:
    text = str(label)
    if text.upper() in {"LOW", "SILENT", "QUIET", "BACKGROUND"}:
        return (0, (), text)
    numbers = tuple(sorted(int(value) for value in re.findall(r"E(\d+)", text)))
    if not numbers:
        numbers = tuple(sorted(int(value) for value in re.findall(r"\d+", text)))
    return (len(numbers) if numbers else 99, numbers, text)


def _ordered_state_indices(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    keys = [str(value) for value in np.asarray(payload.get("state_keys", np.zeros(0, dtype=str))).tolist()]
    n_states = int(np.asarray(payload.get("n_states", len(keys))).item()) if "n_states" in payload else len(keys)
    if not keys:
        keys = [str(idx) for idx in range(n_states)]
    order = np.array(sorted(range(len(keys)), key=lambda idx: _state_sort_key(keys[idx])), dtype=np.int64)
    return order, [keys[idx] for idx in order]


def _state_occupancy(payload: dict[str, np.ndarray], n_states: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(payload.get("state_segments_state", np.zeros(0)), dtype=np.int64)
    durations = np.asarray(payload.get("state_segments_duration_s", np.zeros(0)), dtype=float)
    total = float(np.sum(durations))
    time_by_state = np.zeros(n_states, dtype=float)
    for state, duration in zip(states, durations):
        if 0 <= int(state) < n_states:
            time_by_state[int(state)] += float(duration)
    fraction = time_by_state / total if total > 0.0 else np.zeros_like(time_by_state)
    return time_by_state, fraction


def _block_edges(length: int, bins: int) -> np.ndarray:
    return np.linspace(0, int(length), int(bins) + 1, dtype=np.int64)


def _dense_block_mean(matrix: np.ndarray, target_size: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    rows, cols = arr.shape
    out_rows = min(rows, int(target_size))
    out_cols = min(cols, int(target_size))
    row_edges = _block_edges(rows, out_rows)
    col_edges = _block_edges(cols, out_cols)
    out = np.zeros((out_rows, out_cols), dtype=np.float32)
    for row_idx in range(out_rows):
        row_start, row_stop = int(row_edges[row_idx]), int(row_edges[row_idx + 1])
        chunk = arr[row_start:row_stop]
        for col_idx in range(out_cols):
            col_start, col_stop = int(col_edges[col_idx]), int(col_edges[col_idx + 1])
            block = chunk[:, col_start:col_stop]
            out[row_idx, col_idx] = float(np.mean(block)) if block.size else 0.0
    return out


def _csr_block_mean(payload: dict[str, np.ndarray], target_size: int) -> np.ndarray:
    shape = np.asarray(payload["weight_weight_shape"], dtype=np.int64)
    rows, cols = int(shape[0]), int(shape[1])
    out_rows = min(rows, int(target_size))
    out_cols = min(cols, int(target_size))
    row_edges = _block_edges(rows, out_rows)
    col_edges = _block_edges(cols, out_cols)
    data = np.asarray(payload["weight_weights_data"], dtype=float)
    indices = np.asarray(payload["weight_weights_indices"], dtype=np.int64)
    indptr = np.asarray(payload["weight_weights_indptr"], dtype=np.int64)
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    for row_bin in range(out_rows):
        row_start, row_stop = int(row_edges[row_bin]), int(row_edges[row_bin + 1])
        for row in range(row_start, row_stop):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop <= start:
                continue
            col_bins = np.searchsorted(col_edges, indices[start:stop], side="right") - 1
            valid = (col_bins >= 0) & (col_bins < out_cols)
            np.add.at(out[row_bin], col_bins[valid], data[start:stop][valid])
    row_widths = np.diff(row_edges).astype(float)
    col_widths = np.diff(col_edges).astype(float)
    denom = row_widths[:, None] * col_widths[None, :]
    return np.divide(out, denom, out=np.zeros_like(out), where=denom > 0).astype(np.float32)


def _weight_display_matrix(payload: dict[str, np.ndarray], target_size: int) -> np.ndarray | None:
    if "weight_weights" in payload:
        return _dense_block_mean(np.asarray(payload["weight_weights"], dtype=float), target_size)
    required = {"weight_weights_data", "weight_weights_indices", "weight_weights_indptr", "weight_weight_shape"}
    if required.issubset(payload):
        return _csr_block_mean(payload, target_size)
    return None


def _imshow_signed(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    xlabel: str = "Pre",
    ylabel: str = "Post",
    signed: bool = True,
) -> None:
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if signed:
        vmax = float(np.nanquantile(np.abs(finite), 0.995)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        image = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    else:
        vmax = float(np.nanquantile(finite, 0.995)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        image = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return image


def _plot_population_matrix(
    ax: plt.Axes,
    payload: dict[str, np.ndarray],
    key: str,
    title: str,
    labels: list[str],
    *,
    signed: bool,
) -> None:
    matrix = np.asarray(payload[key], dtype=float)
    image = _imshow_signed(ax, matrix, title, signed=signed)
    if len(labels) <= 50:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def _plot_state_emissions(
    fig: plt.Figure,
    outer_spec: Any,
    payload: dict[str, np.ndarray],
) -> None:
    order, ordered_keys = _ordered_state_indices(payload)
    emissions = np.asarray(payload.get("state_emissions_full", np.zeros((0, 0))), dtype=float)
    if emissions.size and order.size:
        emissions = emissions[order]
    cluster_names = [str(value) for value in np.asarray(payload.get("full_cluster_names", [])).tolist()]
    time_by_state, fraction = _state_occupancy(payload, len(order))
    if order.size:
        time_by_state = time_by_state[order]
        fraction = fraction[order]

    sub = outer_spec.subgridspec(1, 2, width_ratios=[5.0, 1.2], wspace=0.08)
    ax_emit = fig.add_subplot(sub[0, 0])
    ax_occ = fig.add_subplot(sub[0, 1], sharey=ax_emit)
    if emissions.size:
        vmax = float(np.nanquantile(emissions, 0.99))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = None
        image = ax_emit.imshow(emissions, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        plt.colorbar(image, ax=ax_emit, fraction=0.025, pad=0.01)
    ax_emit.set_title("State Emissions")
    ax_emit.set_xlabel("Population")
    ax_emit.set_ylabel("State")
    ax_emit.set_yticks(np.arange(len(ordered_keys)))
    ax_emit.set_yticklabels(ordered_keys, fontsize=7)
    if len(cluster_names) <= 50:
        ax_emit.set_xticks(np.arange(len(cluster_names)))
        ax_emit.set_xticklabels(cluster_names, rotation=90, fontsize=5)

    y = np.arange(len(ordered_keys))
    ax_occ.barh(y, fraction, color="#4C78A8")
    ax_occ.set_title("Occupancy")
    ax_occ.set_xlabel("Fraction")
    ax_occ.set_xlim(0.0, max(0.05, float(np.max(fraction)) * 1.1 if fraction.size else 1.0))
    ax_occ.tick_params(axis="y", labelleft=False)
    for yy, frac, seconds in zip(y, fraction, time_by_state):
        ax_occ.text(frac, yy, f" {frac:.2f}\n {seconds:.1f}s", va="center", fontsize=6)


def _plot_dwell_distribution(ax: plt.Axes, payload: dict[str, np.ndarray]) -> None:
    dwell = np.asarray(payload.get("state_segments_duration_s", np.zeros(0)), dtype=float)
    dwell = dwell[np.isfinite(dwell) & (dwell > 0.0)]
    if dwell.size:
        bins = min(50, max(5, int(np.sqrt(dwell.size))))
        ax.hist(dwell, bins=bins, color="#444444", alpha=0.85)
        ax.axvline(float(np.mean(dwell)), color="#D62728", linewidth=1.5, label=f"mean={np.mean(dwell):.2f}s")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Dwell-Time Distribution")
    ax.set_xlabel("Dwell time [s]")
    ax.set_ylabel("Episode count")


def _sorted_state_emissions(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    order, ordered_keys = _ordered_state_indices(payload)
    emissions = np.asarray(payload.get("state_emissions_full", np.zeros((0, 0))), dtype=float)
    if emissions.size and order.size:
        emissions = emissions[order]
    return emissions, np.asarray(ordered_keys, dtype=str)


def _compact_parameter_dict(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    params = json.loads(str(_scalar(payload.get("simulation_parameter_json", np.array("{}")))))
    binary_cfg = json.loads(str(_scalar(payload.get("binary_config_json", np.array("{}")))))
    keys = (
        "seed",
        "condition",
        "kappa",
        "R_j",
        "R_Eplus",
        "connection_type",
        "N_E",
        "N_I",
        "Q",
        "focus_count",
        "p0_ee",
        "p0_ie",
        "p0_ei",
        "p0_ii",
        "tau_e",
        "tau_i",
        "sample_interval",
        "simulation_steps",
        "warmup_steps",
        "batch_size",
    )
    merged: dict[str, Any] = {
        "seed": int(_scalar(payload.get("seed", np.array(-1)))),
        "condition": str(_scalar(payload.get("condition", np.array("")))),
    }
    merged.update({key: params.get(key) for key in keys if key in params})
    for key in ("sample_interval", "simulation_steps", "warmup_steps", "batch_size"):
        if key in binary_cfg:
            merged[key] = binary_cfg[key]
    merged["seed"] = int(_scalar(payload.get("seed", np.array(-1))))
    return merged


def plot_collection(collection_file: Path, output: Path, *, weight_display_size: int, dpi: int) -> None:
    payload = _load_npz(collection_file)
    condition = str(_scalar(payload.get("condition", np.array("?"))))
    seed = int(_scalar(payload.get("seed", np.array(-1))))
    summary = json.loads(str(_scalar(payload.get("network_summary_json", np.array("{}")))))
    title = f"{collection_file.name} | condition={condition}, seed={seed}, states={summary.get('n_states', payload.get('n_states', '?'))}"

    raw_weight = _weight_display_matrix(payload, int(weight_display_size))
    pop_labels = [str(value) for value in np.asarray(payload.get("matrix_analysis_population_names", [])).tolist()]
    matrix_panels = [
        ("matrix_analysis_mean_weight", "Mean Synaptic Weight", True),
        ("matrix_analysis_var_weight", "Variance Synaptic Weight", False),
        ("matrix_analysis_mean_indegree", "Mean Indegree", False),
        ("matrix_analysis_var_indegree", "Variance Indegree", False),
    ]
    missing = [key for key, _title, _signed in matrix_panels if key not in payload]
    if missing:
        raise KeyError(f"{collection_file} is missing {missing}. Available keys include: {sorted(payload)[:20]} ...")

    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.15])
    fig.suptitle(title, fontsize=12)

    ax_raw = fig.add_subplot(grid[0:2, 0])
    if raw_weight is not None:
        image = _imshow_signed(
            ax_raw,
            raw_weight,
            f"Raw Weight Matrix\nblock mean display {raw_weight.shape[0]}x{raw_weight.shape[1]}",
            xlabel="Pre neuron bin",
            ylabel="Post neuron bin",
            signed=True,
        )
        plt.colorbar(image, ax=ax_raw, fraction=0.046, pad=0.02)
    else:
        ax_raw.text(0.5, 0.5, "No embedded weight matrix\n(use collection with weights)", ha="center", va="center")
        ax_raw.set_axis_off()

    matrix_axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    for ax, (key, panel_title, signed) in zip(matrix_axes, matrix_panels):
        _plot_population_matrix(ax, payload, key, panel_title, pop_labels, signed=signed)

    ax_dwell = fig.add_subplot(grid[0:2, 3])
    _plot_dwell_distribution(ax_dwell, payload)
    _plot_state_emissions(fig, grid[2, :], payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)

    sorted_emissions, sorted_labels = _sorted_state_emissions(payload)
    order, _ordered_keys = _ordered_state_indices(payload)
    _time_by_state, occupancy_fraction = _state_occupancy(payload, len(order))
    if order.size:
        occupancy_fraction = occupancy_fraction[order]
    parameter_json = json.dumps(_compact_parameter_dict(payload), sort_keys=True)
    emissions_path = output.with_name(f"{output.stem}_sorted_state_emissions.npz")
    np.savez_compressed(
        emissions_path,
        state_emissions=sorted_emissions,
        state_labels=sorted_labels,
        state_occupancy=occupancy_fraction,
        parameters_json=np.array(parameter_json),
        mean_indegree=np.asarray(payload["matrix_analysis_mean_indegree"], dtype=float),
        var_indegree=np.asarray(payload["matrix_analysis_var_indegree"], dtype=float),
        mean_weight=np.asarray(payload["matrix_analysis_mean_weight"], dtype=float),
    )


def main() -> None:
    args = parse_args()
    collection_file = Path(args.collection_file).expanduser()
    if not collection_file.exists():
        raise FileNotFoundError(collection_file)
    output = Path(args.output) if args.output else collection_file.with_name(f"{collection_file.stem}_summary.png")
    plot_collection(
        collection_file,
        output,
        weight_display_size=int(args.weight_display_size),
        dpi=int(args.dpi),
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
