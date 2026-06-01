from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_trace(trace_path: str | Path) -> tuple[np.ndarray, list[str]]:
    with np.load(trace_path, allow_pickle=True) as payload:
        rates = np.asarray(payload["rates"], dtype=float)
        names = [str(name) for name in np.asarray(payload["names"], dtype=object).tolist()]
    return rates, names


def _e_population_matrix(trace_path: str | Path) -> np.ndarray:
    rates, names = _load_trace(trace_path)
    e_indices = [idx for idx, name in enumerate(names) if name.startswith("E")]
    if not e_indices:
        midpoint = rates.shape[1] // 2
        e_indices = list(range(midpoint))
    return rates[:, e_indices].T


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in ("R_j", "kappa", "seed", "simulation") if col in summary.columns]
    return summary.sort_values(columns).reset_index(drop=True) if columns else summary.reset_index(drop=True)


def _parse_breakpoints(value: object, n_timepoints: int) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [int(n_timepoints)]
    text = str(value).strip()
    if not text:
        return [int(n_timepoints)]
    points = []
    for item in text.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        points.append(int(float(item)))
    if not points or points[-1] != int(n_timepoints):
        points.append(int(n_timepoints))
    return [max(0, min(int(n_timepoints), point)) for point in points]


def _segmentation_labels(breakpoints: list[int], n_timepoints: int) -> np.ndarray:
    labels = np.zeros(int(n_timepoints), dtype=int)
    start = 0
    for idx, stop in enumerate(breakpoints):
        stop = max(start, min(int(n_timepoints), int(stop)))
        labels[start:stop] = idx
        start = stop
    return labels


def plot_e_population_summary(
    summary_paths: Sequence[str | Path],
    output_prefix: str | Path,
    *,
    columns: int = 5,
    dpi: int = 180,
    overlay_segmentations: bool = False,
) -> Path:
    frames = [pd.read_csv(path) for path in summary_paths]
    summary = _sort_summary(pd.concat(frames, ignore_index=True))
    if summary.empty:
        raise ValueError("No simulations found in the provided summary files.")
    required = {"simulation", "trace_path"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")

    matrices = [_e_population_matrix(path) for path in summary["trace_path"]]
    finite_values = np.concatenate([matrix[np.isfinite(matrix)].ravel() for matrix in matrices if matrix.size])
    vmax = float(np.percentile(finite_values, 99.0)) if finite_values.size else 1.0
    vmax = max(vmax, 1e-6)
    n_items = len(matrices)
    n_cols = max(1, int(columns))
    n_rows = int(np.ceil(n_items / n_cols))
    if overlay_segmentations:
        fig = plt.figure(figsize=(3.2 * n_cols, 2.55 * n_rows))
        outer = fig.add_gridspec(n_rows, n_cols)
        panel_axes = []
        seg_axes = []
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                inner = outer[row_idx, col_idx].subgridspec(2, 1, height_ratios=[0.35, 2.0], hspace=0.05)
                seg_axes.append(fig.add_subplot(inner[0, 0]))
                panel_axes.append(fig.add_subplot(inner[1, 0]))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.25 * n_rows), squeeze=False)
        panel_axes = list(axes.ravel())
        seg_axes = [None] * len(panel_axes)
    for idx, (ax, seg_ax, matrix) in enumerate(zip(panel_axes, seg_axes, matrices)):
        row = summary.iloc[idx]
        n_timepoints = matrix.shape[1]
        if overlay_segmentations and seg_ax is not None:
            if "breakpoints" in summary.columns:
                breakpoints = _parse_breakpoints(row["breakpoints"], n_timepoints)
            else:
                breakpoints = [n_timepoints]
            labels = _segmentation_labels(breakpoints, n_timepoints)
            seg_ax.imshow(labels[None, :], aspect="auto", interpolation="nearest", cmap="tab20")
            start = 0
            for point in breakpoints[:-1]:
                seg_ax.axvline(point - 0.5, color="white", linewidth=0.8, alpha=0.9)
                start = point
            seg_ax.set_yticks([])
            seg_ax.set_xticks([])
            seg_ax.set_xlim(-0.5, n_timepoints - 0.5)
        ax.imshow(matrix, aspect="auto", interpolation="nearest", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        title = str(row["simulation"])
        if {"kappa", "R_j", "seed"} <= set(summary.columns):
            title = f"k={float(row['kappa']):g}, Rj={float(row['R_j']):g}, seed={int(row['seed'])}"
        if "algorithm_cp" in summary.columns:
            title += f", CP={int(row['algorithm_cp'])}"
        elif "n_segments" in summary.columns:
            title += f", CP={max(0, int(row['n_segments']) - 1)}"
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("E pop", fontsize=7)
        ax.set_xlabel("sample", fontsize=7)
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.text(
            0.01,
            0.98,
            f"{idx:02d}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
        )
    for ax in panel_axes[n_items:]:
        ax.set_axis_off()
    for ax in seg_axes[n_items:]:
        if ax is not None:
            ax.set_axis_off()
    fig.suptitle("Full-size Figure3 simulations: excitatory population activity", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)

    index_path = output_prefix.with_name(output_prefix.name + "_index.csv")
    summary.to_csv(index_path, index=False)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a montage of E-population activity heatmaps from simulation summaries.")
    parser.add_argument("summaries", nargs="+", help="One or more simulation_summary*.csv files.")
    parser.add_argument("--output-prefix", required=True, help="Output path without extension.")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overlay-segmentations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = plot_e_population_summary(
        args.summaries,
        args.output_prefix,
        columns=args.columns,
        dpi=args.dpi,
        overlay_segmentations=bool(args.overlay_segmentations),
    )
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
