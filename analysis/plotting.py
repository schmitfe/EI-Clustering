from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .types import AnalysisInput, StateInferenceResult


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_cluster_activity_heatmap(
    X: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    dpi: int = 150,
    title: str = "Cluster activity",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axes[0].imshow(np.asarray(X, dtype=float).T, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    axes[0].set_ylabel("Cluster")
    axes[0].set_title(title)
    axes[1].imshow(np.asarray(labels, dtype=np.int64)[None, :], aspect="auto", interpolation="nearest", cmap="tab20")
    axes[1].set_ylabel("State")
    axes[1].set_xlabel("Time bin")
    _save(fig, path, dpi)


def plot_binary_activity_matrix(
    X_binary: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    plot_cluster_activity_heatmap(X_binary, labels, path, dpi=dpi, title="Binary active-cluster matrix")


def plot_state_sequence(labels: np.ndarray, path: Path, *, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.step(np.arange(len(labels)), labels, where="post", linewidth=1.2)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("State")
    ax.set_title("Inferred state sequence")
    _save(fig, path, dpi)


def plot_dwell_histograms(dwell_times: Mapping[int, np.ndarray], path: Path, *, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for state, values in dwell_times.items():
        arr = np.asarray(values, dtype=float)
        if arr.size:
            ax.hist(arr, bins=min(20, max(5, arr.size)), histtype="step", linewidth=1.5, label=f"State {state}")
    ax.set_xlabel("Dwell time")
    ax.set_ylabel("Count")
    ax.set_title("Dwell-time histograms")
    if dwell_times:
        ax.legend()
    _save(fig, path, dpi)


def plot_dwell_survival(dwell_times: Mapping[int, np.ndarray], path: Path, *, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for state, values in dwell_times.items():
        arr = np.sort(np.asarray(values, dtype=float))
        if arr.size == 0:
            continue
        survival = 1.0 - np.arange(1, arr.size + 1) / float(arr.size)
        ax.step(arr, survival, where="post", linewidth=1.5, label=f"State {state}")
    ax.set_xlabel("Dwell time")
    ax.set_ylabel("Survival")
    ax.set_title("Dwell-time survival functions")
    if dwell_times:
        ax.legend()
    _save(fig, path, dpi)


def plot_transition_matrix(matrix: np.ndarray, path: Path, *, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(np.asarray(matrix, dtype=float), interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("Transition matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path, dpi)


def plot_state_mean_templates(
    state_means: np.ndarray,
    path: Path,
    *,
    dpi: int = 150,
    title: str = "State mean activity",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(np.asarray(state_means, dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("State")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path, dpi)


def plot_method_comparison(
    results: Mapping[str, StateInferenceResult],
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    names = list(results.keys())
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(10, max(2.5, 1.8 * len(names))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        ax.imshow(np.asarray(results[name].labels, dtype=np.int64)[None, :], aspect="auto", interpolation="nearest", cmap="tab20")
        ax.set_ylabel(name)
    axes[-1].set_xlabel("Time bin")
    fig.suptitle("Method comparison")
    _save(fig, path, dpi)


def save_result_plots(
    result: StateInferenceResult,
    data: AnalysisInput,
    output_dir: Path,
    *,
    dpi: int = 150,
    save_format: str = "png",
) -> None:
    suffix = str(save_format).lower()
    X = data.preferred_matrix()
    plot_cluster_activity_heatmap(X, result.labels, output_dir / f"cluster_activity_heatmap.{suffix}", dpi=dpi)
    if data.X_binary is not None:
        plot_binary_activity_matrix(data.X_binary, result.labels, output_dir / f"binary_activity.{suffix}", dpi=dpi)
    plot_state_sequence(result.labels, output_dir / f"state_sequence.{suffix}", dpi=dpi)
    plot_dwell_histograms(result.dwell_times, output_dir / f"dwell_histograms.{suffix}", dpi=dpi)
    plot_dwell_survival(result.dwell_times, output_dir / f"dwell_survival.{suffix}", dpi=dpi)
    plot_transition_matrix(result.transition_matrix, output_dir / f"transition_matrix.{suffix}", dpi=dpi)
    plot_state_mean_templates(result.state_means, output_dir / f"state_means.{suffix}", dpi=dpi)
