from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None
else:  # pragma: no cover - optional dependency
    yaml.Loader  # hint for linters


STATE_COLUMNS = (
    ("E within", "state_excit_within"),
    ("E across", "state_excit_between"),
)
FIELD_COLUMNS = (
    ("E within", "field_excit_within"),
    ("E across", "field_excit_between"),
)
VIOLIN_COLORS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#ccb974")
MAX_CORRELATION_SAMPLES = 5000


@dataclass
class NetworkRecord:
    folder: str
    tag: str
    connection_type: str | None
    kappa: float | None
    mean_connectivity: float | None
    state_corr: Dict[str, np.ndarray]
    field_corr: Dict[str, np.ndarray]
    var_temporal: np.ndarray
    var_quenched: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect legacy binary-network analyses (max-rates + multi-init correlations) "
            "and build a combined correlation/variance summary figure."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root folder containing connection_type/RjXX_XX/<tag> data (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination for the generated figure (default: plots/legacy_correlation_summary.png).",
    )
    parser.add_argument(
        "--connection-type",
        type=str,
        help="Restrict to a specific connection_type (case-insensitive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only include the first N networks after sorting (default: all).",
    )
    return parser.parse_args()


def _coerce_scalar(value: str) -> float | int | str | None:
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
        lowered = text.lower()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_param_scalars(path: str, keys: Iterable[str]) -> Dict[str, float | int | str | None]:
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return {key: data.get(key) for key in keys}
    result: Dict[str, float | int | str | None] = {key: None for key in keys}
    pending = set(keys)
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if key not in pending:
                continue
            result[key] = _coerce_scalar(raw_value)
            pending.discard(key)
            if not pending:
                break
    return result


def _finite_vector(array: np.ndarray | Sequence[float], *, max_samples: int | None = None) -> np.ndarray:
    values = np.asarray(array, dtype=float).ravel()
    if values.size == 0:
        return values
    mask = np.isfinite(values)
    clean = values[mask]
    if max_samples and max_samples > 0 and clean.size > max_samples:
        idx = np.linspace(0, clean.size - 1, max_samples, dtype=int)
        clean = clean[idx]
    return clean


def _safe_float(value: float | int | str | None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean_connectivity(scalars: Dict[str, float | int | str | None]) -> float | None:
    N_E = _safe_float(scalars.get("N_E"))
    N_I = _safe_float(scalars.get("N_I"))
    p_ee = _safe_float(scalars.get("p0_ee"))
    p_ei = _safe_float(scalars.get("p0_ei"))
    p_ie = _safe_float(scalars.get("p0_ie"))
    p_ii = _safe_float(scalars.get("p0_ii"))
    values = (N_E, N_I, p_ee, p_ei, p_ie, p_ii)
    if any(value is None for value in values):
        return None
    assert N_E is not None and N_I is not None
    assert p_ee is not None and p_ei is not None and p_ie is not None and p_ii is not None
    denom = (N_E + N_I) ** 2
    if denom <= 0:
        return None
    numerator = (N_E ** 2) * p_ee + (N_E * N_I) * (p_ei + p_ie) + (N_I ** 2) * p_ii
    return numerator / denom


def _load_summary(summary_path: str) -> NetworkRecord | None:
    parent = os.path.dirname(os.path.dirname(summary_path))
    params_path = os.path.join(os.path.dirname(parent), "params.yaml")
    if not os.path.exists(params_path):
        print(f"Skipping {parent}: missing params.yaml")
        return None
    scalars = _load_param_scalars(
        params_path,
        ("kappa", "N_E", "N_I", "p0_ee", "p0_ei", "p0_ie", "p0_ii", "connection_type"),
    )
    mean_conn = _mean_connectivity(scalars)
    with np.load(summary_path, allow_pickle=True) as data:
        state_corr = {
            key: _finite_vector(data[key], max_samples=MAX_CORRELATION_SAMPLES) if key in data else np.zeros(0)
            for _, key in STATE_COLUMNS
        }
        field_corr = {
            key: _finite_vector(data[key], max_samples=MAX_CORRELATION_SAMPLES) if key in data else np.zeros(0)
            for _, key in FIELD_COLUMNS
        }
        var_temporal = _finite_vector(data["var_temporal_mean"]) if "var_temporal_mean" in data else np.zeros(0)
        var_quenched = _finite_vector(data["var_quenched_mean"]) if "var_quenched_mean" in data else np.zeros(0)
    folder = os.path.dirname(parent)
    tag = os.path.basename(folder)
    connection_type = scalars.get("connection_type")
    return NetworkRecord(
        folder=folder,
        tag=tag,
        connection_type=str(connection_type) if connection_type is not None else None,
        kappa=_safe_float(scalars.get("kappa")),
        mean_connectivity=mean_conn,
        state_corr=state_corr,
        field_corr=field_corr,
        var_temporal=var_temporal,
        var_quenched=var_quenched,
    )


def _iter_network_folders(root: str) -> List[str]:
    summary_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(os.path.abspath(root)):
        if os.path.basename(dirpath) != "single_network_multi_init_corr_var":
            continue
        if "pooled_summary.npz" not in filenames:
            continue
        binary_dir = os.path.dirname(dirpath)
        legacy_dir = os.path.join(binary_dir, "max_rates_distribution_fp_init_legacy")
        summary_file = os.path.join(dirpath, "pooled_summary.npz")
        if not os.path.exists(os.path.join(legacy_dir, "analysis_summary.yaml")):
            print(f"Skipping {dirpath}: missing max_rates_distribution_fp_init_legacy results.")
            continue
        summary_paths.append(summary_file)
    return sorted(summary_paths)


def _sort_value(value: float | None) -> float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return math.inf
    return float(value)


def _format_label(record: NetworkRecord) -> str:
    kappa = "?" if record.kappa is None else f"{record.kappa:.2f}"
    mean_conn = "?" if record.mean_connectivity is None else f"{record.mean_connectivity:.4f}"
    return f"kappa={kappa}, conn={mean_conn}\n{record.tag}"


def _plot_placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def _plot_variance_violin(
    ax: plt.Axes,
    records: Sequence[NetworkRecord],
    row_labels: Sequence[str],
    conn_boundaries: Sequence[int],
) -> None:
    if not records:
        _plot_placeholder(ax, "No variance data")
        return
    base_positions = np.arange(len(records), dtype=float)
    width = 0.35
    temporal_data: List[np.ndarray] = []
    temporal_pos: List[float] = []
    quenched_data: List[np.ndarray] = []
    quenched_pos: List[float] = []
    for idx, record in enumerate(records):
        if record.var_temporal.size:
            temporal_data.append(record.var_temporal)
            temporal_pos.append(base_positions[idx] - width / 2)
        if record.var_quenched.size:
            quenched_data.append(record.var_quenched)
            quenched_pos.append(base_positions[idx] + width / 2)
    handles = []
    if temporal_data:
        viol = ax.violinplot(temporal_data, positions=temporal_pos, widths=width, showmeans=True)
        for body in viol["bodies"]:
            body.set_facecolor("#4c72b0")
            body.set_alpha(0.7)
        handles.append(Line2D([0], [0], color="#4c72b0", lw=4, label="Temporal"))
    if quenched_data:
        viol = ax.violinplot(quenched_data, positions=quenched_pos, widths=width, showmeans=True)
        for body in viol["bodies"]:
            body.set_facecolor("#dd8452")
            body.set_alpha(0.7)
        handles.append(Line2D([0], [0], color="#dd8452", lw=4, label="Quenched"))
    ax.set_xticks(base_positions)
    ax.set_xticklabels(row_labels, rotation=35, ha="right")
    ax.set_ylabel("Variance")
    ax.set_title("Variance split per network (assembly distributions)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    _add_variance_guides(ax, conn_boundaries)
    if handles:
        ax.legend(handles=handles, loc="best")


def _plot_correlation_violins(
    ax: plt.Axes,
    records: Sequence[NetworkRecord],
    columns: Sequence[tuple[str, str]],
    accessor,
    row_labels: Sequence[str],
    title: str,
    conn_boundaries: Sequence[int],
) -> None:
    if not records:
        _plot_placeholder(ax, "No correlation data")
        ax.set_title(title)
        return
    record_count = len(records)
    block_gap = max(1.0, record_count * 0.1)
    width = 0.5
    handles = []
    xticks: List[float] = []
    xticklabels: List[str] = []
    for col_idx, (label, key) in enumerate(columns):
        color = VIOLIN_COLORS[col_idx % len(VIOLIN_COLORS)]
        any_data = False
        block_start = col_idx * (record_count + block_gap)
        for idx, record in enumerate(records):
            pos = block_start + idx
            xticks.append(pos)
            xticklabels.append(f"{row_labels[idx]}\n{label}")
            data_array = accessor(record).get(key, np.zeros(0))
            clean = data_array[np.isfinite(data_array)]
            if clean.size == 0:
                continue
            viol = ax.violinplot(clean, positions=[pos], widths=width, showmeans=True)
            for body in viol["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.7)
            for part in ("cbars", "cmins", "cmaxes", "cmeans"):
                if part in viol:
                    viol[part].set_edgecolor("#333333")
                    viol[part].set_linewidth(0.6)
            any_data = True
        if any_data:
            handles.append(Line2D([0], [0], color=color, lw=4, label=label))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=40, ha="right")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    _add_correlation_guides(ax, conn_boundaries, record_count, block_gap, len(columns))
    if handles:
        ax.legend(handles=handles, loc="best")


def _connection_boundaries(records: Sequence[NetworkRecord]) -> List[int]:
    boundaries: List[int] = []
    if not records:
        return boundaries
    last_value = records[0].mean_connectivity
    for idx in range(1, len(records)):
        value = records[idx].mean_connectivity
        if not _conn_equal(value, last_value):
            boundaries.append(idx)
            last_value = value
    return boundaries


def _conn_equal(a: float | None, b: float | None, *, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale


def _add_correlation_guides(
    ax: plt.Axes,
    boundaries: Sequence[int],
    record_count: int,
    block_gap: float,
    block_count: int,
) -> None:
    if not boundaries:
        return
    for boundary in boundaries:
        for block_idx in range(block_count):
            x = block_idx * (record_count + block_gap) + boundary - 0.5
            ax.axvline(x, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)


def _add_variance_guides(ax: plt.Axes, boundaries: Sequence[int]) -> None:
    if not boundaries:
        return
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)


def main() -> None:
    args = parse_args()
    summary_paths = _iter_network_folders(args.root)
    records: List[NetworkRecord] = []
    for summary_path in summary_paths:
        record = _load_summary(summary_path)
        if record is None:
            continue
        if args.connection_type:
            conn = record.connection_type or ""
            if conn.lower() != args.connection_type.lower():
                continue
        records.append(record)
    records.sort(key=lambda rec: (_sort_value(rec.mean_connectivity), _sort_value(rec.kappa), rec.tag))
    if args.limit:
        records = records[: max(0, args.limit)]
    if not records:
        raise RuntimeError("No matching network folders with both legacy analyses were found.")
    row_labels = [_format_label(record) for record in records]
    conn_boundaries = _connection_boundaries(records)
    output_path = args.output or os.path.join("plots", "legacy_correlation_summary.png")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2])
    ax_state = fig.add_subplot(grid[0, 0])
    ax_field = fig.add_subplot(grid[1, 0])
    ax_variance = fig.add_subplot(grid[2, 0])
    _plot_correlation_violins(
        ax_state,
        records,
        STATE_COLUMNS,
        lambda record: record.state_corr,
        row_labels,
        "State correlations",
        conn_boundaries,
    )
    _plot_correlation_violins(
        ax_field,
        records,
        FIELD_COLUMNS,
        lambda record: record.field_corr,
        row_labels,
        "Subthreshold correlations",
        conn_boundaries,
    )
    _plot_variance_violin(ax_variance, records, row_labels, conn_boundaries)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Saved summary figure to {output_path}")


if __name__ == "__main__":
    main()
