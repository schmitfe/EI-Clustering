#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import matplotlib
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from binary_pipeline import ensure_binary_behavior_defaults, run_binary_simulation
from plot_config import MM_TO_INCH, PlotConfig
from sim_config import deep_update, load_config, parse_overrides, sim_tag_from_cfg
import yaml


REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
EXTERNAL_IMAGE_PATH = FIGURES_DIR / "external" / "Network-Clustering_wide.jpeg"
OUTPUT_PREFIX = FIGURES_DIR / "Figure1"

DEFAULT_FIGURE_HEIGHT_MM = 190.0
BASE_RASTER_OVERRIDES: tuple[str, ...] = (
    "R_Eplus=1.5",
    "binary.seed=3",
    "binary.warmup_steps=5000",
    "binary.simulation_steps=20000",
    "binary.sample_interval=10",
    "binary.batch_size=10",
    "binary.log_decimate_factor=1",
    "binary.state_chunk_size=2000",
)
RASTER_WINDOW_DURATION = 4000


@dataclass(frozen=True)
class RasterPanelSpec:
    label: str
    config_name: str
    overrides: Sequence[str] = ()
    output_name: str = "activity_trace"
    window_start: float = 0.0
    window_duration: float = RASTER_WINDOW_DURATION
    cluster_indices: Sequence[int] | None = None
    cluster_count: int = 20


@dataclass(frozen=True)
class TimeWindow:
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class TracePayload:
    rates: np.ndarray
    times: np.ndarray
    names: List[str]
    sample_interval: int
    warmup_steps: int
    state_source: "StateSource"


@dataclass
class PanelData:
    spec: RasterPanelSpec
    parameter: Dict[str, object]
    binary_cfg: Dict[str, object]
    trace_path: Path
    payload: TracePayload
    excitatory_neurons: int
    excitatory_population_indices: List[int]


@dataclass
class PanelAxes:
    raster: Axes
    rates: Axes


@dataclass
class StateSource:
    inline_states: np.ndarray | None
    chunk_files: Sequence[Path]
    neuron_count: int

    def iter_chunks(self) -> Iterator[np.ndarray]:
        if self.inline_states is not None and self.inline_states.size:
            yield self.inline_states
            return
        if self.chunk_files:
            for path in self.chunk_files:
                if not path.exists():
                    continue
                chunk = np.load(path, allow_pickle=False, mmap_mode="r")
                yield np.asarray(chunk, dtype=np.uint8)
            return
        return


RASTER_PANELS: list[RasterPanelSpec] = [
    RasterPanelSpec(
        label="b1",
        config_name="default_simulation",
        overrides=BASE_RASTER_OVERRIDES,
        output_name="figure1_b1",
        window_start=2000,
        window_duration=RASTER_WINDOW_DURATION,
        cluster_count=20,
    ),
    RasterPanelSpec(
        label="b2",
        config_name="default_simulation",
        overrides=BASE_RASTER_OVERRIDES,
        output_name="figure1_b2",
        window_start=12000,
        window_duration=RASTER_WINDOW_DURATION,
        cluster_count=20,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1 with configurable binary raster panels.")
    parser.add_argument(
        "--config",
        type=str,
        default="default_simulation",
        help="Base YAML configuration to load for all panels (default: %(default)s).",
    )
    parser.add_argument(
        "-O",
        "--override",
        action="append",
        default=[],
        metavar="path=value",
        help="Global configuration override applied to every panel (may be repeated).",
    )
    parser.add_argument(
        "--panel-override",
        action="append",
        default=[],
        metavar="label:path=value",
        help="Panel-specific override using the panel label (e.g., b1:kappa=0).",
    )
    parser.add_argument(
        "--panel-window",
        action="append",
        default=[],
        metavar="label:start:duration",
        help="Override the time window (simulation steps) for a panel.",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        help="Global number of excitatory clusters plotted per panel (default: spec-defined).",
    )
    parser.add_argument(
        "--panel-cluster-count",
        action="append",
        default=[],
        metavar="label:count",
        help="Override the number of clusters for a specific panel.",
    )
    parser.add_argument(
        "--figure-height-mm",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT_MM,
        help="Overall figure height in millimeters (default: %(default)s).",
    )
    return parser.parse_args()


def _normalize_label(label: str) -> str:
    if not label or not label.strip():
        raise ValueError("Panel label must not be empty.")
    return label.strip().lower()


def _parse_panel_override_entries(entries: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Panel override '{raw}' is missing ':' between label and override.")
        label, override = raw.split(":", 1)
        override = override.strip()
        if "=" not in override:
            raise ValueError(f"Panel override '{raw}' must include '=' in the override expression.")
        key = _normalize_label(label)
        mapping.setdefault(key, []).append(override)
    return mapping


def _parse_panel_window_entries(entries: Sequence[str]) -> Dict[str, tuple[float, float]]:
    mapping: Dict[str, tuple[float, float]] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Panel window '{raw}' is missing ':' separators.")
        label, payload = raw.split(":", 1)
        parts = [part.strip() for part in payload.split(":") if part.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Panel window '{raw}' must specify 'label:start:duration' (received {len(parts)} values)."
            )
        start, duration = float(parts[0]), float(parts[1])
        mapping[_normalize_label(label)] = (start, duration)
    return mapping


def _parse_panel_cluster_entries(entries: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Panel cluster entry '{raw}' must follow 'label:count'.")
        label, payload = raw.split(":", 1)
        mapping[_normalize_label(label)] = int(payload.strip())
    return mapping


def _validate_panel_keys(mapping: Dict[str, object], known: set[str], context: str) -> None:
    if not mapping:
        return
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"Unknown panel label(s) for {context}: {', '.join(unknown)}.")


def load_parameter(config_name: str, overrides: Sequence[str]) -> Dict[str, object]:
    parameter = load_config(config_name)
    if overrides:
        parameter = deep_update(parameter, parse_overrides(overrides))
    return parameter


def resolve_binary_config(parameter: Dict[str, object]) -> Dict[str, object]:
    cfg = dict(parameter.get("binary") or {})
    cfg["warmup_steps"] = int(cfg.get("warmup_steps", 5000))
    cfg["simulation_steps"] = int(cfg.get("simulation_steps", 20000))
    cfg["sample_interval"] = int(cfg.get("sample_interval", 10))
    cfg["batch_size"] = int(cfg.get("batch_size", 1))
    cfg["state_chunk_size"] = int(cfg.get("state_chunk_size", 0))
    cfg["plot_activity"] = bool(cfg.get("plot_activity", False))
    cfg["log_step_states"] = bool(cfg.get("log_step_states", True))
    cfg["log_decimate_factor"] = int(cfg.get("log_decimate_factor", 1))
    cfg["log_decimate_zero_phase"] = bool(cfg.get("log_decimate_zero_phase", True))
    cfg["log_decimate_ftype"] = cfg.get("log_decimate_ftype", "iir")
    if "population_rate_init" not in cfg:
        cfg["population_rate_init"] = 0.1
    if "output_name" not in cfg:
        cfg["output_name"] = "activity_trace"
    if cfg.get("seed") is not None:
        cfg["seed"] = int(cfg["seed"])
    queue_cfg = cfg.get("update_queue")
    if queue_cfg is not None and not isinstance(queue_cfg, dict):
        raise ValueError("binary.update_queue must be a mapping when provided.")
    return ensure_binary_behavior_defaults(cfg)


def binary_output_folder(parameter: Dict[str, object]) -> Path:
    conn_name = str(parameter.get("connection_type", "bernoulli")).strip()
    conn_label = conn_name.capitalize()
    r_j = float(parameter.get("R_j", 0.0))
    rj_label = f"Rj{r_j:05.2f}".replace(".", "_")
    filtered = {k: v for k, v in parameter.items() if k != "R_Eplus"}
    tag = sim_tag_from_cfg(filtered)
    return REPO_ROOT / "data" / conn_label / rj_label / tag / "binary"


def expected_trace_path(parameter: Dict[str, object], output_name: str) -> Path:
    return binary_output_folder(parameter) / f"{output_name}.npz"


def ensure_trace_file(parameter: Dict[str, object], binary_cfg: Dict[str, object], spec: RasterPanelSpec) -> Path:
    trace_path = expected_trace_path(parameter, spec.output_name)
    if trace_path.exists():
        return trace_path
    result = run_binary_simulation(parameter, dict(binary_cfg), output_name=spec.output_name)
    resolved = Path(result["trace_path"])
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def load_trace_summary(trace_path: Path) -> Dict[str, object]:
    summary_path = trace_path.with_name(f"{trace_path.stem}_summary.yaml")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file {summary_path} does not exist.")
    with summary_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_state_source(data: np.lib.npyio.NpzFile, summary: Dict[str, object], trace_path: Path) -> StateSource:
    chunk_info = summary.get("state_chunks") or {}
    chunk_files: List[Path] = []
    inline_states: np.ndarray | None = None
    if chunk_info and chunk_info.get("enabled") and chunk_info.get("files"):
        base_folder = trace_path.parent
        chunk_files = [base_folder / str(entry) for entry in chunk_info.get("files", [])]
    else:
        if "neuron_states" in data:
            inline_states = np.asarray(data["neuron_states"], dtype=np.uint8)
    neuron_count = int(
        summary.get("neurons")
        or (inline_states.shape[1] if inline_states is not None and inline_states.ndim == 2 else 0)
    )
    return StateSource(inline_states=inline_states, chunk_files=chunk_files, neuron_count=neuron_count)


def load_trace_payload(path: Path, summary: Dict[str, object]) -> TracePayload:
    if not path.exists():
        raise FileNotFoundError(f"Trace file {path} does not exist.")
    with np.load(path, allow_pickle=False) as data:
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"].tolist()]
        times = np.asarray(data["times"], dtype=float) if "times" in data else np.arange(rates.shape[0], dtype=float)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        warmup_steps = int(np.asarray(data.get("warmup_steps", 0)).item())
        state_source = _build_state_source(data, summary, path)
    return TracePayload(
        rates=rates,
        times=times,
        names=names,
        sample_interval=sample_interval,
        warmup_steps=warmup_steps,
        state_source=state_source,
    )


def select_excitatory_populations(names: Sequence[str]) -> List[int]:
    return [idx for idx, name in enumerate(names) if name.upper().startswith("E")]


def prepare_panel(
    spec: RasterPanelSpec,
    *,
    global_overrides: Sequence[str],
    panel_override_map: Dict[str, Sequence[str]],
) -> PanelData:
    label_key = _normalize_label(spec.label)
    combined_overrides: List[str] = list(spec.overrides)
    combined_overrides.extend(global_overrides)
    combined_overrides.extend(panel_override_map.get(label_key, []))
    parameter = load_parameter(spec.config_name, combined_overrides)
    binary_cfg = resolve_binary_config(parameter)
    trace_path = ensure_trace_file(parameter, binary_cfg, spec)
    summary = load_trace_summary(trace_path)
    payload = load_trace_payload(trace_path, summary)
    if payload.state_source.neuron_count <= 0:
        total_neurons = int(parameter.get("N_E", 0)) + int(parameter.get("N_I", 0))
        payload.state_source.neuron_count = max(0, total_neurons)
    excitatory_neurons = int(parameter.get("N_E", payload.state_source.neuron_count or 0))
    excitatory_neurons = max(0, excitatory_neurons)
    excitatory_population_indices = select_excitatory_populations(payload.names)
    return PanelData(
        spec=spec,
        parameter=parameter,
        binary_cfg=binary_cfg,
        trace_path=trace_path,
        payload=payload,
        excitatory_neurons=excitatory_neurons,
        excitatory_population_indices=excitatory_population_indices,
    )


def mask_window(values: np.ndarray, window: TimeWindow) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    return (values >= window.start) & (values <= window.end)


def select_clusters(panel: PanelData) -> List[int]:
    if not panel.excitatory_population_indices:
        return []
    if panel.spec.cluster_indices:
        selected: List[int] = []
        for idx in panel.spec.cluster_indices:
            idx = int(idx)
            if 0 <= idx < len(panel.excitatory_population_indices):
                selected.append(panel.excitatory_population_indices[idx])
        return selected
    count = min(panel.spec.cluster_count, len(panel.excitatory_population_indices))
    return panel.excitatory_population_indices[:count]


def _collect_onset_events(state_source: StateSource, sample_interval: int, window: TimeWindow) -> tuple[np.ndarray, np.ndarray]:
    if state_source.neuron_count <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    sample_interval = max(1, int(sample_interval))
    times_list: List[np.ndarray] = []
    neurons_list: List[np.ndarray] = []
    prev_state: np.ndarray | None = None
    sample_index = 0
    window_end = window.end
    for chunk in state_source.iter_chunks():
        chunk = np.asarray(chunk, dtype=np.uint8)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            continue
        for row in chunk:
            sample_index += 1
            if prev_state is None:
                prev_state = row.copy()
                continue
            transitions = (prev_state == 0) & (row == 1)
            transition_time = (sample_index - 1) * sample_interval
            prev_state = row.copy()
            if transition_time > window_end:
                return _finalize_events(times_list, neurons_list)
            if not transitions.any():
                continue
            if transition_time < window.start:
                continue
            neurons = np.flatnonzero(transitions)
            if neurons.size == 0:
                continue
            times_list.append(np.full(neurons.size, transition_time, dtype=np.float64))
            neurons_list.append(neurons.astype(np.int64))
    return _finalize_events(times_list, neurons_list)


def _finalize_events(times_list: List[np.ndarray], neurons_list: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not times_list:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(times_list), np.concatenate(neurons_list)


def plot_onset_raster(ax: Axes, panel: PanelData, window: TimeWindow) -> None:
    times, neurons = _collect_onset_events(panel.payload.state_source, panel.payload.sample_interval, window)
    if times.size == 0 or neurons.size == 0:
        ax.text(0.5, 0.5, "No neuron onset events", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    neuron_count = panel.payload.state_source.neuron_count
    if neuron_count <= 0:
        neuron_count = int(panel.parameter.get("N_E", 0)) + int(panel.parameter.get("N_I", 0))
    excit_limit = max(0, min(panel.excitatory_neurons, neuron_count))
    excit_mask = neurons < excit_limit
    inhib_mask = neurons >= excit_limit
    if excit_mask.any():
        ax.scatter(times[excit_mask], neurons[excit_mask], s=6, marker=".", color="black")
    if inhib_mask.any():
        ax.scatter(times[inhib_mask], neurons[inhib_mask], s=6, marker=".", color="#8B0000")
    ax.set_xlim(window.start, window.end)
    ax.set_ylim(-0.5, neuron_count - 0.5)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelbottom=False)
    n_i = max(0, neuron_count - panel.excitatory_neurons)
    text_x = window.start - 0.02 * max(1.0, window.duration)
    exc_pos = 0.5 * max(1, panel.excitatory_neurons)
    inh_pos = panel.excitatory_neurons + 0.5 * max(1, n_i)
    ax.text(
        text_x,
        exc_pos,
        "Exc.",
        color="black",
        va="center",
        ha="right",
        fontsize=7,
        rotation=90,
    )
    ax.text(
        text_x,
        inh_pos,
        "Inh.",
        color="#8B0000",
        va="center",
        ha="right",
        fontsize=7,
        rotation=90,
    )


def _format_time_ticks(start: float, end: float, count: int = 4) -> tuple[np.ndarray, List[str]]:
    if count <= 1 or end <= start:
        return np.array([start, end]), [f"{start:g}", f"{end:g}"]
    values = np.linspace(start, end, count, dtype=float)
    labels = []
    for value in values:
        if abs(value - round(value)) < 1e-6:
            labels.append(f"{round(value):d}")
        else:
            labels.append(f"{value:.1f}")
    return values, labels


def plot_cluster_activity(
    ax: Axes,
    panel: PanelData,
    window: TimeWindow,
    colors: Sequence[str],
    *,
    ylabel: str | None = None,
) -> None:
    rates = panel.payload.rates
    times = panel.payload.times
    cluster_indices = select_clusters(panel)
    if rates.size == 0 or not cluster_indices or times.size == 0:
        ax.text(0.5, 0.5, "No excitatory rate traces", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    mask = mask_window(times, window)
    time_view = times[mask] if mask.any() else times
    rate_view = rates[mask] if mask.any() else rates
    color_cycle = cycle(colors)
    for idx, color in zip(cluster_indices, color_cycle):
        if idx >= rate_view.shape[1]:
            continue
        label = panel.payload.names[idx]
        ax.plot(time_view, rate_view[:, idx], color=color, linewidth=1.2, label=label)
    ax.set_xlim(window.start, window.end)
    ax.set_ylim(0.0, 1.0)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Time [a.u.]")
    ticks, labels = _format_time_ticks(window.start, window.end, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def annotate_panel(
    ax: Axes,
    label: str,
    plot_cfg: PlotConfig,
    *,
    above: bool = False,
    fig: plt.Figure | None = None,
    anchor_ax: Axes | None = None,
) -> None:
    if above:
        coords = plot_cfg.panel_label_above_coords
        align = plot_cfg.panel_label_above_align
    else:
        coords = plot_cfg.panel_label_coords
        align = plot_cfg.panel_label_align
    if fig is not None and anchor_ax is not None:
        anchor_bbox = anchor_ax.get_position(fig.transFigure)
        target_bbox = ax.get_position(fig.transFigure)
        delta_x = coords[0] * anchor_bbox.width
        fig_x = target_bbox.x0 + delta_x
        fig_y = target_bbox.y1 + coords[1] * target_bbox.height
        fig.text(
            fig_x,
            fig_y,
            label,
            ha=align[0],
            va=align[1],
            fontweight="bold",
        )
        return
    ax.text(coords[0], coords[1], label, transform=ax.transAxes, ha=align[0], va=align[1], fontweight="bold")


def ensure_external_image() -> np.ndarray:
    if not EXTERNAL_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Panel a image {EXTERNAL_IMAGE_PATH} is missing.")
    return plt.imread(EXTERNAL_IMAGE_PATH)


def plot_contrast_curves(ax: Axes, q_value: float, rep_value: float, colors: Sequence[str]) -> None:
    kappa = np.linspace(0.0, 1.0, 200, dtype=float)
    rep_value = float(rep_value)
    q_value = float(q_value)
    if rep_value <= 0.0:
        raise ValueError("R_Eplus must be positive for the contrast curves.")
    rep_pow = np.power(rep_value, kappa)
    rep_inv_pow = np.power(rep_value, 1.0 - kappa)
    denom_weight = q_value - rep_pow
    denom_density = q_value - rep_inv_pow
    weight_contrast = np.full_like(kappa, np.nan)
    density_contrast = np.full_like(kappa, np.nan)
    valid_weight = np.abs(denom_weight) > 1e-12
    valid_density = np.abs(denom_density) > 1e-12
    weight_contrast[valid_weight] = rep_pow[valid_weight] * (q_value - 1.0) / denom_weight[valid_weight]
    density_contrast[valid_density] = rep_inv_pow[valid_density] * (q_value - 1.0) / denom_density[valid_density]
    weight_color = "#7FA64B"
    density_color = "#7A6BC6"
    ax_weight = ax
    ax_density = ax_weight.twinx()
    ax_weight.plot(kappa, weight_contrast, color=weight_color, linewidth=1.5)
    ax_density.plot(kappa, density_contrast, color=density_color, linewidth=1.5)
    ax_weight.set_xlabel(r"$\kappa$")
    ax_weight.set_ylabel(r"$w_{in}/w_{out}$", color=weight_color)
    ax_density.set_ylabel(r"$p_{in}/p_{out}$", color=density_color)
    ax_weight.tick_params(axis="y", colors=weight_color)
    ax_density.tick_params(axis="y", colors=density_color)
    ax_weight.spines["left"].set_color(weight_color)
    ax_density.spines["right"].set_color(density_color)
    ax_density.spines["right"].set_visible(True)
    ax_weight.set_xlim(0.0, 1.0)


def validate_panels(panels: Sequence[PanelData]) -> None:
    if not panels:
        raise ValueError("At least one raster panel must be configured.")
    warmups = {int(panel.binary_cfg.get("warmup_steps", 0)) for panel in panels}
    sims = {int(panel.binary_cfg.get("simulation_steps", 0)) for panel in panels}
    durations = {float(panel.spec.window_duration) for panel in panels}
    reps = {float(panel.parameter.get("R_Eplus", 0.0)) for panel in panels}
    q_values = {float(panel.parameter.get("Q", 0.0)) for panel in panels}
    if any(duration <= 0 for duration in durations):
        raise ValueError("Panel window durations must be positive.")
    if len(warmups) > 1:
        raise ValueError(f"Panels use different warmup lengths: {sorted(warmups)}")
    if len(sims) > 1:
        raise ValueError(f"Panels use different simulation lengths: {sorted(sims)}")
    if len(durations) > 1:
        raise ValueError(f"Panels use different window durations: {sorted(durations)}")
    if len(reps) > 1:
        raise ValueError("Panels must share the same R_Eplus to keep contrast curves consistent.")
    if len(q_values) > 1:
        raise ValueError("Panels must share the same Q to keep contrast curves consistent.")


def build_figure_layout(plot_cfg: PlotConfig, figure_height_mm: float) -> tuple[plt.Figure, Axes, PanelAxes, PanelAxes, Axes]:
    figure_height = max(50.0, float(figure_height_mm)) * MM_TO_INCH
    configured = PlotConfig(
        figure_width=plot_cfg.figure_width,
        figure_height=figure_height,
        base_font_size=plot_cfg.base_font_size,
        title_size=plot_cfg.title_size,
        label_size=plot_cfg.label_size,
        tick_size=plot_cfg.tick_size,
        palette=plot_cfg.palette,
        line_colors=plot_cfg.line_colors,
        panel_label_coords=plot_cfg.panel_label_coords,
        panel_label_align=plot_cfg.panel_label_align,
        panel_label_above_coords=plot_cfg.panel_label_above_coords,
        panel_label_above_align=plot_cfg.panel_label_above_align,
    )
    configured.apply()
    fig = plt.figure(constrained_layout=False)
    height_ratios = [1.7, 0.6, 0.6, 0.4]
    grid = fig.add_gridspec(4, 2, height_ratios=height_ratios, hspace=0.4, wspace=0.25)
    ax_image = fig.add_subplot(grid[0, :])
    left_grid = grid[1:3, 0].subgridspec(2, 1, height_ratios=[0.65, 0.35], hspace=0.05)
    right_grid = grid[1:3, 1].subgridspec(2, 1, height_ratios=[0.65, 0.35], hspace=0.05)
    ax_b1_raster = fig.add_subplot(left_grid[0])
    ax_b1_rates = fig.add_subplot(left_grid[1], sharex=ax_b1_raster)
    ax_b2_raster = fig.add_subplot(right_grid[0])
    ax_b2_rates = fig.add_subplot(right_grid[1], sharex=ax_b2_raster)
    ax_contrast = fig.add_subplot(grid[3, :])
    return fig, ax_image, PanelAxes(ax_b1_raster, ax_b1_rates), PanelAxes(ax_b2_raster, ax_b2_rates), ax_contrast


def build_panel_specs(
    config_name: str,
    *,
    global_cluster_count: int | None,
    window_map: Dict[str, tuple[float, float]],
    cluster_map: Dict[str, int],
) -> list[RasterPanelSpec]:
    specs: list[RasterPanelSpec] = []
    for spec in RASTER_PANELS:
        updated = replace(spec, config_name=config_name)
        label_key = _normalize_label(updated.label)
        if global_cluster_count is not None:
            updated = replace(updated, cluster_count=global_cluster_count)
        if label_key in cluster_map:
            updated = replace(updated, cluster_count=cluster_map[label_key])
        if label_key in window_map:
            start, duration = window_map[label_key]
            updated = replace(updated, window_start=start, window_duration=duration)
        specs.append(updated)
    return specs


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    known_labels = {_normalize_label(spec.label) for spec in RASTER_PANELS}
    panel_override_map = _parse_panel_override_entries(args.panel_override)
    panel_window_map = _parse_panel_window_entries(args.panel_window)
    panel_cluster_map = _parse_panel_cluster_entries(args.panel_cluster_count)
    _validate_panel_keys(panel_override_map, known_labels, "panel overrides")
    _validate_panel_keys(panel_window_map, known_labels, "panel windows")
    _validate_panel_keys(panel_cluster_map, known_labels, "panel cluster counts")
    specs = build_panel_specs(
        args.config,
        global_cluster_count=args.cluster_count,
        window_map=panel_window_map,
        cluster_map=panel_cluster_map,
    )
    panels = [
        prepare_panel(spec, global_overrides=args.override, panel_override_map=panel_override_map)
        for spec in specs
    ]
    validate_panels(panels)
    plot_cfg = PlotConfig()
    fig, ax_image, ax_b1, ax_b2, ax_contrast = build_figure_layout(plot_cfg, args.figure_height_mm)
    external_image = ensure_external_image()
    ax_image.imshow(external_image)
    ax_image.set_axis_off()
    annotate_panel(ax_image, "a", plot_cfg)
    panel_lookup = {panel.spec.label: panel for panel in panels}
    for axes, label in ((ax_b1, "b1"), (ax_b2, "b2")):
        panel = panel_lookup.get(label)
        if panel is None:
            continue
        window = TimeWindow(start=float(panel.spec.window_start), duration=float(panel.spec.window_duration))
        plot_onset_raster(axes.raster, panel, window)
        ylabel = r"$\bar{m_c}$" if label == "b1" else None
        plot_cluster_activity(axes.rates, panel, window, plot_cfg.line_colors, ylabel=ylabel)
        annotate_panel(axes.raster, label, plot_cfg)
    reference_panel = panels[0]
    plot_contrast_curves(
        ax_contrast,
        float(reference_panel.parameter.get("Q", 0.0)),
        float(reference_panel.parameter.get("R_Eplus", 0.0)),
        plot_cfg.line_colors,
    )
    annotate_panel(ax_contrast, "c", plot_cfg, fig=fig)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600)
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
