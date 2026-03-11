#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

from pipelines.binary import ensure_binary_behavior_defaults, run_binary_simulation
from sim_config import deep_update, load_config, parse_overrides, sim_tag_from_cfg, write_yaml_config
import yaml

from plotting import (
    BinaryStateSource,
    FontCfg,
    RasterLabels,
    add_image_ax,
    add_panel_label,
    plot_binary_raster,
    plot_spike_raster,
    style_axes,
)


REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
EXTERNAL_IMAGE_PATH = FIGURES_DIR / "external" / "Network_single.jpg"
EXTERNAL_IMAGE_PATH2 = FIGURES_DIR / "external" / "Legend.jpg"
OUTPUT_PREFIX = FIGURES_DIR / "Figure1"

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
    marker_size: float = 1.5
    label_font_size: float = 7.0


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
    state_interval: int
    warmup_steps: int
    state_source: BinaryStateSource
    spike_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    spike_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    spike_trials: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))


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
class PanelContext:
    spec: RasterPanelSpec
    parameter: Dict[str, object]
    binary_cfg: Dict[str, object]


RASTER_PANELS: list[RasterPanelSpec] = [
    RasterPanelSpec(
        label="c1",
        config_name="default_simulation",
        overrides=BASE_RASTER_OVERRIDES,
        output_name="figure1_b1",
        window_start=2000,
        window_duration=RASTER_WINDOW_DURATION,
        cluster_count=20,
    ),
    RasterPanelSpec(
        label="c2",
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
        help="Panel-specific override using the panel label (e.g., c1:kappa=0).",
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
        "--raster-neuron-step",
        type=int,
        default=5,
        metavar="N",
        help="Only plot every Nth neuron in the raster panels (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for the multi-init simulations (default: %(default)s).",
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
    return ensure_binary_behavior_defaults(cfg)


def _filtered_erf_parameter(parameter: Dict[str, object]) -> Dict[str, object]:
    filtered = dict(parameter)
    filtered.pop("R_Eplus", None)
    return filtered


def binary_output_folder(parameter: Dict[str, object], binary_cfg: Dict[str, object]) -> tuple[Path, Path]:
    conn_name = str(parameter.get("connection_type", "bernoulli")).strip()
    conn_label = conn_name.capitalize()
    r_j = float(parameter.get("R_j", 0.0))
    rj_label = f"Rj{r_j:05.2f}".replace(".", "_")
    filtered = _filtered_erf_parameter(parameter)
    base_tag = sim_tag_from_cfg(filtered)
    base_dir = REPO_ROOT / "data" / conn_label / rj_label / base_tag
    binary_payload = {
        "parameter": parameter,
        "binary": binary_cfg,
    }
    binary_tag = sim_tag_from_cfg(binary_payload)
    binary_dir = base_dir / "binary" / binary_tag
    return base_dir, binary_dir


def expected_trace_path(parameter: Dict[str, object], binary_cfg: Dict[str, object], output_name: str) -> Path:
    _, binary_dir = binary_output_folder(parameter, binary_cfg)
    return binary_dir / f"{output_name}.npz"


def _resolve_population_inits(parameter: Dict[str, object], binary_cfg: Dict[str, object]) -> List[float]:
    template = binary_cfg.get("population_rate_init", 0.1)
    q_value = int(parameter.get("Q", 0) or 0)
    if q_value <= 0:
        raise ValueError("Parameter 'Q' must be positive.")
    default_value = 0.1

    def _uniform(value: float | None) -> np.ndarray:
        val = default_value if value is None else float(value)
        return np.full(q_value, np.clip(val, 0.0, 1.0), dtype=float)

    excit = inhib = None
    if template is None:
        excit = inhib = _uniform(None)
    elif isinstance(template, dict):
        excit = _uniform(template.get("excitatory"))
        inhib = _uniform(template.get("inhibitory"))
    elif isinstance(template, (list, tuple, np.ndarray)):
        arr = np.asarray(template, dtype=float).ravel()
        if arr.size == 2 * q_value:
            return arr.tolist()
        if arr.size == 2:
            excit = _uniform(arr[0])
            inhib = _uniform(arr[1])
        elif arr.size == 1:
            excit = inhib = _uniform(arr[0])
        else:
            raise ValueError(
                f"population_rate_init sequence must have length 1, 2, or 2*Q ({2 * q_value}), got {arr.size}."
            )
    else:
        excit = inhib = _uniform(template)
    if excit is None or inhib is None:
        raise ValueError("Could not infer population_rate_init values.")
    return np.concatenate([excit, inhib]).tolist()


def _resolve_binary_seed(binary_cfg: Dict[str, object]) -> int:
    seed = binary_cfg.get("seed")
    if seed is None:
        return int(np.random.randint(0, 2**31 - 1))
    return int(seed)


def ensure_trace_file(parameter: Dict[str, object], binary_cfg: Dict[str, object], spec: RasterPanelSpec) -> Path:
    run_cfg = dict(binary_cfg)
    run_cfg["output_name"] = spec.output_name
    seed = _resolve_binary_seed(run_cfg)
    run_cfg["seed"] = seed
    base_dir, binary_dir = binary_output_folder(parameter, run_cfg)
    trace_path = binary_dir / f"{spec.output_name}.npz"
    if trace_path.exists():
        return trace_path
    init_rates = _resolve_population_inits(parameter, run_cfg)
    base_dir.mkdir(parents=True, exist_ok=True)
    params_path = base_dir / "params.yaml"
    if not params_path.exists():
        write_yaml_config(_filtered_erf_parameter(parameter), params_path)
    binary_dir.mkdir(parents=True, exist_ok=True)
    binary_params_path = binary_dir / "params.yaml"
    if not binary_params_path.exists():
        write_yaml_config({"parameter": parameter, "binary": run_cfg}, binary_params_path)
    resolved = run_binary_simulation(
        parameter,
        run_cfg,
        output_name=spec.output_name,
        population_rate_inits=init_rates,
    )
    return Path(str(resolved["trace_path"]))


def _ensure_trace_for_context(context: PanelContext) -> str:
    path = ensure_trace_file(context.parameter, context.binary_cfg, context.spec)
    return str(path)


def ensure_traces_for_contexts(contexts: Sequence[PanelContext], jobs: int) -> List[Path]:
    if not contexts:
        return []
    worker_count = max(1, min(int(jobs or 1), len(contexts)))
    if worker_count > 1:
        with multiprocessing.Pool(processes=worker_count) as pool:
            resolved = pool.map(_ensure_trace_for_context, contexts)
        return [Path(entry) for entry in resolved]
    return [ensure_trace_file(context.parameter, context.binary_cfg, context.spec) for context in contexts]


def load_trace_summary(trace_path: Path) -> Dict[str, object]:
    summary_path = trace_path.with_name(f"{trace_path.stem}_summary.yaml")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file {summary_path} does not exist.")
    with summary_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_state_source(
    data: np.lib.npyio.NpzFile, summary: Dict[str, object], trace_path: Path
) -> BinaryStateSource:
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
    if "state_updates" in data and "state_deltas" in data:
        updates = np.asarray(data["state_updates"], dtype=np.uint16)
        deltas = np.asarray(data["state_deltas"], dtype=np.int8)
        init_state = None
        if "initial_state" in data and np.asarray(data["initial_state"]).size:
            init_state = np.asarray(data["initial_state"], dtype=np.uint8)
        return BinaryStateSource.from_diff_logs(
            updates,
            deltas,
            neuron_count=neuron_count,
            initial_state=init_state,
        )
    return BinaryStateSource(inline_states=inline_states, chunk_files=chunk_files, neuron_count=neuron_count)


def load_trace_payload(path: Path, summary: Dict[str, object]) -> TracePayload:
    if not path.exists():
        raise FileNotFoundError(f"Trace file {path} does not exist.")
    with np.load(path, allow_pickle=True) as data:
        available = set(data.files)
        rates = np.asarray(data["rates"], dtype=float)
        names = [str(name) for name in data["names"].tolist()]
        times = np.asarray(data["times"], dtype=float) if "times" in data else np.arange(rates.shape[0], dtype=float)
        sample_interval = int(np.asarray(data.get("sample_interval", 1)).item())
        state_times = (
            np.asarray(data["neuron_state_times"], dtype=np.int64)
            if "neuron_state_times" in available
            else np.zeros(0, dtype=np.int64)
        )
        state_interval = int(np.asarray(data.get("neuron_state_interval", 0)).item())
        if state_interval <= 0 and state_times.ndim == 1 and state_times.size > 1:
            diffs = np.diff(state_times.astype(np.int64, copy=False))
            valid = diffs[diffs > 0]
            if valid.size:
                state_interval = int(np.median(valid))
        if state_interval <= 0:
            state_interval = 1
        warmup_steps = int(np.asarray(data.get("warmup_steps", 0)).item())
        state_source = _build_state_source(data, summary, path)
        spike_times = (
            np.asarray(data["spike_times"], dtype=float).ravel()
            if "spike_times" in available
            else np.zeros(0, dtype=float)
        )
        spike_ids = (
            np.asarray(data["spike_ids"], dtype=np.int64).ravel()
            if "spike_ids" in available
            else np.zeros(0, dtype=np.int64)
        )
        spike_trials = (
            np.asarray(data["spike_trials"], dtype=np.int16).ravel()
            if "spike_trials" in available
            else np.zeros(0, dtype=np.int16)
        )
    return TracePayload(
        rates=rates,
        times=times,
        names=names,
        sample_interval=sample_interval,
        state_interval=state_interval,
        warmup_steps=warmup_steps,
        state_source=state_source,
        spike_times=spike_times,
        spike_ids=spike_ids,
        spike_trials=spike_trials,
    )


def select_excitatory_populations(names: Sequence[str]) -> List[int]:
    return [idx for idx, name in enumerate(names) if name.upper().startswith("E")]


def build_panel_context(
    spec: RasterPanelSpec,
    *,
    global_overrides: Sequence[str],
    panel_override_map: Dict[str, Sequence[str]],
) -> PanelContext:
    label_key = _normalize_label(spec.label)
    combined_overrides: List[str] = list(spec.overrides)
    combined_overrides.extend(global_overrides)
    combined_overrides.extend(panel_override_map.get(label_key, []))
    parameter = load_parameter(spec.config_name, combined_overrides)
    binary_cfg = resolve_binary_config(parameter)
    return PanelContext(spec=spec, parameter=parameter, binary_cfg=binary_cfg)


def prepare_panel(
    context: PanelContext,
    trace_path: Path,
) -> PanelData:
    spec = context.spec
    parameter = context.parameter
    binary_cfg = context.binary_cfg
    summary = load_trace_summary(trace_path)
    payload = load_trace_payload(trace_path, summary)
    if payload.state_source.neuron_count <= 0:
        total_neurons = int(parameter.get("N_E", 0)) + int(parameter.get("N_I", 0))
        payload.state_source.neuron_count = max(0, total_neurons)
    excitatory_neurons = int(parameter.get("N_E", payload.state_source.neuron_count or 0))
    excitatory_neurons = max(0, excitatory_neurons)
    excitatory_population_indices = select_excitatory_populations(payload.names)
    return PanelData(
        spec=context.spec,
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


def _time_axis_scale_from_taus(parameter: Dict[str, object]) -> tuple[float, str]:
    tau_e = float(parameter.get("tau_e", math.nan))
    tau_i = float(parameter.get("tau_i", math.nan))
    n_e = int(parameter.get("N_E", 0) or 0)
    n_i = int(parameter.get("N_I", 0) or 0)
    if not np.isfinite(tau_e) or not np.isfinite(tau_i) or tau_e <= 0.0 or tau_i <= 0.0:
        return 1.0, "Time [s]"
    max_tau = max(tau_e, tau_i)
    min_tau = min(tau_e, tau_i)
    n_max = n_e if tau_e >= tau_i else n_i
    n_min = n_i if tau_e >= tau_i else n_e
    expected_updates = float(n_max) + (max_tau / min_tau) * float(n_min)
    if not np.isfinite(expected_updates) or expected_updates <= 0.0 or max_tau <= 0.0:
        return 1.0, "Time [s]"
    max_tau_seconds = max_tau / 1000.0
    if max_tau_seconds <= 0.0 or not np.isfinite(max_tau_seconds):
        return 1.0, "Time [s]"
    return float(expected_updates / max_tau_seconds), "Time [s]"


def plot_cluster_activity(
    ax: Axes,
    panel: PanelData,
    window: TimeWindow,
    *,
    ylabel: str | None = None,
    time_scale: float,
    time_label: str,
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
    safe_scale = time_scale if time_scale > 0 else 1.0
    scaled_time_view = time_view / safe_scale
    cmap = plt.get_cmap("Greys")
    if len(cluster_indices) <= 1:
        shade_values = [0.6]
    else:
        shade_values = np.linspace(0.2, 0.85, len(cluster_indices))
    for idx, shade in zip(cluster_indices, shade_values):
        if idx >= rate_view.shape[1]:
            continue
        label = panel.payload.names[idx]
        ax.plot(scaled_time_view, rate_view[:, idx], color=cmap(shade), linewidth=1.2, label=label)
    window_start = window.start / safe_scale
    window_end = window.end / safe_scale
    ax.set_xlim(window_start, window_end)
    ax.set_ylim(0.0, 1.05)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    ax.set_xlabel(time_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


PANEL_LABEL_COORDS = (-0.12, 1.02)


def annotate_panel(ax: Axes, label: str, font_cfg: FontCfg) -> None:
    add_panel_label(ax, label, font_cfg, x=PANEL_LABEL_COORDS[0], y=PANEL_LABEL_COORDS[1])


def plot_contrast_curves(ax: Axes, q_value: float, rep_value: float, font_cfg: FontCfg) -> None:
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
    ax_density = ax
    ax_weight = ax_density.twinx()
    ax_weight.spines["left"].set_visible(False)
    ax_density.spines["right"].set_visible(False)
    ax_density.plot(kappa, density_contrast, color=density_color, linewidth=1.5)
    ax_weight.plot(kappa, weight_contrast, color=weight_color, linewidth=1.5)
    density_limit, density_ticks = _two_tick_limits(density_contrast)
    weight_limit, weight_ticks = _two_tick_limits(weight_contrast)
    ax_density.set_ylim(0.0, density_limit)
    ax_weight.set_ylim(0.0, weight_limit)
    ax_density.set_yticks(density_ticks)
    ax_density.set_yticklabels([f"{int(tick)}" for tick in density_ticks])
    ax_weight.set_yticks(weight_ticks)
    ax_weight.set_yticklabels([f"{int(tick)}" for tick in weight_ticks])
    ax_density.set_xlabel(r"$\kappa$", labelpad=-20.0)
    ax_density.set_ylabel(r"$p_{in}/p_{out}$", color=density_color)
    ax_weight.set_ylabel(r"$w_{in}/w_{out}$", color=weight_color)
    ax_density.tick_params(axis="y", colors=density_color)
    ax_weight.tick_params(axis="y", colors=weight_color)
    ax_density.spines["left"].set_color(density_color)
    ax_weight.spines["right"].set_color(weight_color)
    ax_weight.spines["right"].set_visible(True)
    ax_density.set_xlim(0.0, 1.0)
    style_axes(ax_density, font_cfg)
    style_axes(ax_weight, font_cfg, set_xlabel=False)


def _two_tick_limits(values: np.ndarray) -> tuple[float, list[int]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        max_val = 2.0
    else:
        max_val = float(arr.max())
        if max_val <= 0.0:
            max_val = 2.0
    scale = int(math.ceil(max_val / 2.0)) * 2
    if scale <= 0:
        scale = 2
    limit = scale * 1.05
    ticks = [max(1, scale // 2), scale]
    return limit, ticks


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
    contexts = [
        build_panel_context(spec, global_overrides=args.override, panel_override_map=panel_override_map)
        for spec in specs
    ]
    trace_paths = ensure_traces_for_contexts(contexts, jobs=args.jobs)
    panels = [
        prepare_panel(context, trace_path)
        for context, trace_path in zip(contexts, trace_paths)
    ]
    validate_panels(panels)
    font_cfg = FontCfg(base=12, scale=1.3).resolve()
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.2, 0.35, 0.6, 0.6], hspace=0.00, wspace=0.25)
    ax_image = fig.add_subplot(grid[0, :])
    left_grid = grid[2:4, 0].subgridspec(2, 1, height_ratios=[0.7, 0.2], hspace=0.015)
    right_grid = grid[2:4, 1].subgridspec(2, 1, height_ratios=[0.7, 0.2], hspace=0.015)
    ax_c1_raster = fig.add_subplot(left_grid[0])
    ax_c1_rates = fig.add_subplot(left_grid[1], sharex=ax_c1_raster)
    ax_c2_raster = fig.add_subplot(right_grid[0])
    ax_c2_rates = fig.add_subplot(right_grid[1], sharex=ax_c2_raster)
    contrast_grid = grid[1, :].subgridspec(
        1, 4,
        width_ratios=[0.025,0.375, 0.5, 0.125],  # <-- increase 0.08 for more side whitespace
        hspace = 0.00, wspace = 0.05
    )
    ax_contrast = fig.add_subplot(contrast_grid[0, 2])
    ax_legend = fig.add_subplot(contrast_grid[0, 1])
    add_image_ax(ax_image, str(EXTERNAL_IMAGE_PATH), fc=font_cfg)
    add_image_ax(ax_legend, str(EXTERNAL_IMAGE_PATH2), fc=font_cfg)
    add_panel_label(ax_image, "a", font_cfg, x=PANEL_LABEL_COORDS[0], y=PANEL_LABEL_COORDS[1])
    panel_lookup = {panel.spec.label: panel for panel in panels}
    axis_pairs = {
        "c1": (ax_c1_raster, ax_c1_rates),
        "c2": (ax_c2_raster, ax_c2_rates),
    }
    for label, (raster_ax, rate_ax) in axis_pairs.items():
        panel = panel_lookup.get(label)
        if panel is None:
            continue
        window = TimeWindow(start=float(panel.spec.window_start), duration=float(panel.spec.window_duration))
        time_scale, time_label = _time_axis_scale_from_taus(panel.parameter)
        total_neurons = panel.payload.state_source.neuron_count
        if total_neurons <= 0:
            total_neurons = int(panel.parameter.get("N_E", 0)) + int(panel.parameter.get("N_I", 0))
        labels = RasterLabels(
            show=True,
            excitatory="Exc.",
            inhibitory="Inh.",
            location="left",
            kwargs={
                "fontsize": font_cfg.tick,
                "rotation": 90,
                "ha": "right",
                "va": "center",
            },
        )
        existing = {id(text) for text in raster_ax.texts}
        n_inh = max(total_neurons - panel.excitatory_neurons, 0)
        if panel.payload.spike_times.size and panel.payload.spike_ids.size:
            safe_scale = time_scale if time_scale > 0 else 1.0
            scaled_times = panel.payload.spike_times / safe_scale
            t_start = window.start / safe_scale
            t_end = window.end / safe_scale
            plot_spike_raster(
                ax=raster_ax,
                spike_times_ms=scaled_times,
                spike_ids=panel.payload.spike_ids,
                n_exc=panel.excitatory_neurons,
                n_inh=n_inh,
                stride=args.raster_neuron_step,
                t_start=t_start,
                t_end=t_end,
                marker=".",
                marker_size=max(2.0, float(panel.spec.marker_size)),
                labels=labels,
            )
        else:
            plot_binary_raster(
                ax=raster_ax,
                state_source=panel.payload.state_source,
                sample_interval=panel.payload.state_interval,
                n_exc=panel.excitatory_neurons,
                total_neurons=total_neurons,
                window=(window.start, window.end),
                time_scale=time_scale,
                stride=args.raster_neuron_step,
                labels=labels,
                marker=".",
                marker_size=max(2.0, float(panel.spec.marker_size)),
            )
        for text in raster_ax.texts:
            if id(text) not in existing:
                text_label = text.get_text().strip().lower()
                if text_label.startswith("inh"):
                    text.set_color("#8B0000")
                elif text_label.startswith("exc"):
                    text.set_color("black")
        raster_ax.tick_params(axis="y", left=False, labelleft=False)
        raster_ax.tick_params(axis="x", labelbottom=False)
        ylabel = r"$m_c$" if label == "c1" else None
        plot_cluster_activity(
            rate_ax,
            panel,
            window,
            ylabel=ylabel,
            time_scale=time_scale,
            time_label=time_label,
        )
        style_axes(raster_ax, font_cfg, set_xlabel=False, set_ylabel=False)
        style_axes(rate_ax, font_cfg)
        if label == "c1":
            raster_ax.set_title(r"$\kappa=0$", fontsize=font_cfg.title)
        else:
            raster_ax.set_title(r"$\kappa=1$", fontsize=font_cfg.title)
        add_panel_label(raster_ax, label, font_cfg, x=PANEL_LABEL_COORDS[0], y=PANEL_LABEL_COORDS[1]*1.05)
    reference_panel = panels[0]
    plot_contrast_curves(
        ax_contrast,
        float(reference_panel.parameter.get("Q", 0.0)),
        float(reference_panel.parameter.get("R_Eplus", 0.0)),
        font_cfg,
    )
    add_panel_label(ax_contrast, "b", font_cfg, x=1.5*PANEL_LABEL_COORDS[0], y=PANEL_LABEL_COORDS[1]*1.05)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax_contrast.set_xlabel(r"$\kappa$", labelpad=-12.0)
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600)
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
