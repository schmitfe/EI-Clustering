#!/usr/bin/env python3
"""Long-run Figure 4 analysis restricted to single-active-cluster episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Figure3 import BinaryRunSettings, _resolve_binary_config, _time_axis_scale_from_taus
from Figure4 import _build_instances
from analysis.episode_inference import infer_active_set_episodes
from analysis.high_state import episode_statistics, sample_episode_pairs
from analysis.io import analysis_input_from_binary_trace
from figure_cli import add_v_sweep_arguments, parse_int_values, resolve_float_values, resolve_v_sweep
from pipelines import figure_helpers as helpers
from pipelines.binary import run_binary_simulation
from pipelines.figure_helpers import (
    _assembly_membership,
    _binary_output_folder,
    _fields_from_states,
    _instantiate_replay_network,
    _load_trace_payload,
    _resolve_fixpoint_candidates,
    _sampled_states_from_payload,
    _select_fixpoints,
)
from plotting import FontCfg, _prepare_value_color_map, draw_listed_colorbar, style_axes
from sim_config import add_override_arguments, load_from_args, write_yaml_config


SCALAR_METRICS = (
    "output_all_within_r",
    "output_all_within_z",
    "output_all_across_r",
    "output_all_across_z",
    "input_all_within_r",
    "input_all_within_z",
    "input_all_across_r",
    "input_all_across_z",
    "output_active_within_r",
    "output_active_within_z",
    "output_active_across_r",
    "output_active_across_z",
    "input_active_within_r",
    "input_active_within_z",
    "input_active_across_r",
    "input_active_across_z",
    "activity_active",
    "input_mean_global",
    "input_mean_active",
    "input_mean_inactive",
    "input_var_temporal_global",
    "input_var_temporal_active",
    "input_var_temporal_inactive",
    "input_var_quenched_global",
    "input_var_quenched_active",
    "input_var_quenched_inactive",
    "input_var_total_global",
    "input_var_total_active",
    "input_var_total_inactive",
)
COUNT_METRICS = (
    "output_all_within_n",
    "output_all_across_n",
    "input_all_within_n",
    "input_all_across_n",
    "output_active_within_n",
    "output_active_across_n",
    "input_active_within_n",
    "input_active_across_n",
)
VECTOR_METRICS = (
    "activity_mean_population",
    "input_mean_population",
    "input_var_temporal_population",
    "input_var_quenched_population",
    "input_var_total_population",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_override_arguments(parser)
    parser.add_argument("--kappas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])
    parser.add_argument("--mean-connectivity", nargs="+")
    parser.add_argument("--focus-counts", nargs="+", default=["5:1:-1"])
    parser.add_argument("--stability-filter", choices=("stable", "unstable", "any"), default="any")
    parser.add_argument("--n-networks", type=int, default=15)
    parser.add_argument("--n-inits", type=int, default=3)
    parser.add_argument("--network-indices", nargs="+", type=int)
    parser.add_argument("--init-indices", nargs="+", type=int)
    parser.add_argument("--network-seed-start", type=int, default=100)
    parser.add_argument("--init-seed-start", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--warmup-steps", type=int, default=400000)
    parser.add_argument("--sample-interval", type=int, default=12000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--minimum-episode-ms", type=float, default=200.0)
    parser.add_argument("--exclude-edge-episodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmentation", choices=("fixed", "pelt"), default="pelt")
    parser.add_argument("--fixed-width", type=int, default=1)
    parser.add_argument("--pelt-penalty", type=float, default=10.0)
    parser.add_argument("--changepoint-backend", choices=("skchange", "sktime", "ruptures"), default="skchange")
    parser.add_argument(
        "--changepoint-method",
        choices=("pelt", "moving_window", "seeded_binseg", "binseg"),
        default="pelt",
    )
    parser.add_argument("--pelt-min-size", type=int, default=2)
    parser.add_argument("--pelt-jump", type=int, default=2)
    parser.add_argument("--pelt-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pelt-smooth-width", type=int, default=1)
    parser.add_argument("--changepoint-bandwidth", type=int, default=20)
    parser.add_argument("--changepoint-max-interval-length", type=int, default=200)
    parser.add_argument("--changepoint-parallel-backend", default="None")
    parser.add_argument("--changepoint-parallel-jobs", type=int, default=1)
    parser.add_argument("--em-max-iter", type=int, default=30)
    parser.add_argument("--kmax", type=int)
    parser.add_argument("--canonical-kmax", type=int, default=2)
    parser.add_argument("--canonical-z-threshold", type=float, default=3.0)
    parser.add_argument("--canonical-similarity", type=float, default=0.5)
    parser.add_argument("--canonical-noise-floor", type=float, default=1e-6)
    parser.add_argument("--merge-after-em", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta-merge", type=float, default=0.0)
    parser.add_argument("--min-flicker-duration", type=int, default=3)
    parser.add_argument("--flicker-max-hamming", type=int, default=2)
    parser.add_argument("--merge-max-iter", type=int, default=10)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--use-existing-bundle", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite-simulation", action="store_true")
    parser.add_argument("--overwrite-analysis", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--plot-decomposed-variance", action="store_true")
    parser.add_argument("--output-dir", default="plots/Figure4_HighState")
    parser.add_argument("--output-prefix", default="Figures/Figure4_HighState")
    parser.add_argument("--dpi", type=int, default=300)
    add_v_sweep_arguments(parser)
    parser.add_argument("--retry-step", type=float)
    parser.add_argument("--erf-jobs", type=int, default=1)
    parser.add_argument("--overwrite-erf", action="store_true")
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _slug(connectivity: float, kappa: float, network_index: int, init_index: int) -> str:
    conn = format(float(connectivity), ".8g").replace(".", "p").replace("-", "m")
    kap = format(float(kappa), ".8g").replace(".", "p").replace("-", "m")
    return f"conn{conn}_kappa{kap}_net{network_index:03d}_init{init_index:03d}"


def _analysis_signature(parameter: dict[str, Any], args: argparse.Namespace) -> str:
    ignored = {
        "aggregate_only", "allow_incomplete", "analysis_only", "jobs", "no_aggregate",
        "output_dir", "output_prefix", "overwrite_analysis", "overwrite_erf", "overwrite_simulation",
        "prepare_only", "use_existing_bundle", "network_indices", "init_indices", "kappas",
        "mean_connectivity", "bootstrap_samples", "plot_decomposed_variance", "dpi",
    }
    settings = {key: value for key, value in vars(args).items() if key not in ignored}
    encoded = json.dumps(
        {"parameter": parameter, "settings": settings}, default=_json_default, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _prepare_bundle(
    parameter: dict[str, Any], args: argparse.Namespace, focus_counts: Sequence[int]
) -> str:
    if args.use_existing_bundle:
        path = helpers.compute_fixpoint_bundle_path(parameter)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return path
    v_start, v_stop, v_steps = resolve_v_sweep(args)
    sweep = helpers.PipelineSweepSettings(
        v_start=v_start,
        v_end=v_stop,
        v_steps=v_steps,
        retry_step=args.retry_step,
        jobs=max(1, int(args.erf_jobs)),
        overwrite_simulation=bool(args.overwrite_erf),
        plot_erfs=False,
    )
    _, path = helpers.ensure_fixpoint_bundle(
        deepcopy(parameter), focus_counts, [float(parameter["R_Eplus"])], sweep
    )
    return path


def _accepted_segments(canonical: Any, *, minimum_bins: int, exclude_edges: bool) -> pd.DataFrame:
    segments = canonical.segments.copy()
    if segments.empty:
        return segments
    segments["episode_index"] = np.arange(len(segments), dtype=int)
    mask = segments["K"].to_numpy(dtype=int) == 1
    mask &= segments["duration_bins"].to_numpy(dtype=int) >= int(minimum_bins)
    if exclude_edges and len(segments):
        mask[0] = False
        mask[-1] = False
    return segments.loc[mask].copy()


def _run_shard(task: dict[str, Any]) -> str:
    args: argparse.Namespace = task["args"]
    parameter = task["parameter"]
    bundle_path = task["bundle_path"]
    connectivity = float(task["connectivity"])
    kappa = float(parameter["kappa"])
    network_index = int(task["network_index"])
    init_index = int(task["init_index"])
    network_seed = int(args.network_seed_start) + network_index
    init_seed = int(args.init_seed_start) + network_index
    shard_dir = Path(args.output_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{_slug(connectivity, kappa, network_index, init_index)}.npz"
    signature = _analysis_signature(parameter, args)
    if shard_path.exists() and not args.overwrite_analysis:
        try:
            with np.load(shard_path, allow_pickle=False) as cached:
                cached_metadata = json.loads(str(cached["metadata_json"]))
            if cached_metadata.get("analysis_signature") == signature:
                return str(shard_path)
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            pass

    candidates = _resolve_fixpoint_candidates(parameter, bundle_path, task["focus_counts"], args.stability_filter)
    selected = _select_fixpoints(candidates, seed=init_seed, count=int(args.n_inits))
    if init_index >= len(selected):
        raise IndexError(f"Initialization {init_index} unavailable; selected {len(selected)} candidates")
    candidate = selected[init_index]
    updates_per_second, _ = _time_axis_scale_from_taus(parameter)
    simulation_steps = max(1, int(round(float(args.duration) * updates_per_second)))
    binary_cfg = _resolve_binary_config(
        parameter,
        BinaryRunSettings(
            warmup_steps=int(args.warmup_steps),
            simulation_steps=simulation_steps,
            sample_interval=int(args.sample_interval),
            batch_size=args.batch_size,
            seed=network_seed,
            output_name="figure4_high_state",
        ),
    )
    label = _slug(connectivity, kappa, network_index, init_index)
    trace_path = Path(_binary_output_folder(parameter, binary_cfg)) / f"{label}.npz"
    if not trace_path.exists() or args.overwrite_simulation:
        if args.analysis_only:
            raise FileNotFoundError(trace_path)
        result = run_binary_simulation(
            parameter,
            binary_cfg,
            output_name=label,
            population_rate_inits=candidate["rates"],
            population_init_seed=int(args.init_seed_start) + network_index * int(args.n_inits) + init_index,
        )
        trace_path = Path(result["trace_path"])

    dt = float(binary_cfg["sample_interval"]) / float(updates_per_second)
    full_data = analysis_input_from_binary_trace(trace_path, parameter=parameter, analysis_cfg={"dt": dt})
    inference = infer_active_set_episodes(
        full_data,
        args,
        condition=f"conn={connectivity:.6g},kappa={kappa:.6g}",
        title="Figure 4 high state",
        seed=network_seed,
        population_source="excitatory",
        canonicalize=True,
        exclude_edges=False,
    )
    canonical = inference.result
    minimum_bins = max(2, int(np.ceil(float(args.minimum_episode_ms) / (1000.0 * dt))))
    accepted = _accepted_segments(
        canonical, minimum_bins=minimum_bins, exclude_edges=bool(args.exclude_edge_episodes)
    )
    payload = _load_trace_payload(str(trace_path))
    states = _sampled_states_from_payload(payload, stride=1).astype(np.float32, copy=False)
    rates = np.asarray(payload["rates"], dtype=np.float32)
    n_samples = min(states.shape[0], rates.shape[0], int(full_data.n_timepoints))
    states, rates = states[:n_samples], rates[:n_samples]
    membership, population_names = _assembly_membership(parameter)
    q = int(parameter["Q"])
    pairs = sample_episode_pairs(
        membership, q=q, max_pairs=max(1, int(args.max_pairs)), seed=network_seed * 1009 + init_index
    )
    network = _instantiate_replay_network(parameter, binary_cfg) if len(accepted) else None
    records: list[dict[str, Any]] = []
    vectors: dict[str, list[np.ndarray]] = {key: [] for key in VECTOR_METRICS}
    analyzed_names = list(canonical.metadata.get("canonical_cluster_names", []))
    for _, segment in accepted.iterrows():
        start = max(0, int(segment["start_bin"]))
        stop = min(n_samples, int(segment["stop_bin"]))
        if stop - start < minimum_bins:
            continue
        cluster_name = str(segment["clusters"]).split(",")[0]
        if cluster_name not in analyzed_names:
            continue
        active_population = analyzed_names.index(cluster_name)
        fields = _fields_from_states(network, states[start:stop])  # type: ignore[arg-type]
        stats = episode_statistics(
            states=states[start:stop],
            fields=fields,
            rates=rates[start:stop],
            membership=membership,
            active_population=active_population,
            q=q,
            pairs=pairs,
        )
        record = {
            "episode_index": int(segment["episode_index"]),
            "state": int(segment["state"]),
            "state_key": str(segment["state_key"]),
            "active_population": int(active_population),
            "active_population_name": cluster_name,
            "start_bin": start,
            "stop_bin": stop,
            "duration_bins": stop - start,
            "start_time_s": float(segment["start_time"]),
            "stop_time_s": float(segment["stop_time"]),
            "duration_s": float(segment["duration_time"]),
        }
        for key in SCALAR_METRICS + COUNT_METRICS:
            record[key] = stats[key]
        records.append(record)
        for key in VECTOR_METRICS:
            vectors[key].append(np.asarray(stats[key], dtype=np.float32))
    metadata = {
        "connectivity": connectivity,
        "kappa": kappa,
        "network_index": network_index,
        "init_index": init_index,
        "network_seed": network_seed,
        "initialization_seed": int(args.init_seed_start) + network_index * int(args.n_inits) + init_index,
        "trace_path": str(trace_path),
        "bundle_path": str(bundle_path),
        "candidate": candidate,
        "population_names": population_names,
        "sample_dt_seconds": dt,
        "minimum_episode_bins": minimum_bins,
        "n_total_episodes": int(len(canonical.segments)),
        "n_accepted_episodes": int(len(records)),
        "analysis_signature": signature,
    }
    save_payload: dict[str, Any] = {
        "metadata_json": np.array(json.dumps(metadata, default=_json_default, sort_keys=True)),
        "episode_json": np.array(json.dumps(records, default=_json_default, sort_keys=True)),
    }
    for key in VECTOR_METRICS:
        save_payload[key] = (
            np.stack(vectors[key]) if vectors[key] else np.zeros((0, len(population_names)), dtype=np.float32)
        )
    save_payload.update(pairs)
    np.savez_compressed(shard_path, **save_payload)
    return str(shard_path)


def _read_shards(output_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in sorted((output_dir / "shards").glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            episodes = json.loads(str(payload["episode_json"]))
        manifests.append({**metadata, "shard_path": str(path)})
        for episode in episodes:
            rows.append({**metadata, **episode, "shard_path": str(path)})
    return pd.DataFrame(rows), manifests


def _network_balanced(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["connectivity", "kappa", "network_index", "init_index"]
    init_means = rows.groupby(keys, as_index=False)[list(SCALAR_METRICS)].mean()
    return init_means.groupby(["connectivity", "kappa", "network_index"], as_index=False)[list(SCALAR_METRICS)].mean()


def _bootstrap_curve(
    values: np.ndarray, *, fisher: bool, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    transform = np.tanh if fisher else (lambda value: value)
    mean = float(transform(np.mean(arr)))
    if arr.size == 1 or samples <= 0:
        return mean, mean, mean
    indices = rng.integers(0, arr.size, size=(samples, arr.size))
    boot = transform(np.mean(arr[indices], axis=1))
    low, high = np.percentile(boot, [2.5, 97.5])
    return mean, float(low), float(high)


def _plot(rows: pd.DataFrame, args: argparse.Namespace) -> None:
    networks = _network_balanced(rows)
    fig, axes_grid = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[0, 2], axes_grid[1, 0], axes_grid[1, 1]]
    axes_grid[1, 2].axis("off")
    font = FontCfg(base=10, scale=1.1).resolve()
    connectivities = sorted(networks["connectivity"].unique())
    color_map, colorbar_entries = _prepare_value_color_map(connectivities)
    rng = np.random.default_rng(2024)
    styles = {"within": ("-", "o"), "across": ("--", "s"), "active": ("-", "o"), "inactive": ("--", "s")}

    def draw(ax: plt.Axes, metric: str, conn: float, *, label: str, category: str, fisher: bool = False) -> None:
        subset = networks[networks["connectivity"] == conn]
        xs, means, lows, highs = [], [], [], []
        for kappa, group in subset.groupby("kappa"):
            mean, low, high = _bootstrap_curve(
                group[metric].to_numpy(), fisher=fisher, samples=int(args.bootstrap_samples), rng=rng
            )
            xs.append(float(kappa)); means.append(mean); lows.append(low); highs.append(high)
        order = np.argsort(xs)
        x = np.asarray(xs)[order]
        y = np.asarray(means)[order]
        line, marker = styles[category]
        ax.plot(x, y, color=color_map[conn], linestyle=line, marker=marker, markersize=3, label=label)
        ax.fill_between(x, np.asarray(lows)[order], np.asarray(highs)[order], color=color_map[conn], alpha=0.12)

    for conn in connectivities:
        draw(axes[0], "input_all_within_z", conn, label="Within", category="within", fisher=True)
        draw(axes[0], "input_all_across_z", conn, label="Across", category="across", fisher=True)
        draw(axes[1], "output_all_within_z", conn, label="Within", category="within", fisher=True)
        draw(axes[1], "output_all_across_z", conn, label="Across", category="across", fisher=True)
        draw(axes[2], "activity_active", conn, label="Active", category="active")
        draw(axes[3], "input_mean_active", conn, label="Active", category="active")
        draw(axes[3], "input_mean_inactive", conn, label="Non-active", category="inactive")
        if args.plot_decomposed_variance:
            for component, line_category in (("temporal", "active"), ("quenched", "inactive"), ("total", "active")):
                draw(axes[4], f"input_var_{component}_active", conn, label=f"Active {component}", category=line_category)
                draw(axes[4], f"input_var_{component}_inactive", conn, label=f"Non-active {component}", category="inactive")
        else:
            draw(axes[4], "input_var_total_active", conn, label="Active", category="active")
            draw(axes[4], "input_var_total_inactive", conn, label="Non-active", category="inactive")
    labels = [r"$\overline{r_{\mathrm{in}}}$", r"$\overline{r_{\mathrm{out}}}$", r"$m_{\mathrm{active}}$", r"$\mu_{\mathrm{in}}$", r"$\sigma^2_{\mathrm{in}}$"]
    titles = ["Input correlation", "Output correlation", "Active-cluster activity", "Mean input", "Input variance"]
    for ax, ylabel, title in zip(axes, labels, titles):
        ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(ylabel); ax.set_title(title); style_axes(ax, font)
    handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="Within / active"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="Across / non-active"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    if len(colorbar_entries) > 1:
        cax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
        draw_listed_colorbar(fig, cax, colorbar_entries, font_cfg=font, label=r"$\overline{p}$")
    fig.subplots_adjust(
        left=0.08,
        right=0.88 if len(colorbar_entries) > 1 else 0.97,
        bottom=0.1,
        top=0.86,
        wspace=0.38,
        hspace=0.42,
    )
    base = Path(args.output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def aggregate(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    rows, manifests = _read_shards(output_dir)
    if not manifests:
        raise FileNotFoundError(f"No analysis shards found in {output_dir / 'shards'}")
    manifest_frame = pd.DataFrame(manifests)
    if not args.allow_incomplete:
        counts = manifest_frame.groupby(["connectivity", "kappa"]).size()
        expected = int(args.n_networks) * int(args.n_inits)
        incomplete = counts[counts != expected]
        if not incomplete.empty:
            details = ", ".join(f"{key}: {count}/{expected}" for key, count in incomplete.items())
            raise RuntimeError(f"Incomplete condition shards: {details}. Use --allow-incomplete to override.")
    manifest_frame.to_csv(output_dir / "shard_manifest.csv", index=False)
    rows.to_csv(output_dir / "episode_metrics.csv", index=False)
    if rows.empty:
        raise RuntimeError("No eligible single-active-cluster episodes were found.")
    network_rows = _network_balanced(rows)
    network_rows.to_csv(output_dir / "network_balanced_metrics.csv", index=False)
    _plot(rows, args)
    write_yaml_config(vars(args), output_dir / "run_config.yaml")
    return output_dir


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        return aggregate(args)
    if args.n_networks <= 0 or args.n_inits <= 0 or args.duration <= 0:
        raise ValueError("n-networks, n-inits, and duration must be positive")
    parameter = load_from_args(args)
    if parameter.get("R_Eplus") is None:
        parameter["R_Eplus"] = 7.25
    explicit_rj = any(str(value).split("=", 1)[0].strip() == "R_j" for value in args.overwrite)
    if (args.config == "default_simulation" and not explicit_rj) or parameter.get("R_j") is None:
        parameter["R_j"] = 0.79
    focus_counts = helpers.resolve_focus_counts(parameter, parse_int_values(args.focus_counts, option_name="--focus-counts"))
    kappas = resolve_float_values(args.kappas, option_name="--kappas") or []
    connectivity_values = resolve_float_values(args.mean_connectivity, option_name="--mean-connectivity")
    instances = _build_instances(parameter, connectivity_values)
    tasks: list[dict[str, Any]] = []
    network_indices = args.network_indices or list(range(int(args.n_networks)))
    init_indices = args.init_indices or list(range(int(args.n_inits)))
    for instance in instances:
        for kappa in kappas:
            condition_parameter = deepcopy(instance.parameter)
            condition_parameter["kappa"] = float(kappa)
            bundle = _prepare_bundle(condition_parameter, args, focus_counts)
            if args.prepare_only:
                print(f"Prepared {bundle}")
                continue
            for network_index in network_indices:
                for init_index in init_indices:
                    tasks.append(
                        {
                            "args": args,
                            "parameter": condition_parameter,
                            "bundle_path": bundle,
                            "connectivity": instance.value,
                            "focus_counts": focus_counts,
                            "network_index": int(network_index),
                            "init_index": int(init_index),
                        }
                    )
    if args.prepare_only:
        return output_dir
    workers = min(max(1, int(args.jobs)), max(1, len(tasks)))
    if workers == 1:
        for task in tasks:
            print(f"Wrote {_run_shard(task)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for path in pool.map(_run_shard, tasks):
                print(f"Wrote {path}", flush=True)
    if not args.no_aggregate:
        aggregate(args)
    return output_dir


def main() -> None:
    output = run(parse_args())
    print(f"Figure 4 high-state analysis: {output.resolve()}")


if __name__ == "__main__":
    main()
