#!/usr/bin/env python3
"""Run independently randomized BrainScaleS-2 clustered networks.

This script is intended to run in the BrainScaleS-2 Jupyter/container
environment used by Simulate_kappa.ipynb.  It writes one pickle per trial and
updates manifest.json after every trial, so an interrupted 100-network run can
be resumed safely.

Raw spikes, cluster-sorted raster plots, connectivity hashes, cluster
assignments, and hardware placement are retained for later inspection.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time

import numpy as np

DEFAULT_NET = {
    "neuron_type": "iaf_psc_exp",
    "E_L": 0.0,
    "C_m": 1.0,
    "tau_E": 20.0,
    "tau_I": 10.0,
    "t_ref": 5.0,
    "V_th_E": 20.0,
    "V_th_I": 20.0,
    "V_r": 0.0,
    "tau_syn_ex": 5.0,
    "tau_syn_in": 5.0,
    "delay": 0.1,
    "I_th_E": 1.05,
    "I_th_I": 0.95,
    "delta_I_xE": 0.0,
    "delta_I_xI": 0.0,
    "V_m": "rand",
    "N_E": 240,
    "N_I": 60,
    "n_clusters": 6,
    "baseline_conn_prob": 0.15 * np.ones((2, 2)),
    "gei": 1.2,
    "gie": 1.0,
    "gii": 1.0,
    "s": 1.0,
    "fixed_indegree": True,
    "kappa": 0.0,
    "rj": 0.65,
    "rep": 4.5,
}

DEFAULT_STIM = {
    "stim_clusters": None,
    "stim_amp": 0.00015,
    "stim_starts": [2000, 6000],
    "stim_ends": [3500, 7500],
}


def parse_args(argv=None):
    """Parse command-line options, or an explicit notebook argument list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-N", "--networks", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--warmup-ms", type=float, default=80.0)
    parser.add_argument("--simtime-ms", type=float, default=100.0)
    parser.add_argument(
        "--raster-window-ms",
        type=float,
        default=6.0,
        help="Only pass spikes in this initial time window to Matplotlib.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("bs2_metastability_runs"))
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=1700.0,
        help="Stop cleanly after a trial if process RSS reaches this value; 0 disables.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-rasters",
        action="store_true",
        help="Do not save one cluster-sorted raster PNG per completed trial.",
    )
    return parser.parse_args(argv)


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(_jsonable(data), handle, indent=2)
    os.replace(temporary, path)


def _current_rss_mb():
    """Return current resident memory on Linux, or None when unavailable."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _placement_snapshot(net, sim_module):
    """Capture mapper output for parent populations and cluster views."""
    state = sim_module.simulator.state
    placement = getattr(state, "neuron_placement", None)
    result = {"repr": repr(placement), "parents": {}, "views": {}}
    if placement is None:
        return result

    for parent in net._parent_populations:
        try:
            coords = placement.id2first_circuit(list(parent.all_cells))
            result["parents"][parent.label] = [int(coord) for coord in coords]
        except Exception as exc:
            result["parents"][parent.label] = {"error": repr(exc)}
    for views in net._populations:
        for view in views:
            try:
                coords = placement.id2first_circuit(list(view.all_cells))
                result["views"][view.label] = [int(coord) for coord in coords]
            except Exception as exc:
                result["views"][view.label] = {"error": repr(exc)}
    return result


def fingerprint_realized_connectivity(net):
    """Hash realized PyNN connections without retaining their edge lists.

    The topology digest includes projection labels, source/target indices, and
    duplicate rows (multapses).  The weighted digest additionally includes the
    exact hexadecimal representation of each returned weight.  Sorting makes
    both digests independent of the order in which the backend returns rows.
    """
    topology_all = hashlib.sha256()
    weighted_all = hashlib.sha256()
    projections = {}
    total_connections = 0

    ordered_projections = sorted(
        net._projections,
        key=lambda projection: str(getattr(projection, "label", "")),
    )
    for projection in ordered_projections:
        label = str(getattr(projection, "label", ""))
        try:
            try:
                raw = projection.get(
                    ["weight"],
                    format="list",
                    with_address=True,
                )
            except TypeError:
                # Compatibility with PyNN versions expecting a scalar name.
                raw = projection.get(
                    "weight",
                    format="list",
                    with_address=True,
                )

            rows = np.asarray(raw)
            if rows.size == 0:
                rows = np.empty((0, 3))
            elif rows.ndim == 1:
                rows = rows.reshape(1, -1)
            if rows.ndim != 2 or rows.shape[1] < 3:
                raise RuntimeError(
                    f"Unexpected Projection.get result shape {rows.shape}"
                )

            canonical_rows = sorted(
                (int(row[0]), int(row[1]), float(row[2]).hex())
                for row in rows
            )
            topology_projection = hashlib.sha256()
            weighted_projection = hashlib.sha256()
            label_record = (label + "\0").encode("utf-8")
            topology_all.update(label_record)
            weighted_all.update(label_record)
            for source, target, weight_hex in canonical_rows:
                topology_record = f"{source}\0{target}\n".encode("ascii")
                weighted_record = (
                    f"{source}\0{target}\0{weight_hex}\n".encode("ascii")
                )
                topology_projection.update(topology_record)
                weighted_projection.update(weighted_record)
                topology_all.update(topology_record)
                weighted_all.update(weighted_record)

            count = len(canonical_rows)
            total_connections += count
            projections[label] = {
                "count": count,
                "topology_sha256": topology_projection.hexdigest(),
                "weighted_sha256": weighted_projection.hexdigest(),
            }
        except Exception as exc:
            return {
                "available": False,
                "error": f"{label}: {exc!r}",
                "total_connections": None,
                "topology_sha256": None,
                "weighted_sha256": None,
                "projections": projections,
            }

    return {
        "available": True,
        "error": None,
        "total_connections": total_connections,
        "topology_sha256": topology_all.hexdigest(),
        "weighted_sha256": weighted_all.hexdigest(),
        "projections": projections,
    }


def save_cluster_sorted_raster(
    path,
    spikes,
    *,
    n_e,
    n_i,
    n_clusters,
    raster_window_ms,
):
    """Plot remapped cluster-major IDs, with cluster boundaries made explicit."""
    import matplotlib.pyplot as plt

    # Filtering before plotting is essential: setting xlim alone still creates
    # Matplotlib artists containing every spike from the full simulation.
    plot_spikes = spikes[:, spikes[0] <= raster_window_ms]
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        excitatory = plot_spikes[1] < n_e
        ax.plot(
            plot_spikes[0, excitatory],
            plot_spikes[1, excitatory],
            ".",
            color="black",
            markersize=1.0,
            rasterized=True,
            label="E",
        )
        ax.plot(
            plot_spikes[0, ~excitatory],
            plot_spikes[1, ~excitatory],
            ".",
            color="darkred",
            markersize=1.0,
            rasterized=True,
            label="I",
        )
        e_cluster_size = n_e // n_clusters
        i_cluster_size = n_i // n_clusters
        for cluster in range(1, n_clusters):
            ax.axhline(
                cluster * e_cluster_size - 0.5,
                color="0.75",
                linewidth=0.4,
            )
            ax.axhline(
                n_e + cluster * i_cluster_size - 0.5,
                color="#d9aaaa",
                linewidth=0.4,
            )
        ax.axhline(n_e - 0.5, color="darkred", linewidth=0.8)
        ax.set(
            xlim=(0, raster_window_ms),
            ylim=(-0.5, n_e + n_i - 0.5),
        )
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Cluster-major neuron ID")
        ax.legend(loc="upper right", markerscale=5, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        if fig is not None:
            plt.close(fig)
        del plot_spikes


def main(argv=None):
    """Run the ensemble and return its summary.

    ``argv`` may be a list of command-line-style strings when called from a
    notebook.  With ``None`` the normal process command line is used.
    """
    args = parse_args(argv)
    if args.networks <= 0:
        raise ValueError("--networks must be positive")
    if not 0.0 <= args.kappa <= 1.0:
        raise ValueError("--kappa must be in [0, 1]")
    if args.simtime_ms <= 0 or args.raster_window_ms <= 0:
        raise ValueError("--simtime-ms and --raster-window-ms must be positive")
    if args.max_rss_mb < 0:
        raise ValueError("--max-rss-mb cannot be negative")

    # Hardware-only imports are delayed so --help and utility functions remain
    # usable in an ordinary development environment.
    sys.path.append("/opt/app-root/src/brainscales2-demos.git")
    from _static.common.helpers import get_nightly_calibration, setup_hardware_client
    import pynn_brainscales.brainscales2 as sim
    from network_views import ViewClusteredNetwork

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.json"
    manifest = []
    if args.resume and manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
    by_index = {int(row["trial_index"]): row for row in manifest}

    experiment_configuration = {
        "base_seed": args.base_seed,
        "kappa": args.kappa,
        "warmup_ms": args.warmup_ms,
        "simtime_ms": args.simtime_ms,
        "raster_window_ms": args.raster_window_ms,
        "net": _jsonable(DEFAULT_NET),
        "stim": _jsonable(DEFAULT_STIM),
    }
    configuration_hash = hashlib.sha256(
        json.dumps(experiment_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mismatched = [
        row["trial_index"]
        for row in manifest
        if row.get("status") == "ok"
        and row.get("configuration_hash") != configuration_hash
    ]
    if args.resume and mismatched:
        raise RuntimeError(
            "Refusing to mix a changed configuration into the existing output "
            f"directory; mismatched completed trials: {mismatched[:10]}"
        )

    setup_hardware_client()
    # A Chip calibration object is large.  Loading a new pybind object for every
    # trial caused resident memory to grow in the constrained notebook process.
    calibration = get_nightly_calibration()
    sim_base = {
        "warmup": float(args.warmup_ms),
        "simtime": float(args.simtime_ms),
        "dt": 0.1,
        "n_vp": 4,
    }

    stopped_for_memory = False
    for trial_index in range(args.networks):
        outfile = args.outdir / f"network_{trial_index:03d}.pkl"
        rasterfile = args.outdir / f"network_{trial_index:03d}_raster.png"
        if (
            args.resume
            and trial_index in by_index
            and by_index[trial_index].get("status") == "ok"
            and outfile.exists()
        ):
            print(f"skip {trial_index:03d}: already complete")
            continue

        trial_seed = int(args.base_seed + trial_index)
        connector_seed, assignment_seed = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in np.random.SeedSequence(trial_seed).spawn(2)
        ]
        sim_dict = dict(sim_base, randseed=connector_seed)
        net_dict = copy.deepcopy(DEFAULT_NET)
        net_dict.update(kappa=float(args.kappa), cluster_assignment_seed=assignment_seed)
        started = datetime.now(timezone.utc)
        wall_start = time.time()
        row = {
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "connector_seed": connector_seed,
            "cluster_assignment_seed": assignment_seed,
            "configuration_hash": configuration_hash,
            "status": "running",
            "pickle_file": str(outfile),
            "utc_start": started.isoformat(),
            "rss_before_mb": _current_rss_mb(),
        }
        by_index[trial_index] = row
        _atomic_json(manifest_path, [by_index[index] for index in sorted(by_index)])

        net = None
        result = None
        connectivity = None
        placement = None
        payload = None
        try:
            sim_dict["initial_config"] = calibration
            net = ViewClusteredNetwork(sim_dict, net_dict, copy.deepcopy(DEFAULT_STIM))
            result = net.get_simulation()
            connectivity = fingerprint_realized_connectivity(net)
            if not connectivity["available"]:
                print(
                    "warning: realized connectivity hashing unavailable: "
                    f"{connectivity['error']}",
                    file=sys.stderr,
                )
            placement = _placement_snapshot(net, sim)
            parent_placement_fingerprint = hashlib.sha256(
                json.dumps(placement["parents"], sort_keys=True).encode("utf-8")
            ).hexdigest()
            view_placement_fingerprint = hashlib.sha256(
                json.dumps(placement["views"], sort_keys=True).encode("utf-8")
            ).hexdigest()
            payload = {
                "trial_index": trial_index,
                "trial_seed": trial_seed,
                "connector_seed": connector_seed,
                "cluster_assignment_seed": assignment_seed,
                "spiketimes": result["spiketimes"],
                "e_rate_hz": float(result["e_rate"]),
                "i_rate_hz": float(result["i_rate"]),
                "realized_connectivity": connectivity,
                "cluster_membership_parent_indices": net.get_cluster_membership(),
                "recording_id_conventions": dict(net._recording_id_conventions),
                "placement": placement,
                "net_dict": _jsonable(net_dict),
                # Do not serialize/repr the roughly 10 MB calibration object in
                # every run.  It is identified by this explicit provenance tag.
                "sim_dict": {
                    **_jsonable(sim_base),
                    "randseed": connector_seed,
                    "initial_config": "nightly_calibration_reused_for_ensemble",
                },
                "stim_dict": _jsonable(DEFAULT_STIM),
                "run_metadata": {
                    "hostname": socket.gethostname(),
                    "utc_start": started.isoformat(),
                    "utc_end": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - wall_start,
                },
            }
            with outfile.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if not args.no_rasters:
                save_cluster_sorted_raster(
                    rasterfile,
                    result["spiketimes"],
                    n_e=net_dict["N_E"],
                    n_i=net_dict["N_I"],
                    n_clusters=net_dict["n_clusters"],
                    raster_window_ms=min(
                        args.raster_window_ms,
                        sim_dict["simtime"],
                    ),
                )

            row.update(
                status="ok",
                e_rate_hz=float(result["e_rate"]),
                i_rate_hz=float(result["i_rate"]),
                connectivity_hash_available=bool(connectivity["available"]),
                realized_connection_count=connectivity["total_connections"],
                connectivity_topology_sha256=connectivity["topology_sha256"],
                connectivity_weighted_sha256=connectivity["weighted_sha256"],
                connectivity_hash_error=connectivity["error"],
                parent_placement_fingerprint=parent_placement_fingerprint,
                view_placement_fingerprint=view_placement_fingerprint,
                raster_file=None if args.no_rasters else str(rasterfile),
                elapsed_seconds=time.time() - wall_start,
                utc_end=datetime.now(timezone.utc).isoformat(),
            )
            print(
                f"trial {trial_index:03d} seed={trial_seed}: "
                f"E={row['e_rate_hz']:.2f}Hz I={row['i_rate_hz']:.2f}Hz"
            )
        except Exception as exc:
            row.update(
                status="failed",
                error=repr(exc),
                elapsed_seconds=time.time() - wall_start,
                utc_end=datetime.now(timezone.utc).isoformat(),
            )
            print(f"trial {trial_index:03d} failed: {exc!r}", file=sys.stderr)
            if args.fail_fast:
                raise
        finally:
            try:
                sim.end()
            except Exception:
                pass
            # Drop all references into Neo, PyNN, grenade, and the large spike
            # arrays before continuing.  CPython cyclic collection is forced so
            # the manifest's RSS value reflects post-trial memory.
            sim_dict.pop("initial_config", None)
            net = None
            result = None
            connectivity = None
            placement = None
            payload = None
            gc.collect()
            row["rss_after_cleanup_mb"] = _current_rss_mb()
            if (
                row["rss_before_mb"] is not None
                and row["rss_after_cleanup_mb"] is not None
            ):
                row["rss_growth_mb"] = (
                    row["rss_after_cleanup_mb"] - row["rss_before_mb"]
                )
            print(
                f"trial {trial_index:03d} cleanup RSS: "
                f"{row['rss_after_cleanup_mb']} MB "
                f"(trial growth {row.get('rss_growth_mb')} MB)"
            )
            _atomic_json(manifest_path, [by_index[index] for index in sorted(by_index)])

        rss_after = row.get("rss_after_cleanup_mb")
        if args.max_rss_mb and rss_after is not None and rss_after >= args.max_rss_mb:
            stopped_for_memory = True
            print(
                f"stopping cleanly at {rss_after:.1f} MB RSS before the "
                f"{args.max_rss_mb:.1f} MB limit; restart the kernel and use --resume"
            )
            break

    requested_rows = [by_index[index] for index in range(args.networks) if index in by_index]
    completed = [row for row in requested_rows if row.get("status") == "ok"]
    cleanup_rss = [
        row["rss_after_cleanup_mb"]
        for row in requested_rows
        if row.get("rss_after_cleanup_mb") is not None
    ]
    summary = {
        "requested_networks": args.networks,
        "completed_networks": len(completed),
        "failed_networks": sum(row.get("status") == "failed" for row in requested_rows),
        "connectivity_hash_failures": sum(
            not row.get("connectivity_hash_available", False) for row in completed
        ),
        "distinct_realized_connectivity_topologies": len(
            {
                row["connectivity_topology_sha256"]
                for row in completed
                if row.get("connectivity_topology_sha256") is not None
            }
        ),
        "distinct_parent_placements": len(
            {
                row["parent_placement_fingerprint"]
                for row in completed
                if row.get("parent_placement_fingerprint") is not None
            }
        ),
        "distinct_cluster_view_placements": len(
            {
                row["view_placement_fingerprint"]
                for row in completed
                if row.get("view_placement_fingerprint") is not None
            }
        ),
        "configuration_hash": configuration_hash,
        "experiment_configuration": experiment_configuration,
        "configuration": _jsonable(vars(args)),
        "stopped_for_memory": stopped_for_memory,
        "peak_post_cleanup_rss_mb": max(
            cleanup_rss,
            default=None,
        ),
        "post_cleanup_rss_change_mb": (
            cleanup_rss[-1] - cleanup_rss[0] if len(cleanup_rss) >= 2 else 0.0
        ),
    }
    _atomic_json(args.outdir / "summary.json", summary)
    print(
        f"completed networks: {len(completed)}/{args.networks}; "
        f"summary: {args.outdir / 'summary.json'}"
    )
    return summary


if __name__ == "__main__":
    main()
