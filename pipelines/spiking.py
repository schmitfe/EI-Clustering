from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import sys

# pipelines/spiking.py lives one level below the repository root.
# The NEST reference code uses plain module imports such as `import helper`,
# so we need to add the actual NEST module directory itself to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
NEST_MODULE_DIR = REPO_ROOT / "NEST" / "EI_clustered_network"
if str(NEST_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(NEST_MODULE_DIR))

from network import ClusteredNetwork
from network_params import net_dict as DEFAULT_NET
from sim_params import sim_dict as DEFAULT_SIM
from stimulus_params import stim_dict as DEFAULT_STIM
from MeanField.rate_system import ensure_output_folder
from sim_config import add_override_arguments, deep_update, load_from_args, sim_tag_from_cfg, write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the clustered spiking EI network using YAML configs and overrides."
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--save-spikes",
        type=str,
        help="Optional path to an .npz file storing spike times, rates, and parameters.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for the repository-managed spiking output files.",
    )
    return parser.parse_args()


def _build_spiking_dicts(parameter: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base_sim = deepcopy(DEFAULT_SIM)
    base_net = deepcopy(DEFAULT_NET)
    base_stim = deepcopy(DEFAULT_STIM)

    spiking_cfg = parameter.get("spiking") or {}
    if not isinstance(spiking_cfg, dict):
        raise TypeError("parameter['spiking'] must be a mapping when provided.")

    sim_override = spiking_cfg.get("sim") or {}
    net_override = spiking_cfg.get("net") or {}
    stim_override = spiking_cfg.get("stim") or {}
    if not isinstance(sim_override, dict):
        raise TypeError("spiking.sim overrides must be provided as a mapping.")
    if not isinstance(net_override, dict):
        raise TypeError("spiking.net overrides must be provided as a mapping.")
    if not isinstance(stim_override, dict):
        raise TypeError("spiking.stim overrides must be provided as a mapping.")

    sim_cfg = deep_update(base_sim, sim_override)
    net_cfg = deep_update(base_net, net_override)
    stim_cfg = deep_update(base_stim, stim_override)

    if "kappa" in parameter and parameter["kappa"] is not None:
        net_cfg["kappa"] = float(parameter["kappa"])
    return sim_cfg, net_cfg, stim_cfg


def run_spiking_simulation(parameter: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate and run the NEST ClusteredNetwork with resolved parameters."""

    sim_cfg, net_cfg, stim_cfg = _build_spiking_dicts(parameter)
    network = ClusteredNetwork(sim_cfg, net_cfg, stim_cfg)
    result = network.get_simulation()
    return {
        "spiketimes": result["spiketimes"],
        "e_rate": float(result["e_rate"]),
        "i_rate": float(result["i_rate"]),
        "params": result["_params"],
        "sim_dict": sim_cfg,
        "net_dict": net_cfg,
        "stim_dict": stim_cfg,
    }


def _save_npz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        spiketimes=payload["spiketimes"],
        e_rate=payload["e_rate"],
        i_rate=payload["i_rate"],
        params=payload["params"],
        sim_dict=np.array(payload["sim_dict"], dtype=object),
        net_dict=np.array(payload["net_dict"], dtype=object),
        stim_dict=np.array(payload["stim_dict"], dtype=object),
    )


def _taggable_root_parameter(parameter: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(parameter)
    for key in ("R_Eplus", "focus_count", "focus_counts"):
        filtered.pop(key, None)
    return filtered


def _taggable_spiking_config(parameter: Dict[str, Any], payload: Dict[str, Any], output_name: str) -> Dict[str, Any]:
    return {
        "parameter": dict(parameter),
        "spiking": {
            "sim_dict": dict(payload["sim_dict"]),
            "net_dict": dict(payload["net_dict"]),
            "stim_dict": dict(payload["stim_dict"]),
            "output_name": str(output_name),
        },
    }


def save_spiking_output(parameter: Dict[str, Any], payload: Dict[str, Any], *, output_name: str | None = None) -> Dict[str, str]:
    root_tag = sim_tag_from_cfg(_taggable_root_parameter(parameter))
    folder = ensure_output_folder(parameter, tag=root_tag)
    spiking_cfg = dict(parameter.get("spiking") or {})
    resolved_output_name = str(output_name or spiking_cfg.get("output_name") or "spike_trace")
    spiking_tag = sim_tag_from_cfg(_taggable_spiking_config(parameter, payload, resolved_output_name))
    spiking_folder = os.path.join(folder, "spiking", spiking_tag)
    os.makedirs(spiking_folder, exist_ok=True)
    params_path = os.path.join(folder, "params.yaml")
    if not os.path.exists(params_path):
        write_yaml_config(_taggable_root_parameter(parameter), params_path)
    spiking_params_path = os.path.join(spiking_folder, "params.yaml")
    if not os.path.exists(spiking_params_path):
        write_yaml_config(
            {
                "parameter": parameter,
                "resolved_spiking": {
                    "sim_dict": payload["sim_dict"],
                    "net_dict": payload["net_dict"],
                    "stim_dict": payload["stim_dict"],
                },
            },
            spiking_params_path,
        )
    trace_path = os.path.join(spiking_folder, f"{resolved_output_name}.npz")
    _save_npz(Path(trace_path), payload)
    summary = {
        "output_name": resolved_output_name,
        "e_rate": float(payload["e_rate"]),
        "i_rate": float(payload["i_rate"]),
        "spike_count": int(np.asarray(payload["spiketimes"]).shape[1]),
        "simtime_ms": float(payload["sim_dict"].get("simtime", 0.0)),
        "dt_ms": float(payload["sim_dict"].get("dt", 0.0)),
    }
    summary_path = os.path.join(spiking_folder, f"{resolved_output_name}_summary.yaml")
    write_yaml_config(summary, summary_path)
    return {
        "spiking_folder": spiking_folder,
        "trace_path": trace_path,
        "summary_path": summary_path,
        "output_name": resolved_output_name,
    }


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    result = run_spiking_simulation(parameter)
    saved = save_spiking_output(parameter, result, output_name=args.output_name)
    print(
        f"Simulated clustered spiking network: e_rate={result['e_rate']:.2f} Hz, "
        f"i_rate={result['i_rate']:.2f} Hz"
    )
    print(f"Stored spiking output at {saved['trace_path']}")
    if args.save_spikes:
        _save_npz(Path(args.save_spikes), result)
    analysis_cfg = dict(parameter.get("analysis") or {})
    if bool(analysis_cfg.get("enabled", False)):
        from analysis.pipeline import run_analysis_on_spiking_payload

        analysis_result = run_analysis_on_spiking_payload(
            result,
            parameter=parameter,
            analysis_cfg=analysis_cfg,
            base_output_dir=saved["spiking_folder"],
        )
        print(f"Stored analysis outputs at {analysis_result['output_dir']}")


if __name__ == "__main__":
    main()
