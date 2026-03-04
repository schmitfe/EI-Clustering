from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent
NEST_MODULE_DIR = REPO_ROOT / "NEST" / "EI_clustered_network"
if str(NEST_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(NEST_MODULE_DIR))

from network import ClusteredNetwork
from network_params import net_dict as DEFAULT_NET
from sim_params import sim_dict as DEFAULT_SIM
from stimulus_params import stim_dict as DEFAULT_STIM
from sim_config import add_override_arguments, deep_update, load_from_args


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
    )


def main() -> None:
    args = parse_args()
    parameter = load_from_args(args)
    result = run_spiking_simulation(parameter)
    print(
        f"Simulated clustered spiking network: e_rate={result['e_rate']:.2f} Hz, "
        f"i_rate={result['i_rate']:.2f} Hz"
    )
    if args.save_spikes:
        _save_npz(Path(args.save_spikes), result)


if __name__ == "__main__":
    main()
