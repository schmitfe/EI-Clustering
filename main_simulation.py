"""
Entry point for simulating EI-cluster networks across a sweep of ``P_Eplus``.

The shared parameter defaults live in :mod:`simulation_config`; this script only
handles multiprocessing and passing the per-sweep dictionaries to
:func:`safe_data.main`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict

import numpy as np

import safe_data
from simulation_config import parameter_from_env


def _should_plot() -> bool:
    return os.getenv("plot", "False").lower() == "true"


def execute(parameter: Dict) -> str:
    """Run the ERF analysis for a single ``R_Eplus`` configuration."""
    return safe_data.main(
        f"Q{parameter['Q']}R_j{parameter['R_j']}",
        parameter,
        plot=_should_plot(),
    )


if __name__ == "__main__":
    n_cores = int(os.getenv("n_cores", mp.cpu_count()))
    pool = mp.Pool(n_cores)

    base_parameter = parameter_from_env()
    print(base_parameter['R_j'])
    print(base_parameter['kappa'])

    # example sweep; customize as needed or replace with a list of values
    R_Eplus_values = np.arange(1, 20.01, 0.2)

    parameter_list = []
    for value in R_Eplus_values:
        entry = dict(base_parameter)
        entry["R_Eplus"] = float(value)
        parameter_list.append(entry)

    result = pool.map(execute, parameter_list)
