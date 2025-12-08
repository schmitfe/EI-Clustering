"""
Helper utilities for constructing simulation parameter dictionaries.

The numerical pipeline expects plain dicts, so this module provides helpers to
materialize the commonly used defaults and optionally override a subset of
values via environment variables.  Downstream code should copy the returned
dicts before mutating them so each simulation keeps an isolated parameter set.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

DEFAULT_PARAMETER: Dict[str, float | int | None | str] = dict(
    N=5000,
    N_E=4000,
    N_I=1000,
    Q=20,
    V_th=1,
    g=1.2,
    p0_ee=0.1,
    p0_ie=0.1,
    p0_ei=0.1,
    p0_ii=0.1,
    tau_e=10,
    tau_i=5,
    R_Eplus=None,
    R_j=None,
    kappa=0.0,
    connection_type="bernoulli",
    m_X=0.03,
)


def base_parameters() -> Dict[str, float | int | None | str]:
    """Return a shallow copy of the default parameter dictionary."""
    return dict(DEFAULT_PARAMETER)


def parameter_from_env(overrides: Optional[Dict[str, float | int | None | str]] = None) -> Dict[str, float | int | None | str]:
    """
    Build a parameter dictionary that honors environment overrides.

    Recognized environment variables
    --------------------------------
    Rj:
        Sets ``parameter["R_j"]`` and defaults to 0 when unset.
    kappa:
        Sets the mixing coefficient between probability (0) and weight (1)
        clustering.
    connection_type:
        Chooses how connection variance is computed:
        ``"bernoulli"`` (default), ``"poisson"``, or ``"fixed-indegree"``.
    R_Eplus:
        Allows single-configuration runs without editing the source.
    """
    parameter = base_parameters()
    if overrides:
        parameter.update(overrides)

    env_rj = os.getenv("Rj")
    if env_rj is not None:
        parameter["R_j"] = float(env_rj)

    env_kappa = os.getenv("kappa")
    if env_kappa is not None:
        parameter["kappa"] = float(env_kappa)

    env_connection = os.getenv("connection_type")
    if env_connection:
        parameter["connection_type"] = env_connection

    env_replus = os.getenv("R_Eplus")
    if env_replus is not None:
        parameter["R_Eplus"] = float(env_replus)

    if parameter["R_j"] is None:
        parameter["R_j"] = 0.0

    return parameter


__all__ = ["DEFAULT_PARAMETER", "base_parameters", "parameter_from_env"]
