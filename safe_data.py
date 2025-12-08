from __future__ import annotations

import os
import pickle
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import run

"""
In this module, the dynamics of the cluster network is saved.
The effect of sweeping a given parameter (``P_Eplus`` or ``R_j``) is considered.

``generate_erf_curve`` only computes the Effective Response Function (ERF) and
returns the sampled data so the caller decides how to persist or plot it.  The
``main`` helper keeps backward compatibility for the original scripts by
handling plotting and pickle generation on top of that pure computation step.
"""


def ensure_output_folder(parameter: dict) -> str:
    """Return (and create) the folder that should contain the sweep result."""
    kappa = float(parameter.get("kappa", 0.0))
    conn_kind = str(parameter.get("connection_type", "bernoulli")).lower().replace(" ", "_")
    base = os.path.join(conn_kind, f"Kappa_{kappa:.2f}".replace(".", "_"))
    folder = os.path.join(base, f"R_j{parameter['R_j']}")
    os.makedirs(folder, exist_ok=True)
    return folder


def generate_erf_curve(parameter: dict, start: float = 0.0, end: float = 1.0, step_number: int = 20,
                       mixing_parameter: float | None = None, connection_type: str | None = None) -> Tuple[Tuple[List[float], List[float], List[np.ndarray]], bool]:
    """
    Compute the ERF for a given parameter dictionary without touching the filesystem.

    Returns the sampled ``v_in``, ``v_out`` pairs and the solver states that seed
    the next iteration.
    """
    mixing = mixing_parameter if mixing_parameter is not None else parameter.get("kappa", 0.0)
    conn_kind = str(connection_type or parameter.get("connection_type", "bernoulli"))
    initial = None
    v1_0 = start
    step = (end - start) / max(step_number, 1)

    x_data: List[float] = []
    y_data: List[float] = []
    solves: List[np.ndarray] = []

    aborted = False
    while v1_0 <= end + 1e-12:
        print("-----------------------------")
        print(f"Set input rate v_in: {v1_0}")
        result = run.simulation(parameter, v1_0, initial, kappa=mixing, connection_type=conn_kind)
        if result is None:
            print("Solver did not converge for the remaining inputs. Stopping sweep early.")
            aborted = True
            break
        x, y, solve = result
        x_data.append(x)
        y_data.append(y)
        solves.append(solve)
        initial = solve
        if step == 0:
            break
        v1_0 = x + step

    completed = not aborted and (step == 0 or v1_0 > end + 1e-12)
    return (x_data, y_data, solves), completed


def serialize_erf(filename: str, folder: str, parameter: dict,
                  curve: Sequence[Sequence[float | np.ndarray]]) -> str:
    """Persist ERF data for a single ``R_Eplus`` value and return the file path."""
    x_data, y_data, solves = curve
    re_key = str(parameter["R_Eplus"])
    payload = {re_key: [x_data, y_data, solves, parameter]}
    encoded_re = f"{parameter['R_Eplus']:.2f}".replace(".", "_")
    path = os.path.join(folder, f"{filename}R_Eplus{encoded_re}.pkl")
    with open(path, "wb") as file:
        pickle.dump(payload, file)
    return path


def plot_curve(x_data: Sequence[float], y_data: Sequence[float], label: str) -> None:
    plt.plot(x_data, y_data, label=label)
    plt.legend()
    plt.plot([0, 1], [0, 1], "black")
    plt.show()


def main(filename, parameter, plot=False, start=0., end=1., step_number=1000):
    """
    Solve the ERF for ``parameter`` and persist the results as ``filename``.
    """
    R_Eplus = parameter['R_Eplus']
    mixing_parameter = parameter.get('kappa', 0.0)
    connection_type = parameter.get('connection_type', 'bernoulli')
    print('##############################################################')
    print(f"Simulate Network for R_Eplus = {R_Eplus}, kappa = {mixing_parameter}, connection_type = {connection_type}")

    foldername = ensure_output_folder(parameter)

    curve, completed = generate_erf_curve(parameter, start=start, end=end, step_number=step_number,
                                          mixing_parameter=mixing_parameter, connection_type=connection_type)
    x_data, y_data, _ = curve

    if plot:
        plot_curve(x_data, y_data, label=str(R_Eplus))
    else:
        print(f"R_Eplus = {R_Eplus} done")

    if not completed:
        print("Skipping serialization because the ERF sweep did not fully converge.")
        return None

    return serialize_erf(filename, foldername, parameter, curve)
