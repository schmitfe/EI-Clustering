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
    clustering = parameter.get("clustering_type", "probability")
    base = "Weight" if clustering == "weight" else "Probability"
    folder = os.path.join(base, f"R_j{parameter['R_j']}")
    os.makedirs(folder, exist_ok=True)
    return folder


def generate_erf_curve(parameter: dict, start: float = 0.0, end: float = 1.0, step_number: int = 20,
                       clustering_type: str | None = None) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    Compute the ERF for a given parameter dictionary without touching the filesystem.

    Returns the sampled ``v_in``, ``v_out`` pairs and the solver states that seed
    the next iteration.
    """
    clustering = clustering_type or parameter.get("clustering_type", "probability")
    Q = parameter['Q']
    initial = np.ones(2 * Q - 1) * 0.01
    v1_0 = start
    step = (end - start) / max(step_number, 1)

    x_data: List[float] = []
    y_data: List[float] = []
    solves: List[np.ndarray] = []

    while v1_0 <= end + 1e-12:
        print("-----------------------------")
        print(f"Set input rate v_in: {v1_0}")
        x, y, solve = run.simulation(parameter, v1_0, initial, clustering_type=clustering)
        x_data.append(x)
        y_data.append(y)
        solves.append(solve)
        initial = solve
        if step == 0:
            break
        v1_0 = x + step

    return x_data, y_data, solves


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
    clustering_type = parameter['clustering_type']
    print('##############################################################')
    print(f"Simulate Network for R_Eplus = {R_Eplus}")

    foldername = ensure_output_folder(parameter)

    curve = generate_erf_curve(parameter, start=start, end=end, step_number=step_number,
                               clustering_type=clustering_type)
    x_data, y_data, _ = curve

    if plot:
        plot_curve(x_data, y_data, label=str(R_Eplus))
    else:
        print(f"R_Eplus = {R_Eplus} done")

    return serialize_erf(filename, foldername, parameter, curve)
