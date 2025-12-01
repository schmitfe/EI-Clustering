"""
This module solves the system by iterative solving methods for a given input value v_in and returns v_out
"""
import numpy as np

from rate_system import RateSystem


def simulation(parameter, v1_0, initial, clustering_type=None):
    """
    Recursive function that calculates the output rate for a given input.
    If no solution can be found for the given input that is within the error tolerance,
    the function is recursively called again and the input is slightly increased until a solution is found.

    :param parameter: dictionary with the set parameter
    :param v1_0: fixed rate of cluster 1
    :param initial: start-value for the solving method (last solve)
    :return: (v1_0, v1_out, solve), where v1_0: input,
                                        v1_out: output for the given input # v1_out = f(solve)
                                         solve: found solve of the system
    """
    Q = parameter['Q']
    dim = 2 * Q - 1
    clustering = clustering_type or "probability"

    rate_system = RateSystem(parameter, v1_0, clustering_type=clustering)
    initial_vec = np.asarray(initial, dtype=float).reshape((dim,))

    solve, value, success = rate_system.solve(initial_vec)

    if success:
        phi_values = rate_system.phi_numpy(solve)
        v1_out = phi_values[0]
        print("Funktionswerte Phi:")
        for val in phi_values:
            print(val)
        return (v1_0, v1_out, solve)
    else:
        print("Die Lösung des Lösungsverfahren liegt für v_in = " + str(v1_0) + " außerhalb der Toleranzgrenze. "
            "Deshalb wird alternativ die Berechnung für v_in = " + str(v1_0 + 0.005) + " wiederholt.")
        return simulation(parameter, v1_0 + 0.005, solve, clustering_type=clustering_type)
