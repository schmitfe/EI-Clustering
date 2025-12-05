"""
This module solves the system by iterative solving methods for a given input value v_in and returns v_out
"""
import numpy as np

from rate_system import RateSystem


def simulation(parameter, v1_0, initial, kappa=None):
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
    mixing = kappa if kappa is not None else parameter.get("kappa", 0.0)
    rate_system = RateSystem(parameter, v1_0, kappa=mixing)
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
        if v1_0 < 0.995:
            return simulation(parameter, v1_0 + 0.005, solve, kappa=kappa)
        else:
            raise

if __name__ == "__main__":
    v10 =0.5
    parameter = {
        "Q": 20,
        "kappa": 0.0,
        "tau_e": 20.,
        "tau_i": 10.,
        "N": 5000,
        "N_E": 4000,
        "N_I": 1000,
        "V_th": 1.,
        "g": 1.2,
        "p0_ee": 0.1,
        "p0_ii": 0.2,
        "p0_ie": 0.2,
        "p0_ei": 0.2,
        "m_X": 0.03,
        "R_Eplus": 9.,
        "R_j": 0.78,
    }

    initial = [0.2] *(2*parameter["Q"]-1)
    Result = simulation(parameter, v10, initial=initial)
    print(initial)
