"""
This module solves the system by iterative solving methods for a given input value v_in and returns v_out
"""
from rate_system import RateSystem


def simulation(parameter, v1_0, initial=None, kappa=None, connection_type=None):
    """
    Solve the rate equations for a given input rate.  When the solver does not
    converge at ``v_in``, the helper incrementally increases the input and
    retries.  If no configuration converges within the allowed range, ``None``
    is returned so callers can still persist previously converged ERF values.

    :param parameter: dictionary with the set parameter
    :param v1_0: fixed rate of cluster 1
    :param initial: start-value for the solving method (last solve)
    :return: Tuple (v_in, v_out, solve) if convergence is achieved, otherwise None
    """
    mixing = kappa if kappa is not None else parameter.get("kappa", 0.0)
    conn_kind = connection_type or parameter.get("connection_type", "bernoulli")
    retry_step = 0.005
    max_input = 1.0 + retry_step
    current_input = float(v1_0)
    current_initial = initial

    while current_input <= max_input + 1e-9:
        rate_system = RateSystem(parameter, current_input, kappa=mixing, connection_type=conn_kind)
        solve, value, success = rate_system.solve(current_initial)
        if success:
            phi_values = rate_system.phi_numpy(solve)
            v1_out = phi_values[0]
            print("Funktionswerte Phi:")
            for val in phi_values:
                print(val)
            return (current_input, v1_out, solve)

        next_input = current_input + retry_step
        print(
            "Die Lösung des Lösungsverfahren liegt für v_in = "
            + str(current_input)
            + " außerhalb der Toleranzgrenze. Deshalb wird alternativ die Berechnung für v_in = "
            + str(next_input)
            + " wiederholt."
        )
        if next_input > max_input:
            break
        current_input = next_input
        current_initial = solve

    print("Abbruch: Für die verbleibenden Eingabewerte konnte keine Konvergenz erreicht werden.")
    return None

if __name__ == "__main__":
    v10 =0.5
    parameter = {
        "Q": 20,
        "kappa": 0.0,
        "connection_type": "bernoulli",
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
