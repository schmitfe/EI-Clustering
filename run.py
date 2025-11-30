"""
This modules solves the system by iterative solving methods for a given input value v_in and returns v_out
"""
import numpy as np
import sympy
from sympy import *
import connectivit
import newton
import halley


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
    #get parameter
    Q = parameter['Q']
    tau_e = parameter["tau_e"]
    tau_i = parameter["tau_i"]
    tau = np.ones((2*Q))
    tau[:int(len(tau) / 2)] *= tau_e
    tau[int(len(tau) / 2):] *= tau_i

    # calc mean-value and variance of the rates for each cluster
    if clustering_type == "probability":
        mean, variance, vector = connectivit.mean_var(v1_0, **parameter)
    elif clustering_type == "weight":
        mean, variance, vector = connectivit.mean_var(v1_0, **parameter, probability_clustering = False)
    else:
        print("No Clustering-Type is given. Therefore clustering by probabilities is chosen.")
        mean, variance, vector = connectivit.mean_var(v1_0, **parameter)

    # lists for saving the System of equations (phi1,ph2,...) and needed variables (v1,v2,...)
    all_phis = [] #List in len of Pop-Numbers, contains per Cluster the ERF
    phi_minus = [] #as all_phis, but ERF converted to zero
    variables = [] #List in len of Pop-Numbers, contain the name (string) of the rate variables per cluster (v0,v1,v2,...)

    #filling the list
    for i in range(2 * Q):
        all_phis.append(( 0.5 * (1 - sympy.erf(-mean[i] / sqrt(2 * variance[i]) ))) )
        phi_minus.append((0.5 * (1 - sympy.erf(-mean[i] / sqrt(2 * (variance[i] ))))- vector[i]) / tau[i])
        variables.append("v" + str(i + 1))

    """
    Solving the system by the Newton-Method.
    For that: calculating a suitable startpoint by the Halley-Method
    """
    # calculate Jacobian-Matrxi J and gradient of the jacobi H
    J = sympy.lambdify(variables[1:], Matrix(halley.jacobi_matrix(phi_minus[1:], variables[1:])))
    H = sympy.lambdify(variables[1:], Matrix(halley.hessian_matrix(phi_minus[1:], variables[1:])))
    # functions to be solved, stored in a 2*Q-dim vector
    F = sympy.lambdify(variables[1:], Matrix(np.array(phi_minus[1:])))

    # calculating start value by halley-method
    start = halley.solver(phi_minus[1:], variables[1:], variance, initial.reshape((len(variance) - 1,)), F, J, H)
    # solving system by newton-method
    #!!!!!!
    solve, value, success = newton.solver(phi_minus[1:], variables[1:], variance, start.reshape((len(variance) - 1,)), F, J)

    # recursion termination
    if success:
        """
        if the solution of the newthon-method is within the error-tolerance, the recursion can be terminated 
        """
        # solve = fixpoint.iterative(sympy.lambdify(variables[1:], Matrix(np.array(all_phis[1:]))),variables[1:],solve)
        functions = []
        for i in range(2 * Q):
            functions.append(sympy.lambdify(variables, all_phis[i]))

        #output rate can be calculated by using solve in the ERF of the fixef cluster #v_out=f(solve)
        v1_out = functions[0](v1_0, *solve)
        print("Funktionswerte Phi:")
        for func in functions:
            print(func(v1_0, *solve))
        return (v1_0, v1_out, solve)

    else:
        """
        if the solution is NOT within the error-tolerance, 
        the calculation is repeated with an increased input 
        """
        functions = []
        functions_minus = []
        for i in range(2 * Q):
            functions.append(sympy.lambdify(variables, all_phis[i]))
            functions_minus.append(sympy.lambdify(variables, phi_minus[i]))
        print("Die Lösung des Lösungsverfahren liegt für v_in = " + str(v1_0) + " außerhalb der Toleranzgrenze. "
            "Deshalb wird alternativ die Berechnung für v_in = " + str(v1_0 + 0.005) + " wiederholt.")
        return simulation(parameter, v1_0 + 0.005, solve, clustering_type=clustering_type)
