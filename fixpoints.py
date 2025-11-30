import os

import numpy as np
import sympy
from sympy import Matrix
import connectivit
import halley
import newton
import run
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy import linalg

tol = 1e-4
steps = 20000
def function(x,y):
    """
    Do Interpolation to get smaller data-grid
    :param x: array with x_data
    :param y: array with y_data
    :return:
    """
    f1 = interp1d(x, y,fill_value="extrapolate")
    x_new = np.linspace(0,1,steps)
    y_new = f1(x_new)
    return (x_new, y_new)



def Derfc(x):
    return -sympy.exp(-x ** 2 / 2) / sympy.sqrt(2 * np.pi)

def calc_jacobi(parameter,v1_0, solve, vector, mean, variance, cluster_bool):
    """

    :param parameter: dictionary with network parameter
    :param v1_0: actual viewed cross point which stability is to be verified
    :param solve: solve for cross point
    :param vector: array with needed rate variables for v1, v2,... for each cluster
    :param mean: array with the mean rate for each cluster depending of v2, v3, ... (v1 fixed)
    :param variance: same as mean, but with variance
    :param cluster_bool: describes kind of clustering. True if probability clustering else False.
    :return: Jacobi Matrix
    """
    #calc mean of the weights and variance of the weights
    mean_weights, weight_vars = connectivit.mean_var(v1_0, **parameter,probability_clustering=cluster_bool,only_matrix=True)
    #get parameters
    Q = parameter['Q']
    tau_e = parameter["tau_e"]
    tau_i = parameter["tau_i"]
    tau = np.ones((2*Q))
    tau[:int(len(tau) / 2)] *= tau_e
    tau[int(len(tau) / 2):] *= tau_i

    #empty jacobi matrix
    jac = np.zeros((len(vector),len(vector)))
    variables = []
    for i in range(len(vector)):
        variables.append("v" + str(i + 1))

    #fill jacobi matrix
    for alpha in range(len(vector)):
        for beta in range(len(vector)):
            f = sympy.lambdify(variables, -Derfc(-mean[alpha]/variance[alpha])*(mean_weights[alpha][beta]*variance[alpha]
            - mean[alpha] * weight_vars[alpha][beta]/(2*variance[alpha]))/variance[alpha]**2)

            jac[alpha][beta] = f(v1_0, *solve)
    jac -= np.eye(len(vector))
    jac /= tau[:, np.newaxis]

    return jac

def calc_fixpoints(file, clustering_type):

    v_in_old, v_out_old, solves, parameter = file
    #interpolation
    v_in, v_out = function(v_in_old,v_out_old)

    Q = parameter['Q']
    tau_e = parameter["tau_e"]
    tau_i = parameter["tau_i"]
    tau = np.ones((2 * Q))
    tau[:int(len(tau) / 2)] *= tau_e
    tau[int(len(tau) / 2):] *= tau_i
    R_Eplus = parameter["R_Eplus"]

    crossings = {}
    fixpoints = {}

    #find crossings and safe crosspoint in "crossings"
    for i, v in enumerate(v_in):
        if np.abs(v - v_out[i]) <= tol:
            crossings[v_out[i]] = (i,v - v_out[i])


                #find exact solution for each cross-point
    print("R_E+: " + str(R_Eplus))
    for cross_point in crossings:
        print("Cross-Point:" + str(cross_point))

        i = crossings[cross_point][0]
        slope = (v_out[i] - v_out[i-1]) / (v_in[i] - v_in[i-1])
        if slope > 1:
            fixpoints[cross_point] = 'unstable'
        #check if point is global stable
        else:

            # search for solve near to the cross point: new start value
            i = 0

            while cross_point < v_out_old[i] and i < len(solves)-1:
                i += 1


            initial = solves[i]
            if clustering_type == "probability":
                cluster_bool = True
                mean, variance, vector = connectivit.mean_var(cross_point, **parameter)
            elif clustering_type == "weight":
                cluster_bool = False
                mean, variance, vector = connectivit.mean_var(cross_point, **parameter, probability_clustering=False)
            else:
                print("No Clustering-Type is given. Therefore clustering by probabilities is chosen.")
                cluster_bool = True
                mean, variance, vector = connectivit.mean_var(cross_point, **parameter)

            phi_minus = []
            variables = []
            for i in range(len(vector)):
                phi_minus.append((0.5 * (1 - sympy.erf(-mean[i] / sympy.sqrt(2 * (variance[i])))) - vector[i]) / tau[i])
                variables.append("v" + str(i + 1))

            J = sympy.lambdify(variables[1:], Matrix(halley.jacobi_matrix(phi_minus[1:], variables[1:])))
            H = sympy.lambdify(variables[1:], Matrix(halley.hessian_matrix(phi_minus[1:], variables[1:])))
            # functions to be solved, stored in a 2*Q-dim vector
            F = sympy.lambdify(variables[1:], Matrix(np.array(phi_minus[1:])))

            # find good start value by halley-method
            print("Optimize start value")
            start = halley.solver(phi_minus[1:], variables[1:], variance, initial, F, J, H)
            print(start)
            # solving system by newton-method
            print("Solving...")
            solve, value, success = newton.solver(phi_minus[1:], variables[1:], variance,
                                                  start.reshape((len(variance) - 1,)), F, J)
            if not success:
                print('warning! convergence problems')
            else:
                print('Converged!')

            variables = []
            for i in range(len(solve)+1):
                variables.append("v" + str(i + 1))
            #check stability of each cross pont by  calc eigenvalues of jacobi matrix
            jacobi =  calc_jacobi(parameter, cross_point, solve, vector, mean, variance, cluster_bool)
            eigval = np.linalg.eigvals(jacobi)
            if (eigval < 0).all():  #if all eigenvalues > 0: crosspoint is stable
                fixpoints[cross_point] = "stable"
            else:
                fixpoints[cross_point] = "unstable"

    return fixpoints



    # lower_stable = []
    # unstable = []
    # upper_stable = []
    # # the keys of the dict crossings are values
    # for value in crossings:
    #     if crossings[value] == 'unstable':
    #         unstable.append(value)
    #     elif value < 0.5:
    #         lower_stable.append(value)
    #     elif value >= 0.5:
    #         upper_stable.append(value)
    # f_points_return = {}
    # lower_median = np.median(lower_stable)
    # middle_median = np.median(unstable)
    # upper_median = np.median(upper_stable)
    # ind_1 = (len(lower_stable))//2
    # ind_2 = (len(unstable))//2
    # ind_3 = (len(upper_stable))//2
    # lower_median = lower_stable[ind_1] if lower_stable != [] else None
    # middle_median = unstable[ind_2] if unstable != [] else None
    # upper_median = upper_stable[ind_3] if upper_stable != [] else None
    # f_points_return[lower_median] = 'stable'
    # f_points_return[middle_median] = 'unstable'
    # f_points_return[upper_median] = 'stable'

    #return crossings
    #return f_points_return







