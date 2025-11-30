import numpy as np
import sympy
from sympy import Matrix

def give_values(phi_minus, var, solve):
    """
    calculate value = f(x)
    :param phi_minus: vector of the functions converted to zero
    :param var: variables
    :param solve: value x that should be inserted to the function
    :return: vector f(x)
    """

    functions_minus = []
    values = []
    for i in range(len(var)):
        functions_minus.append(sympy.lambdify(var, phi_minus[i]))
    for func in functions_minus:
        values.append((func(*solve)))
    return values

def updated_mini(x0, values, mini):
    """
    checke if the given value f(x0) is smaller than the value stored in mini.
    if so replace the stored value in mini with the newly found minimum.
    If no mini remains unchanged.
    :param x0: solve
    :param values: f(x0)
    :param mini: tuple with the last found minimum and the norm of this minimum #(x_mini, norm(f(x_mini)))
    :return: updated mini -> (x_mini, norm(f(x_mini)))
    """
    norm = np.linalg.norm(values)
    if mini == None:
        return (x0, norm)

    if mini[1] > norm:
        mini[0] = x0
        mini[1] = norm
    return mini


def recursion(funcs, var, variance, F, J, starter, x0, mini, i=0, tolerance=1e-6, max_iter=100):
    """
    Recursive function, which is to find a solution for the equation system by means of the newton method.
     If the solution of the system converges or the value is within the tolerance near zero, the solution is returned.
      Otherwise, another iterative step is performed until the solution is found or the recursion depth is reached.
      In this case the best value, i.e. the minimum, is returned.
    :param funcs: (2*Q-1)-dim vector with the functions
    :param var: (2*Q-1)-dim vector with variables v2, v3 ...
    :param variance: (2*Q-1)-dim vector with variance for each cluster (except cluster 1)
    :param F: symbolic version of funcs
    :param J: symbolic jacobian matrix of funcs
    :param starter: original start-value of the newton-iteration
    :param x0: actual start-value of the newton-iteration
    :param mini: tuple with the stored last founded minimum-value
    :param i: counter of iteration steps
    :param tolerance: tolerance for the newton-iteration
    :return: 2*Q-1)-dim vector x that solves:  funcs(x) = 0
    """
    try:
        shape = int(len(var))
        # Newton
        dx = np.linalg.solve(J(*x0), - F(*x0)).reshape(shape, )
        xn = x0 + dx

        if i > max_iter:
            raise RecursionError

        if np.any(variance(*xn) < 0):
            return recursion(funcs, var, variance, F, J, starter + 0.001, starter + 0.001, mini,
                             i + 1)  # ? starter = x0

        val = give_values(funcs, var, xn)

        if np.all(np.abs(val) < tolerance) or np.linalg.norm(xn - x0) < tolerance:
            return (xn, val, True)

        else:
            return recursion(funcs, var, variance, F, J, starter, xn, updated_mini(xn, val, mini), i + 1)

    except RecursionError:
        return (*mini, False)


def solver(funcs, var, variance, x0, F, J):
    """

    :param funcs:
    :param var:
    :param variance:
    :param x0:
    :param F:
    :param J:
    :return:
    """
    print("Newtonverfahren wird angewendet.")
    mini = [x0, np.linalg.norm(give_values(funcs, var, x0))]
    return recursion(funcs, var, sympy.lambdify(var, Matrix(np.array(variance))), F, J, np.zeros_like(x0), x0, mini)
