import numpy as np
import sympy
from sympy import Matrix, Symbol, diff

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


def jacobi_matrix(funcs, rates):
    """
    calculates the Jacobin Matrix
    :param funcs: (2*Q-1)-dim vector with response functions that should be solved
    :param rates: (2*Q-1)-dim vector v2, v3, ...
    :return: (2*Q-1) x (2*Q-1) Jacobian Matrix
    """
    jacobi = []
    for i, phi in enumerate(funcs):
        row = []
        for v in rates:
            v = Symbol(str(v))
            row.append(diff(phi, v))
        jacobi.append(row)
    return np.array(jacobi)


def hessian_matrix(funcs, rates):
    """
    calculates the gradient of the Jacobin Matrix
    :param funcs: (2*Q-1)-dim vector with response functions that should be solved
    :param rates: (2*Q-1)-dim vector v2, v3, ...
    :return: (2*Q-1) x (2*Q-1) grad (Jacobian Matrix)
    """
    hessian = []
    for i, phi in enumerate(funcs):
        row = []
        for v in rates:
            v = Symbol(str(v))
            row.append(diff(diff(phi, v), v))
        hessian.append(row)
    return np.array(hessian)


def update_mini(x0, values, mini):
    """
    check if the given value f(x0) is smaller than the value stored in mini.
    if so replace the stored value in mini with the newly found minimum.
    If no mini remains unchanged.
    :param x0: solve
    :param values: f(x0)
    :param mini: tuple with the last found minimum and the norm of this minimum #(x_mini, norm(f(x_mini)))
    :return:
    """
    norm = np.linalg.norm(values)
    if mini[1] > norm:
        mini[0] = x0
        mini[1] = norm
    return mini


def recursion(funcs, var, variance, F, H, J, starter, x0, mini, i=0, max_iter=200):
    """
        Recursive function, which is to find a solve for the equation system by means of the Halley method.
         If the solution of the system converges or the value is within the tolerance near zero, the solution is returned.
          Otherwise the best value, i.e. the minimum, is returned.
          The solve is used as the start value for the Newton-Method
        :param funcs: (2*Q-1)-dim vector with the functions
        :param var: (2*Q-1)-dim vector with variables v2, v3 ...
        :param variance: (2*Q-1)-dim vector with variance for each cluster (except cluster 1)
        :param F: symbolic version of funcs
        :param J: symbolic jacobian matrix of funcs
        :param starter: original start-value of the newton-iteration
        :param x0: actual start-value of the newton-iteration
        :param mini: tuple with the stored last founded minimum-value
        :param i: counter of iteration steps
        :return: 2*Q-1)-dim vector x that solves:  funcs(x) = 0
        """
    while i < max_iter:
        shape = int(len(var))
        x0.reshape((shape,))
        # Newton
        xa = np.linalg.solve(J(*x0), - F(*x0)).reshape(shape, )
        # Halley - Korrektur
        xb = np.linalg.solve(J(*x0) + 0.5 * H(*x0).dot(xa), - F(*x0)).dot(xa.dot(xa)).reshape(shape, )
        xn = x0 + xa + xb

        if np.any(variance(*xn) < 0):
            return recursion(funcs, var, variance, F, H, J, starter + 0.001, starter + 0.001, mini, i + 1)

        val = give_values(funcs, var, xn)

        if np.all(np.abs(val) < 1e-1):
            return xn

        else:
            return recursion(funcs, var, variance, F, H, J, starter, xn, update_mini(xn, val, mini), i + 1)
    return mini[0]


def solver(funcs, var, variance, initial, F, J, H):
    print("Startwert für das Lösungsverfahren wird bestimmt")
    variance = sympy.lambdify(var, Matrix(np.array(variance)))
    x0 = initial.reshape((int(len(var)),))
    mini = [x0, np.linalg.norm(give_values(funcs, var, x0))]
    return recursion(funcs, var, variance, F, H, J, x0, x0, mini)
