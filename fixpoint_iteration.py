import numpy as np
import sympy

def update_mini(x0,values,mini):
    norm = np.linalg.norm(values)
    if mini == []:
        mini.append(x0)
        mini.append(norm)
    elif mini[1] > norm:
        mini[0] = x0
        mini[1] = norm
    return mini

def give_values(F,var, solve):
    functions = []
    values = []
    for i in range(len(var)):
        functions.append(sympy.lambdify(var, F[i]))
    for func in functions:
        values.append((func(*solve)))
    return values

def iterative(F,var, x0,mini=[], max_iter = 2000, tol=1e-6):
    try:
        xn = F(*x0)
        val = give_values(F, var, xn)
        if np.linalg.norm(xn - x0) < tol or np.linalg.norm(val) < tol:
            return mini[0]
        else:
            return iterative(F, var, xn,update_mini(xn, val, mini))
    except RecursionError:
        print("Die Lösung des Fixpunktverfahren liegt für v_in = "+str()+" außerhalb der Toleranzgrenze. "
            "Deshalb wird die alternativ die Berechnung für v_in = "+str()+"wiederholt.")
        return (*mini, False)


    return mini[0]

