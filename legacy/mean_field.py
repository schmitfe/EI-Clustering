from math import erfc
import numpy as np
import sympy
from scipy.optimize import fmin
from sympy import Matrix


def dm_dt(ms,variables,all_phis):
    m_k = sympy.lambdify(variables,Matrix(all_phis))
    #print(-(ms.reshape((len(ms),1)) - m_k(*ms)))
    taus = np.ones((len(ms),1)) * 10        #params tau_e
    taus[int(len(taus)/2):] *= 0.5          #params tau_ration
    return -(ms.reshape((len(ms),1)) - m_k(*ms)) #/ taus



def _m_k(mu,s,delta_theta=None,small=None):

    if delta_theta is None:

        return erfc(- mu /(np.sqrt(s+small))) #### erfc ändern


def steady_state(params, ms, mean, variance,variables,all_phis, freeze=None, precission=1e-15):
    #Ns = np.array(Ns)
    frozen = np.zeros_like(ms).astype(bool)
    if freeze is not None:
        frozen[freeze] = True
    all_groups_to_solve = [i for i in range(len(ms))]
    m_solutions = ms.copy()



    def minfunc(solve_ms):
        if min(solve_ms) < 0:
            distance = abs(min(solve_ms))
            return 10. + distance * 1e10
        if max(solve_ms) > 1:
            distance = max(solve_ms) - 1
            return 10. + distance * 1e10

        m_solutions[:] = solve_ms
        taus = None
        residual = dm_dt(m_solutions,variables,all_phis)
        #print(np.sum(residual[all_groups_to_solve] ** 2))
        print("residual")
        print(residual)
        print("all-groups")
        print(all_groups_to_solve)

        print(residual[all_groups_to_solve])
        return np.sum(residual[all_groups_to_solve] ** 2)

    if len(ms) > 0:
        #print("here")
        result = fmin(minfunc, ms, disp=False, full_output=True, maxiter=100000, maxfun=100000, xtol=precission,
                      ftol=precission)
        #print("done")
        func_val = result[1]
        result = result[0]

        m_solutions = result

        converged = (func_val < precission)
    else:
        converged = True

    return m_solutions, converged


def m_steady_state(params, ms, mean, variance,variables,all_phis, freeze=None, precission=1e-15):
    if isinstance(freeze, int):
        freeze = [freeze]
    elif freeze is None:
        freeze = []
    # if no constraints are given, each variable gets its own
    constrain_equal = [[i] for i in range(len(ms))]

    frozen = np.zeros_like(ms).astype(bool)
    if freeze is not None:
        frozen[freeze] = True

    unique_ms = []
    unique_m_groups = []
    all_groups_to_solve = []
    for ce in constrain_equal:

        intersection = list(set(freeze).intersection(set(ce)))
        if len(intersection) > 0:
            # if any variables have an equality constraint with a frozen one, they are also frozen to the same value
            frozen[ce] = True
            ms[ce] = ms[intersection[0]]

        else:
            # otherwise they are set to the same value
            unique_ms.append(ms[ce].mean())
            unique_m_groups.append(ce)
            all_groups_to_solve += ce

    unique_ms = np.array(unique_ms)

    m_solutions = ms.copy()

    def minfunc(solve_ms):

        # heavily penalize out of bounds solutions
        if min(solve_ms) < 0:
            distance = abs(min(solve_ms))
            return 10. + distance * 1e10
        if max(solve_ms) > 1:
            distance = max(solve_ms) - 1
            return 10. + distance * 1e10

        for m, inds in zip(solve_ms, unique_m_groups):
            m_solutions[inds] = m
        # print solve_ms,unique_m_groups
        # print m_solutions
        residual = dm_dt(m_solutions,variables,all_phis)
        #print(residual)
        return sum(residual[all_groups_to_solve] ** 2)

    if len(unique_ms) > 0:

        result = fmin(minfunc, unique_ms, disp=False, full_output=True, maxiter=100000, maxfun=100000, xtol=precission,
                      ftol=precission)  # ,full_output=True,xtol = precission,ftol=precission)#,bounds = [(0,1) for n in range(not_frozen)])#,method = 'slsqp',options={'ftol':1e-10})

        func_val = result[1]
        #success = result[-1] == 0
        result = result[0]

        for m, inds in zip(result, unique_m_groups):
            m_solutions[inds] = m

        converged = (func_val < precission)
    else:

        converged = True
    return m_solutions, converged


def EFR(m_start, params, mean, variance, variables, all_phis, v_in, fix=0, n_retry=10, reverse=False, precission=1e-15, min_rate=1e-10):

    # find a good starting point
    m_start, converged = m_steady_state(params, m_start, mean, variance, variables,all_phis, freeze=None, precission=1e-5)
    m_start[fix] = v_in
    # calculate rates m_others of the not fixed populations
    solve, converged = m_steady_state(params, m_start,mean, variance, variables,all_phis, freeze=fix, precission = precission) #precission=precission)

    if (solve < min_rate).any():
        converged = False
    #print(converged)
    #print(len(solve))
    return solve, converged


def EFR_Rost(m_start, params, mean, variance, variables, all_phis, v_in, fix=0, n_retry=10, reverse=False, precission=1e-15, min_rate=1e-10):
    """ effective response function as in amit&mascaro 1999.

        fix:             index of the focus population
        constrain_equal: list of lists, populations in sublists are constraint to equal values
        passive_fix:     None or (pop,val). additional population to be fixed to val. mainly useful for 2d efr.
        """
    #Ns = np.array(Ns)

    #m_out = np.zeros_like(m_in)
    n_pops = len(m_start)
    m_between = np.zeros(n_pops - 1)


    all_fixes = [fix]
    # the populations that are not fixed
    others = np.array([i for i in range(n_pops) if i not in all_fixes])
    # find a good starting point
    #m_start, converged = m_steady_state(params, m_start, mean, variance,variables,all_phis)
    #print("startwert berechnet, converged: " +str(converged))
    #m_start, converged = newton.solver(all_phis,variables,variance,m_start,halley.)
    m_start[fix] = v_in
    m_others, converged = m_steady_state(params, m_start, mean, variance,variables,all_phis,freeze=fix)

    if not converged or (m_others[others] < min_rate).any():
        print("converged:"+str(converged))
        print("min rate erfüllt:"+str((m_others[others] < min_rate).any()))

        # failed to converge
        # try some other starting values
        m_tries = np.linspace(0.02, 0.98, n_retry)
        for j in range(n_retry):
            m_start = (np.random.rand(n_pops) - 0.5) * 0.01 + m_tries[j]
            m_start[m_start >= 0.99] = 0.99
            m_start[m_start <= 0.01] = 0.01
            m_start[fix] = v_in
            m_others, converged = m_steady_state(params, m_start, mean, variance,variables,all_phis, freeze=all_fixes,
                                    precission=precission)

            if converged and (m_others[others] > min_rate).all():  # zero means trouble...
                break
    if converged:
        print("converged")
    return m_others, converged

