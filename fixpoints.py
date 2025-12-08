import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from rate_system import RateSystem

tol = 1e-4
steps = 20000


def function(x, y):
    """
    Smooth ERF samples via interpolation.  When fewer than two finite points are
    available, fall back to the raw data so partial sweeps can still be plotted
    and analyzed.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return x_arr, y_arr
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, unique_idx = np.unique(x_arr, return_index=True)
    x_arr = unique_x
    y_arr = y_arr[unique_idx]
    if x_arr.size == 1:
        x_new = np.linspace(0, 1, steps)
        y_new = np.full_like(x_new, y_arr[0])
        return x_new, y_new
    f1 = interp1d(x_arr, y_arr, fill_value="extrapolate")
    x_new = np.linspace(0, 1, steps)
    y_new = f1(x_new)
    return (x_new, y_new)


def calc_jacobi(parameter, v1_0, solve, kappa):
    """

    :param parameter: dictionary with network parameter
    :param v1_0: actual viewed cross point which stability is to be verified
    :param solve: solve for cross point
    :param vector: array with needed rate variables for v1, v2,... for each cluster
    :param mean: array with the mean rate for each cluster depending of v2, v3, ... (v1 fixed)
    :param variance: same as mean, but with variance
    :param kappa: mixing coefficient between probability and weight clustering
    :return: Jacobi Matrix
    """
    rate_system = RateSystem(parameter, v1_0, kappa=kappa)
    full_rates = rate_system.full_rates_numpy(solve)
    return rate_system.jacobian_numpy(full_rates)

def calc_fixpoints(file, kappa):

    v_in_old, v_out_old, solves, parameter = file
    #interpolation
    v_in, v_out = function(v_in_old,v_out_old)
    if len(v_in) == 0 or len(v_out) == 0:
        print("Skipping fixpoint analysis: no ERF samples available.")
        return {}
    v_out_old = np.asarray(v_out_old, dtype=float)

    Q = parameter['Q']
    R_Eplus = parameter["R_Eplus"]

    def find_crossings():
        diff = v_in - v_out
        crossings = []
        prev_diff = diff[0]
        for i in range(1, len(diff)):
            curr_diff = diff[i]
            cross_val = None
            if np.abs(curr_diff) <= tol:
                cross_val = v_out[i]
            elif np.abs(prev_diff) <= tol:
                cross_val = v_out[i - 1]
            elif prev_diff * curr_diff < 0:
                weight = prev_diff / (prev_diff - curr_diff)
                cross_val = v_out[i - 1] + weight * (v_out[i] - v_out[i - 1])
            if cross_val is not None:
                if crossings and np.abs(cross_val - crossings[-1][0]) <= tol:
                    crossings[-1] = (float(cross_val), i)
                else:
                    crossings.append((float(cross_val), i))
            prev_diff = curr_diff
        return crossings

    crossings = find_crossings()
    fixpoints = {}

    #find crossings and safe crosspoint in "crossings"
    print("R_E+: " + str(R_Eplus))
    for cross_point, idx in crossings:
        print("Cross-Point:" + str(cross_point))

        if idx <= 0:
            slope = np.inf
        else:
            slope = (v_out[idx] - v_out[idx - 1]) / (v_in[idx] - v_in[idx - 1])
        if not np.isfinite(slope) or slope > 1:
            fixpoints[cross_point] = 'unstable'
        #check if point is global stable
        else:

            # search for solve near to the cross point: new start value
            closest_idx = int(np.argmin(np.abs(v_out_old - cross_point)))
            closest_idx = min(max(closest_idx, 0), len(solves) - 1)
            initial = solves[closest_idx]

            print("Solving...")
            rate_system = RateSystem(parameter, cross_point, kappa=kappa)
            initial_guess = np.asarray(initial, dtype=float)
            solve, value, success = rate_system.solve(initial_guess)
            if not success:
                print('warning! convergence problems')
            else:
                print('Converged!')

            #check stability of each cross pont by  calc eigenvalues of jacobi matrix
            jacobi =  calc_jacobi(parameter, cross_point, solve, kappa)
            if not np.isfinite(jacobi).all():
                print("Skipping stability check for cross point "
                      f"{cross_point}: Jacobian contains non-finite values")
                fixpoints[cross_point] = "unstable"
                continue
            try:
                eigval = np.linalg.eigvals(jacobi)
            except np.linalg.LinAlgError as exc:
                print("Skipping stability check for cross point "
                      f"{cross_point}: {exc}")
                fixpoints[cross_point] = "unstable"
                continue
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
