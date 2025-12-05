import math
from dataclasses import dataclass

import numpy as np
from sympy import Symbol

import matrix_builder


@dataclass(frozen=True)
class ConnectivityMatrices:
    A: np.ndarray
    B: np.ndarray
    V_th: np.ndarray
    u_ext: np.ndarray
    intra_probs: np.ndarray
    inter_probs: np.ndarray

    @property
    def bias(self) -> np.ndarray:
        return self.u_ext - self.V_th


def _build_connectivity_matrices(N, N_E, N_I, Q, V_th, g, p0_ee, p0_ie, p0_ei, p0_ii,
                                 m_X, R_Eplus, R_j, kappa, connection_type="bernoulli"):
    n_er = N_E / N
    n_ir = N_I / N
    n_e = N_E / Q
    n_i = N_I / Q

    theta_E = V_th
    theta_I = V_th
    V_th_vec = np.array([theta_E] * Q + [theta_I] * Q, dtype=float)

    R_Iplus = 1 + R_j * (R_Eplus - 1)

    j_EE = theta_E / (math.sqrt(p0_ee * n_er))
    j_IE = theta_I / math.sqrt(p0_ie * n_er)
    j_EI = -g * j_EE * p0_ee * n_er / (p0_ei * n_ir)
    j_II = -j_IE * p0_ie * n_er / (p0_ii * n_ir)


    j_EE *= 1/math.sqrt(N)
    j_IE *= 1 / math.sqrt(N)
    j_EI *= 1 / math.sqrt(N)
    j_II *= 1 / math.sqrt(N)

    def mix_scales(R_plus: float) -> tuple[float, float, float, float]:
        prob_in = R_plus ** (1.0 - kappa)
        prob_out = (Q - prob_in) / (Q - 1)
        weight_in = R_plus ** kappa
        weight_out = (Q - weight_in) / (Q - 1)
        return prob_in, prob_out, weight_in, weight_out

    P_scale_in_E, P_scale_out_E, J_scale_in_E, J_scale_out_E = mix_scales(R_Eplus)
    P_scale_in_I, P_scale_out_I, J_scale_in_I, J_scale_out_I = mix_scales(R_Iplus)

    P_EE = p0_ee * P_scale_in_E
    p_ee = p0_ee * P_scale_out_E

    P_IE = p0_ie * P_scale_in_I
    p_ie = p0_ie * P_scale_out_I

    P_EI = p0_ei * P_scale_in_I
    p_ei = p0_ei * P_scale_out_I

    P_II = p0_ii * P_scale_in_I
    p_ii = p0_ii * P_scale_out_I

    J_EE = j_EE * J_scale_in_E
    j_ee = j_EE * J_scale_out_E

    J_IE = j_IE * J_scale_in_I
    j_ie = j_IE * J_scale_out_I

    J_EI = j_EI * J_scale_in_I
    j_ei = j_EI * J_scale_out_I

    J_II = j_II * J_scale_in_I
    j_ii = j_II * J_scale_out_I

    EE_IN = J_EE * P_EE * n_e
    EE_OUT = j_ee * p_ee * n_e

    IE_IN = J_IE * P_IE * n_e
    IE_OUT = j_ie * p_ie * n_e

    EI_IN = J_EI * P_EI * n_i
    EI_OUT = j_ei * p_ei * n_i

    II_IN = J_II * P_II * n_i
    II_OUT = j_ii * p_ii * n_i

    conn_kind = (connection_type or "bernoulli").lower()
    allowed_conn = {"bernoulli", "poisson", "fixed-indegree"}
    if conn_kind not in allowed_conn:
        raise ValueError(f"Unknown connection type '{connection_type}'. Expected one of {sorted(allowed_conn)}.")

    def compute_variance(prob: float, weight: float, population: float) -> float:
        if conn_kind == "poisson":
            return prob * weight ** 2 * population
        if conn_kind == "fixed-indegree":
            return prob * (1-(1/population)) * weight ** 2 * population  # omitted * population*(1/population) First
            # for prob*population -> indegree, second for multinominal distribution
        return prob * (1 - prob) * weight ** 2 * population

    var_EE_IN = compute_variance(P_EE, J_EE, n_e)
    var_EE_OUT = compute_variance(p_ee, j_ee, n_e)
    var_IE_IN = compute_variance(P_IE, J_IE, n_e)
    var_IE_OUT = compute_variance(p_ie, j_ie, n_e)
    var_EI_IN = compute_variance(P_EI, J_EI, n_i)
    var_EI_OUT = compute_variance(p_ei, j_ei, n_i)
    var_II_IN = compute_variance(P_II, J_II, n_i)
    var_II_OUT = compute_variance(p_ii, j_ii, n_i)

    mean_values = dict(EE_IN=EE_IN, EE_OUT=EE_OUT, IE_IN=IE_IN, IE_OUT=IE_OUT, EI_IN=EI_IN, EI_OUT=EI_OUT,
                       II_IN=II_IN, II_OUT=II_OUT)
    A = np.asarray(matrix_builder.build(Q, mean_values), dtype=float)

    var_values = dict(EE_IN=var_EE_IN, EE_OUT=var_EE_OUT, IE_IN=var_IE_IN, IE_OUT=var_IE_OUT, EI_IN=var_EI_IN,
                      EI_OUT=var_EI_OUT, II_IN=var_II_IN, II_OUT=var_II_OUT)
    B = np.asarray(matrix_builder.build(Q, var_values), dtype=float)

    J_EX = math.sqrt(p0_ee * N_E)
    J_IX = 0.8 * J_EX
    u_extE = J_EX * m_X
    u_extI = J_IX * m_X
    u_ext = np.array([u_extE] * Q + [u_extI] * Q, dtype=float)

    intra = np.array([P_EE, P_IE, P_EI, P_II], dtype=float)
    inter = np.array([p_ee, p_ie, p_ei, p_ii], dtype=float)

    return ConnectivityMatrices(A=A, B=B, V_th=V_th_vec, u_ext=u_ext, intra_probs=intra, inter_probs=inter)


def linear_connectivity(mixing_parameter=None, connection_type=None, **parameters) -> ConnectivityMatrices:
    params = dict(parameters)
    mix = mixing_parameter if mixing_parameter is not None else params.pop("kappa", 0.0)
    conn_kind = connection_type or params.pop("connection_type", "bernoulli")
    params.pop("kappa", None)
    params.pop("connection_type", None)
    params.pop("tau_e", None)
    params.pop("tau_i", None)
    return _build_connectivity_matrices(kappa=mix, connection_type=conn_kind, **params)


def mean_var(v1_0, N, N_E, N_I, Q, V_th, g, p0_ee, p0_ie, p0_ei, p0_ii, m_X, tau_e, tau_i, R_Eplus, R_j,
             kappa, connection_type="bernoulli", only_matrix=False, **kwargs):
    """
    Calculates the mean-rates and variance for each cluster

    :param v1_0: input rate of cluster 1
    :param N: Number of Neurons
    :param N_E: Number of excitatory neurons
    :param N_I: Number of inhibitory neurons
    :param Q: Number of Cluster per neuron type
    :param V_th: Threshold Voltage
    :param g: ratio ex <> inh
    :param p0_ee: connection probability ex -> ex
    :param p0_ie: connection probability ex -> inh
    :param p0_ei: connection probability inh -> ex
    :param p0_ii: connection probability inh -> inh
    :param R_Eplus: ratio intra <> inter
    :param R_j: ratio intra ex cluster <> intra inh cluster
    :return: (mean, variance, vector with the defined variables) where all are (2*Q)-dim vectors
    """
    matrices = _build_connectivity_matrices(N=N, N_E=N_E, N_I=N_I, Q=Q, V_th=V_th, g=g, p0_ee=p0_ee,
                                            p0_ie=p0_ie, p0_ei=p0_ei, p0_ii=p0_ii, m_X=m_X, R_Eplus=R_Eplus,
                                            R_j=R_j, kappa=kappa, connection_type=connection_type)

    # build vector with all variables v1, v2, ...
    vector = [v1_0]
    for i in range(2, 2 * Q + 1):
        vector.append(Symbol("v" + str(i)))

    # calc mean-rates
    u = np.dot(matrices.A, vector) - matrices.V_th + matrices.u_ext
    # calc variance
    var = np.dot(matrices.B, vector)

    # print("Intra-Cluster:")
    # print([P_EE, P_IE, P_EI, P_II])
    # print("Inter-Cluster:")
    # print([p_ee, p0_ie, p_ei, p_ii])
    if (matrices.intra_probs > 1).any() or (matrices.inter_probs > 1).any():
        print("Attention! "
              "The raised connection probability exceeds 1 "
             # "and thus no longer fulfills the conditions for calculating the variance."
        )

    if only_matrix:
        return (matrices.A, matrices.B)
    else:
        return (u, var, np.array(vector))
