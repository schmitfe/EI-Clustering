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
                                 m_X, R_Eplus, R_j, clustering_type):
    n_e = N_E / N
    n_i = N_I / N

    theta_E = V_th
    theta_I = V_th
    V_th_vec = np.array([theta_E] * Q + [theta_I] * Q, dtype=float)

    R_Eminus = (Q - R_Eplus) / (Q - 1)
    R_Iplus = 1 + R_j * (R_Eplus - 1)
    R_Iminus = (Q - R_Iplus) / (Q - 1)

    j_EE = theta_E / (math.sqrt(p0_ee * n_e))
    j_IE = theta_I / math.sqrt(p0_ie * n_e)
    j_EI = -g * j_EE * p0_ee * n_e / (p0_ei * n_i)
    j_II = -j_IE * p0_ie * n_e / (p0_ii * n_i)

    if True:
        j_EE *= 1/math.sqrt(N)
        j_IE *= 1 / math.sqrt(N)
        j_EI *= 1 / math.sqrt(N)
        j_II *= 1 / math.sqrt(N)

    if clustering_type == "probability":
        P_EE = R_Eplus * p0_ee
        P_IE = R_Iplus * p0_ie
        P_EI = R_Iplus * p0_ei
        P_II = R_Iplus * p0_ii

        p_ee = R_Eminus * p0_ee
        p_ie = R_Iminus * p0_ie
        p_ei = R_Iminus * p0_ei
        p_ii = R_Iminus * p0_ii

        J_EE = j_EE
        J_IE = j_IE
        J_EI = j_EI
        J_II = j_II

        j_ee = j_EE
        j_ie = j_IE
        j_ei = j_EI
        j_ii = j_II
    else:
        P_EE = p0_ee
        P_IE = p0_ie
        P_EI = p0_ei
        P_II = p0_ii

        p_ee = p0_ee
        p_ie = p0_ie
        p_ei = p0_ei
        p_ii = p0_ii

        J_EE = R_Eplus * j_EE
        J_IE = R_Iplus * j_IE
        J_EI = R_Iplus * j_EI
        J_II = R_Iplus * j_II

        j_ee = R_Eminus * j_EE
        j_ie = R_Iminus * j_IE
        j_ei = R_Iminus * j_EI
        j_ii = R_Iminus * j_II

    EE_IN = J_EE * P_EE * n_e * math.sqrt(N)
    EE_OUT = j_ee * p_ee * n_e * math.sqrt(N)

    IE_IN = J_IE * P_IE * n_e * math.sqrt(N)
    IE_OUT = j_ie * p_ie * n_e * math.sqrt(N)

    EI_IN = J_EI * P_EI * n_i * math.sqrt(N)
    EI_OUT = j_ei * p_ei * n_i * math.sqrt(N)

    II_IN = J_II * P_II * n_i * math.sqrt(N)
    II_OUT = j_ii * p_ii * n_i * math.sqrt(N)

    if False:
        var_EE_IN = P_EE * J_EE ** 2 * n_e
        var_EE_OUT = p_ee * j_ee ** 2 * n_e
        var_IE_IN = P_IE * J_IE ** 2 * n_e
        var_IE_OUT = p_ie * j_ie ** 2 * n_e
        var_EI_IN = P_EI * J_EI ** 2 * n_i
        var_EI_OUT = p_ei * j_ei ** 2 * n_i
        var_II_IN = P_II * J_II ** 2 * n_i
        var_II_OUT = p_ii * j_ii ** 2 * n_i
    else:
        var_EE_IN = P_EE * (1-P_EE) * J_EE ** 2 * n_e
        var_EE_OUT = p_ee * (1-p_ee) * j_ee ** 2 * n_e
        var_IE_IN = P_IE * (1-P_IE) * J_IE ** 2 * n_e
        var_IE_OUT = p_ie * (1-p_ie) * j_ie ** 2 * n_e
        var_EI_IN = P_EI * (1-P_EI) * J_EI ** 2 * n_i
        var_EI_OUT = p_ei * (1-p_ei) * j_ei ** 2 * n_i
        var_II_IN = P_II * (1-P_II) * J_II ** 2 * n_i
        var_II_OUT = p_ii * (1-p_ii) * j_ii ** 2 * n_i

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


def linear_connectivity(clustering_type=None, **parameters) -> ConnectivityMatrices:
    params = dict(parameters)
    cluster = clustering_type or params.pop("clustering_type", "probability")
    params.pop("tau_e", None)
    params.pop("tau_i", None)
    return _build_connectivity_matrices(clustering_type=cluster, **params)


def mean_var(v1_0, N, N_E, N_I, Q, V_th, g, p0_ee, p0_ie, p0_ei, p0_ii, m_X, tau_e, tau_i, R_Eplus, R_j,
             clustering_type, only_matrix=False, **kwargs):
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
                                            R_j=R_j, clustering_type=clustering_type)

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
