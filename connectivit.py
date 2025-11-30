import numpy as np
import math
from sympy import Symbol
import matrix_builder


def mean_var(v1_0, N, N_E, N_I, Q, V_th, g, p0_ee, p0_ie, p0_ei, p0_ii, m_X, tau_e, tau_i, R_Eplus, R_j, clustering_type, only_matrix= False, **kwargs):
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
    n_e = N_E / N
    n_i = N_I / N

    theta_E = V_th
    theta_I = V_th
    # threshold vector
    V_th = np.array([theta_E] * Q + [theta_I] * Q)

    R_Eminus = (Q - R_Eplus) / (Q - 1)
    R_Iplus = 1 + R_j * (R_Eplus - 1)
    R_Iminus = (Q - R_Iplus) / (Q - 1)

    # connectivity weights
    # we neglect here and in the following the factor (N_x-1)/(N_x) for EE and II connections
    j_EE = theta_E / (math.sqrt(p0_ee * n_e))
    j_IE = theta_I / math.sqrt(p0_ie * n_e)
    j_EI = -g * j_EE * p0_ee * n_e / (p0_ei * n_i)
    j_II = -j_IE * p0_ie * n_e / (p0_ii * n_i)

    if clustering_type == "probability":
        # Intra Cluster
        P_EE = R_Eplus * p0_ee
        P_IE = R_Iplus * p0_ie
        P_EI = R_Iplus * p0_ei
        P_II = R_Iplus * p0_ii

        # Inter Cluster
        p_ee = R_Eminus * p0_ee
        p_ie = R_Iminus * p0_ie
        p_ei = R_Iminus * p0_ei
        p_ii = R_Iminus * p0_ii

        #Intra cluster weights
        J_EE = j_EE
        J_IE = j_IE
        J_EI = j_EI
        J_II = j_II

        #Inter Cluster weights
        j_ee = j_EE
        j_ie = j_IE
        j_ei = j_EI
        j_ii = j_II

    else:
        # Intra Cluster
        P_EE = p0_ee
        P_IE = p0_ie
        P_EI = p0_ei
        P_II = p0_ii

        # Inter Cluster
        p_ee = p0_ee
        p_ie = p0_ie
        p_ei = p0_ei
        p_ii = p0_ii

        #Intra cluster weights
        J_EE = R_Eplus * j_EE
        J_IE = R_Iplus * j_IE
        J_EI = R_Iplus * j_EI
        J_II = R_Iplus * j_II

        #Inter Cluster weights
        j_ee = R_Eminus * j_EE
        j_ie = R_Iminus * j_IE
        j_ei = R_Iminus * j_EI
        j_ii = R_Iminus * j_II


    # matrix A entries (Mean-Value)
    EE_IN = J_EE * P_EE * n_e * math.sqrt(N)  # tau_e * P_EE * (n_e - 1) * J_EE
    EE_OUT = j_ee * p_ee * n_e * math.sqrt(N)  # tau_e * p_ee * (Q - 1) * n_e * J_EE

    IE_IN = J_IE * P_IE * n_e * math.sqrt(N)  # tau_e * P_IE * n_e * J_IE
    IE_OUT = j_ie * p_ie * n_e * math.sqrt(N)  # tau_e * p_ie * n_e * J_IE

    EI_IN = J_EI * P_EI * n_i * math.sqrt(N)  # tau_i * P_EI * n_i * J_EI
    EI_OUT = j_ei * p_ei * n_i * math.sqrt(N)  # tau_i * p_ei * (Q - 1) * n_i * J_EI

    II_IN = J_II * P_II * n_i * math.sqrt(N)
    II_OUT = j_ii * p_ii * n_i * math.sqrt(N)

    # entries matrix B (variance)
    var_EE_IN = P_EE * (1 - P_EE) * J_EE ** 2 * n_e
    var_EE_OUT = p_ee * (1 - p_ee) * j_ee ** 2 * n_e

    var_IE_IN = P_IE * (1 - P_IE) * J_IE ** 2 * n_e
    var_IE_OUT = p_ie * (1 - p_ie) * j_ie ** 2 * n_e

    var_EI_IN = P_EI * (1 - P_EI) * J_EI ** 2 * n_i
    var_EI_OUT = p_ei * (1 - p_ei) * j_ei ** 2 * n_i

    var_II_IN = P_II * (1 - P_II) * J_II ** 2 * n_i
    var_II_OUT = p_ii * (1 - p_ii) * j_ii ** 2 * n_i



    # external Input
    J_EX = math.sqrt(p0_ee * N_E)
    J_IX = 0.8 * J_EX
    u_extE = J_EX * m_X
    u_extI = J_IX * m_X
    u_ext = np.array([u_extE] * Q + [u_extI] * Q)

    mean_values = dict(EE_IN=EE_IN, EE_OUT=EE_OUT, IE_IN=IE_IN, IE_OUT=IE_OUT, EI_IN=EI_IN, EI_OUT=EI_OUT, II_IN=II_IN,
                       II_OUT=II_OUT)

    A = matrix_builder.build(Q, mean_values)



    var_values = dict(EE_IN=var_EE_IN, EE_OUT=var_EE_OUT, IE_IN=var_IE_IN, IE_OUT=var_IE_OUT, EI_IN=var_EI_IN,
                      EI_OUT=var_EI_OUT, II_IN=var_II_IN,
                      II_OUT=var_II_OUT)
    B = matrix_builder.build(Q, var_values)


    # build vector with all variables v1, v2, ...
    vector = [v1_0]
    for i in range(2, 2 * Q + 1):
        vector.append(Symbol("v" + str(i)))

    # calc mean-rates
    u = np.dot(A, vector) - V_th + u_ext
    # calc variance
    var = np.dot(B, vector)

    # print("Intra-Cluster:")
    # print([P_EE, P_IE, P_EI, P_II])
    # print("Inter-Cluster:")
    # print([p_ee, p0_ie, p_ei, p_ii])
    if (np.array([P_EE, P_IE, P_EI, P_II])>1).any() or (np.array([p_ee, p0_ie, p_ei, p_ii])>1).any():
        print("Attention! "
              "The raised connection probability exceeds 1 "
              "and thus no longer fulfills the conditions for calculating the variance.")

    if only_matrix:
        return (A, B)
    else:
        return (u, var, np.array(vector))



