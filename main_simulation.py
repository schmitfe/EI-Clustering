"""In this module the simulation of a clustered network is called.
It can be selected whether the network is to be clustered by the weighting J or the connectivity probability p.
Also the parameter R_j must be set, which gives the strength of the I-clustering.
All other network parameters are also set here.
A mean-field analysis is used to calculate the dynamics of the network for different e-cluster strengths,
which is specified by the parameter P_Eplus (-> P_Eplus_range).
"""

import numpy as np
import safe_data
import multiprocessing as mp
import os

#Set Parameter
parameter = dict(
    N=5000,         #Number of Neurons
    N_E=4000,       #Number of excitatory Neurons
    N_I=1000,       #Number of inhibitory Neurons

    # Number of Cluster
    Q = 20,

    # Treshold Voltage
    V_th = 1,

    # ratio ex. <> inh. weights
    g=1.2,

    # connectivity propabilities
    p0_ee=0.2,
    p0_ie=0.5,
    p0_ei=0.5,
    p0_ii=0.5,

    #time constants
    tau_e = 10,
    tau_i = 5,

    # ratio connectivity parameter inter <> intra cluster
    R_Eplus = None,
    # use enviroment variable
    R_j = None,
    clustering_type = "probability", # "weight" or "probability"
    m_X = 0.03, # mean of the external input

    )

####################################################################
def execute(parameter):
    """
    method that performs the analysis for the given value of P_Eplus.
    The results are stored as a dictionary in a pickle file with name "filename"

    """
    # read os variable plot
    plot = os.getenv('plot', 'False')
    if plot == 'True':
        plot = True
    else:
        plot = False

    return safe_data.main("Q" + str(parameter["Q"]) +"R_j" + str(parameter["R_j"]), parameter, plot=plot)


if __name__ == "__main__":
    #multiprocessing for different P_Eplus
    #read enviroment variable n_cores for number of cores to use or use all available cores
    n_cores=int(os.getenv('n_cores', mp.cpu_count()))
    pool = mp.Pool(n_cores)
    R_j = float(os.getenv('Rj', '0'))
    parameter['R_j'] = R_j
    print(parameter['R_j'])
    clustering_type = os.getenv('clustering_type', 'probability') # "weight" or "probability" as enviroment variable
    parameter['clustering_type'] = clustering_type
    print(parameter['clustering_type'])
    # instead of a range of P_Eplus values, a list of P_Eplus values can be used or single values can be used which
    # can be simulated in parallel processes -> better utilization if done in parallel by scheduler
    #R_Eplus=np.arange(0, parameter['Q']+0.25, 0.25)
    R_Eplus = np.arange(1, 20.25, 0.25)
    #R_Eplus = np.arange(4, 4.5, 0.25)
    Parameterlist = [parameter.copy() for x in R_Eplus]
    for i in range(len(R_Eplus)):
        Parameterlist[i]['R_Eplus'] = R_Eplus[i]
    result = pool.map(execute, Parameterlist)
    #pool.close()
    #result=list(map(execute, Parameterlist))

