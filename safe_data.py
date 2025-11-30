import numpy as np
import pickle
import matplotlib.pyplot as plt
import run
import os

'''
In this module, the dynamics of the cluster network is saved. 
The effect of a change of a given parameter (P_Eplus or R_J) is considered.

For this purpose, a loop runs through each value that the desired parameter is to assume.
 Within this loop, an analysis is performed for different input values v1_0. 
 Therefore an ERF (effective response function) is created. 
 To do this a unlinear system of equations (response functions) must be solved. 
 The solution is iteratively approximated with the Newton method 
 which starts with an initial value already roughly approximated by the Halley method.

'''

# analysis range

all_data = {}


def main(filename, parameter, plot=False, start=0., end=1., step_number=20):
    """
    method that, for a given parameter (R_Eplus or P_j),
    performs the analysis for different values of this parameter.
    The results are stored as a dictionary in a pickle file
    in form: {P_Eplus_1:((v_in,v_out, solve, parameter), solve), P_Eplus_2 :((v_in,v_out), solve, parameter), ...}

    :param filename: name of the pickle-file that keeps the saved data
    :param parameter: dictionary that stores the parameter
    :param plot: if True, the results are plotted
    :param start: start value of the input rate v1_0
    :param end: end value of the input rate v1_0
    :param step_number: number of steps between start and end
    :return: None - the results are stored in the pickle file
    """
    R_Eplus = parameter['R_Eplus']
    clustering_type = parameter['clustering_type']
    Q = parameter['Q']
    print('##############################################################')
    print("Simulate Network for R_Eplus = "+str(R_Eplus))

    # foldername
    if clustering_type == "weight":
        foldername = "Weight"
    else:
        foldername = "Probability"

    #create folder if not exist
    if not os.path.exists(foldername):
        os.makedirs(foldername, exist_ok = True)

    # create folder for the actual R_j value
    R_j = parameter["R_j"]
    foldername = os.path.join(foldername, "R_j" + str(R_j))
    if not os.path.exists(foldername):
        os.makedirs(foldername, exist_ok = True)

    initial = np.ones(2 * Q - 1) * 0.01     # first initial start value for the solving method
    v1_0 = start                            # input rate of the fixed cluster
    x_data = []
    y_data = []
    solves = []
    solo_data = {}
    # Loop for different input rates v1_0
    while v1_0 <= end:
        print("-----------------------------")
        print("Set input rate v_in: " + str(v1_0))
        # solve system for the actual v1_0 input by Mean-Field
        x, y, solve = run.simulation(parameter, v1_0, initial, clustering_type=clustering_type)
        # store results
        x_data.append(x)
        y_data.append(y)
        solves.append(solve)
        # solve is the new start-value for the next analysis
        initial = solve
        # set up the input rate
        v1_0 = x + (end / step_number)

    if plot:
        # plot results
        plt.plot(x_data, y_data, label=str(R_Eplus))
        plt.legend()
        plt.plot([0, 1], [0, 1], "black")
        plt.show()
    else:
        print("R_Eplus = " + str(R_Eplus) + " done")

    #Dictionary that stores the data
    solo_data[str(R_Eplus)] = [x_data, y_data, solves, parameter]
    #safe Data-Dictionary as Pickle-file
    #get string with 4 digits after the comma

    R_Eplus = f"{R_Eplus:.2f}".replace(".", "_")
    name = str(filename) + "R_Eplus" + R_Eplus + '.pkl'
    with open(os.path.join(foldername,name), 'wb') as file:
        pickle.dump(solo_data, file)
