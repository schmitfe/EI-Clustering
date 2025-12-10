from ClusteredEI_network import *
import BinaryNetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    p = np.ones((2, 2)) * 0.1
    g = 1.2
    Q = 10
    # jep=2.0
    Rj = 0.75 #0.75  # 0.85

    pep = 5.5# 1275
    jep = 4.5  # 1.0
    prob= True



    neuron_parameters = {"N_E": 400, "N_I": 100,
                         "threshold_E": 1.0, "threshold_I": 1.0,
                        "tau_theta_E": 50000., "theta_q_E": 0.0,
                         "tau_theta_I": 50000., "theta_q_I": 0.0,
                         "tau_E": 10.0, "tau_I": 5.0,
                         }

    #model without SFA, even if theta_q is unequal to zero
    if prob:
        network = ProbClusteredEI_Network(Q, p, g, pep, Rj, neuron_parameters=neuron_parameters, neuron_model=BinaryNetwork.BinaryNeuronPopulation)
    else:
        network = WeightClusteredEI_Network(Q, p, g, jep, Rj, neuron_parameters=neuron_parameters, neuron_model=BinaryNetwork.BinaryNeuronPopulation)

    #fig, ax= plt.subplots(1, 3, figsize=(15, 5))
    network.initialize()
    #print(network.state)
    #print(network.weights)
    #ax[0].imshow(network.weights)


    network.homogenize_within_cluster()
    #ax[1].imshow(network.weights)

    network.homogenize_between_cluster()
    #ax[2].imshow(network.weights)

    #plt.show()

    # run x steps of the network
    steps= 25000
    recording = np.zeros((network.N, steps))
    # use tqdm to show progress bar
    for i in tqdm(range(5000)):
        network.run(2)
    for i in tqdm(range(steps)):
        network.run(100)
        recording[:, i] = network.state
    # plot the recording of the network with flipped y axis

    plt.imshow(recording, interpolation=None, aspect='auto', origin='lower')
    #set colormap to black and white
    plt.set_cmap('binary')
    plt.ylabel("NeuronID")
    plt.xlabel("Time [a.u.]")
    # set title to contain pep and Rj
    # check if network is an instance of ProbClusteredEI_Network
    if isinstance(network, ProbClusteredEI_Network):
        plt.title("pep = " + str(pep) + ", Rj = " + str(Rj))
    else:
        plt.title("jep = " + str(jep) + ", Rj = " + str(Rj))
    plt.show()