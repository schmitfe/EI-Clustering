from ClusteredEI_network import *
import BinaryNetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pylab
import plotting
import os
import pickle

if __name__ == "__main__":
    p = np.ones((2, 2)) * 0.2
    g = 1.2
    Q = 10
    # jep=2.0
    Rj = 0.75 #0.75  # 0.85


    jep = 5.  # 1.0




    neuron_parameters = {"N_E": 400, "N_I": 100,
                         "threshold_E": 1.0, "threshold_I": 1.0,
                        "tau_theta_E": 50000., "theta_q_E": 0.0,
                         "tau_theta_I": 50000., "theta_q_I": 0.0,
                         "tau_E": 10.0, "tau_I": 5.0,
                         }




    network = WeightClusteredEI_Network(Q, p, g, jep, Rj, neuron_parameters=neuron_parameters, neuron_model=BinaryNetwork.BinaryNeuronPopulation)

    network.initialize()

    # run x steps of the network
    steps= 25000
    recording = np.zeros((network.N, steps), dtype=np.int16)
    # use tqdm to show progress bar
    for i in tqdm(range(5000)):
        network.run(10)
    for i in tqdm(range(steps)):
        network.run(50)
        recording[:, i] = network.state
    # plot the recording of the network with flipped y axis

    #plt.imshow(recording, interpolation=None, aspect='auto', origin='lower')
    #set colormap to black and white
    #plt.set_cmap('binary')
    #plt.ylabel("NeuronID")
    #plt.xlabel("Time [a.u.]")
    # set title to contain pep and Rj
    # check if network is an instance of ProbClusteredEI_Network


    ms = 4
    mew = 1
    plot = True
    colors = ['k']  # plotting.make_color_list(len(Qs),cmap = 'Greys',minval  =0.25)

    if plot:
        fig = plotting.nice_figure(fig_size_mm=[plotting.biol_cyb_fig_widths[2], plotting.biol_cyb_fig_widths[2] * 0.6],
                                   backend='ps')
        ncols = 2
        nrows = 2
        gs = pylab.GridSpec(nrows, ncols, top=0.85, bottom=0.15, hspace=0.4, left=0.1, right=0.9)

        ms = 4
        mew = 1
        y_offset = 0.04
        x_offset = 0.18

        row = 0
        col = 0
        subplotspec = gs.new_subplotspec((row, col), colspan=1, rowspan=1)
        ax = plotting.simpleaxis(pylab.subplot(subplotspec))
        plt.imshow(recording, interpolation=None, aspect='auto', origin='lower')
        plt.set_cmap('binary')
        plt.ylabel("weight")
        plt.yticks([])
        plotting.ax_label(ax, 'a')
        plt.title("Binary neurons")

    p = np.ones((2, 2)) * 0.1
    g = 1.2
    Q = 10
    # jep=2.0
    Rj = 0.75  # 0.75  # 0.85
    pep = 5.  # 1275


    network = ProbClusteredEI_Network(Q, p, g, pep, Rj, neuron_parameters=neuron_parameters,
                                      neuron_model=BinaryNetwork.BinaryNeuronPopulation)

    network.initialize()

    # run x steps of the network
    recording = np.zeros((network.N, steps), dtype=np.int16)
    # use tqdm to show progress bar
    for i in tqdm(range(5000)):
        network.run(10)
    for i in tqdm(range(steps)):
        network.run(50)
        recording[:, i] = network.state

    row = 1
    col = 0
    subplotspec = gs.new_subplotspec((row, col), colspan=1, rowspan=1)
    ax = plotting.simpleaxis(pylab.subplot(subplotspec))
    plt.imshow(recording, interpolation=None, aspect='auto', origin='lower')
    plt.set_cmap('binary')
    plt.ylabel("prob.")
    plt.yticks([])
    plt.xlabel("Time [a.u.]")
    plotting.ax_label(ax, 'c')


    ## Spiking data
    ms=0.0025
    xlim=[0, 1.05]
    row = 0
    col = 1
    subplotspec = gs.new_subplotspec((row, col), colspan=1, rowspan=1)
    ax = plotting.simpleaxis(pylab.subplot(subplotspec))

    with open('Data/Data_weight.pkl', 'rb') as f:
        Result = pickle.load(f)

    plt.plot(Result['spiketimes'][0]/1000, Result['spiketimes'][1], 'k.', markersize=ms)
    plt.ylim([0, 5000])
    plt.xlim(xlim)
    plt.yticks([])
    #plt.xlabel("Time [s]")
    plotting.ax_label(ax, 'b')
    plt.title("LIF neurons")

    row = 1
    col = 1
    subplotspec = gs.new_subplotspec((row, col), colspan=1, rowspan=1)
    ax = plotting.simpleaxis(pylab.subplot(subplotspec))

    with open('Data/Data_prob.pkl', 'rb') as f:
        Result = pickle.load(f)

    plt.plot(Result['spiketimes'][0]/1000, Result['spiketimes'][1], 'k.', markersize=ms)

    plt.yticks([])
    plt.ylim([0,5000])
    plt.xlim(xlim)
    plt.xlabel("Time [s]")
    plotting.ax_label(ax, 'd')


    # test if figure is already saved, then iterate filename
    name='fig_poster_simulations'
    i=0
    while os.path.isfile(name+str(i)+'.pdf'):
        i+=1
    pylab.savefig(name+str(i)+'.pdf')
    pylab.savefig(name+'.eps')
