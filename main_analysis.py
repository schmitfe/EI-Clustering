"""
From this main module, the data of a clustered (e.g. created in "execute.py) network can be analyzed.
The function "plot_fixpoints" analyzed the stability of the dynamics and returns a plot with stable/ unstable fixpoints.
"""
import itertools
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import fixpoints as fx
import multiprocessing as mp
from functools import partial
import sys
n_cores=int(os.getenv('n_cores', mp.cpu_count()))

#choose clustering_type in "probability" or "weight"
#clustering_type = "weight"#probability"

def search_file(folder):
    """
    Creates a data-file "all_data_P_Eplus.pkl" that includes all solo-data that are stored in "folder"
    :param folder: string of the folder name where data is stored
    :return: name of the data-file that insert all solo-datas
    (stored as a Dictionary wit form: {P_Eplus: {v_in:v_out,...},...}
    """
    os.chdir(str(folder))
    list_dir = os.listdir()
    all_files = pd.read_pickle(list_dir[0])
    for name in list_dir[1:]:
        data = pd.read_pickle(name)
        all_files.update(data)
    name = "all_data_P_Eplus.pkl"
    with open(name, 'wb') as file:
        pickle.dump(all_files, file)
    return str(folder)+"/"+str(name)


def plot_fixpoints(folder):
    """
    :param: string of the data file
    :return: plot of the stable / unstable fixpoints for a given R_j where P_Eplus is elem [0,Q]
    """


    file_name = search_file(folder)
    file_name = "all_data_P_Eplus.pkl"
    data = pd.read_pickle(file_name)
    _, _, _, parameter = list(data.values())[0]
    clustering_type = parameter['clustering_type']
    R_j=parameter['R_j']

    def calc_all_fixpoints(data):
        """
        :param data: dictionary with key: value of a changing parameter and value: file with data of (v_in -> v_out)
        :return: calculates the fixpoints and stability of these fixpoints
        """
        all_fixpoints = {}
        for d in data:
            print("P_Eplus: "+str(d))
            fixpoint = fx.calc_fixpoints(data[d], clustering_type)
            all_fixpoints[d] = fixpoint

        return all_fixpoints

    def calc_all_fixpoints_MP(data):
        """
        :param data: dictionary with key: value of a changing parameter and value: file with data of (v_in -> v_out)
        :return: calculates the fixpoints and stability of these fixpoints
        """
        pool = mp.Pool(n_cores)
        data_list = list(data.values())
        fixpoint=pool.map(partial(fx.calc_fixpoints, clustering_type=clustering_type), data_list)
        #fixpoint=pool.map(lambda x: fx.calc_fixpoints(x, clustering_type), data)
        all_fixpoints = {d:fixpoint[i] for i,d in enumerate(data)}
        pool.close()
        return all_fixpoints
    all_fix = calc_all_fixpoints_MP(data)#calc_all_fixpoints(data)

    x_stable = []
    y_stable = []
    x_unstable = []
    y_unstable = []

    for r in all_fix:
        # print(r)
        for point in all_fix[r]:
           # print(point)
            if all_fix[r][point] == 'stable':
                x_stable.append(float(r))
                y_stable.append(point)
            else:
                x_unstable.append(float(r))
                y_unstable.append(point)
    plt.scatter(x_stable, y_stable, color='b', label="stable")
    plt.scatter(x_unstable, y_unstable, color='r', label="unstable")
    plt.xlabel("P_E+")
    plt.ylabel("v_out")
    # plt.xticks(np.arange(0, 10, step=1))
    plt.title("R_j = " + str(R_j))
    plt.legend()
    plt.savefig("fixpoints_"+str(clustering_type)+"Clustering_Rj"+str(R_j)+".png")
    with open("all_fixpoints_"+str(clustering_type)+"Clustering_Rj"+str(R_j)+".pkl", 'wb') as file:
        pickle.dump(all_fix, file)

def plot_rates(folder):
    """
    :param filename: string of the data file
    :return: plot all rates for each P_Eplus, v_in -> v_out
    """

    file_name = search_file(folder)
    data = pd.read_pickle(file_name)
    for file in data:
        if len(data[file]) != 4:
            data[file] = data[file][:4]
        x_data, y_data, solves, parameter = data[file]
        x_data, y_data = fx.function(x_data, y_data)
        plt.plot(x_data,y_data,label=str(file))
        plt.xlabel("v_in")
        plt.ylabel("v_out")
        plt.title("R_j ="+str(parameter['R_j']))
        plt.legend()
        clustering_type = parameter['clustering_type']
    plt.savefig("all_rates_"+str(clustering_type)+"Clustering_Rj"+str(parameter['R_j'])+".png")
    #plt.show()


if __name__ == '__main__':
    #get folder as argument or use default
    if len(sys.argv) > 1:
        folder=sys.argv[1]
    else:
        #folder under the data that will be analyzed is safed
        folder="Weight/R_j0.25"

    #plot_rates(folder,0)
    plot_fixpoints(folder)
