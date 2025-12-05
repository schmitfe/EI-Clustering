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
import glob
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
    list_dir = sorted(
        f for f in glob.glob(f"{folder}/*.pkl")
        if os.path.basename(f) != "all_data_P_Eplus.pkl"
    )
    if not list_dir:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    #os.chdir(str(folder))
    #list_dir = os.listdir()
    all_files = pd.read_pickle(list_dir[0])
    for name in list_dir[1:]:
        data = pd.read_pickle(name)
        all_files.update(data)
    name = "all_data_P_Eplus.pkl"
    path_sum = str(folder) + "/" + str(name)
    with open(path_sum, 'wb') as file:
        pickle.dump(all_files, file)
    return path_sum


def plot_fixpoints(folder):
    """
    :param: string of the data file
    :return: plot of the stable / unstable fixpoints for a given R_j where P_Eplus is elem [0,Q]
    """


    #file_name = search_file(folder)
    file_name = "all_data_P_Eplus.pkl"
    data = pd.read_pickle(folder+"/"+file_name)
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
    try:
        all_fix = calc_all_fixpoints_MP(data)#calc_all_fixpoints(data)
    except (PermissionError, OSError) as exc:
        print(f"Falling back to single process due to: {exc}")
        all_fix = calc_all_fixpoints(data)

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
    plt.figure()
    plt.scatter(x_stable, y_stable, color='b', label="stable")
    plt.scatter(x_unstable, y_unstable, color='r', label="unstable")
    plt.xlabel("P_E+")
    plt.ylabel("v_out")
    # plt.xticks(np.arange(0, 10, step=1))
    plt.title("R_j = " + str(R_j))
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig("fixpoints_"+str(clustering_type)+"Clustering_Rj"+str(R_j)+".png")
    plt.close()
    with open("all_fixpoints_"+str(clustering_type)+"Clustering_Rj"+str(R_j)+".pkl", 'wb') as file:
        pickle.dump(all_fix, file)

def plot_rates(folder):
    """
    :param filename: string of the data file
    :return: plot all rates for each P_Eplus, v_in -> v_out
    """

    file_name = search_file(folder)
    data = pd.read_pickle(file_name)
    fig, ax = plt.subplots()
    clustering_type = None
    parameter_ref = None
    for file in data:
        if len(data[file]) != 4:
            data[file] = data[file][:4]
        x_data, y_data, solves, parameter = data[file]
        x_data, y_data = fx.function(x_data, y_data)
        ax.plot(x_data, y_data, label=str(file))
        clustering_type = parameter['clustering_type']
        parameter_ref = parameter
    ax.set_xlabel("v_in")
    ax.set_ylabel("v_out")
    if parameter_ref is not None:
        ax.set_title(f"R_j ={parameter_ref['R_j']}")
    ax.legend()
    plt.savefig(folder+"/all_rates_"+str(clustering_type)+"Clustering_Rj"+str(parameter_ref['R_j'])+".png")
    plt.close(fig)


if __name__ == '__main__':
    #get folder as argument or use default
    if len(sys.argv) > 1:
        folder=sys.argv[1]
    else:
        #folder under the data that will be analyzed is safed
        folder="Weight/R_j0.25"

    plot_rates(folder)
    plot_fixpoints(folder)
