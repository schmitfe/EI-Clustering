# EI-Clustering (Simulation)
### main_simulate.py 
...runs a network simulation to study EI - cluster networks. 
There are two main parameters that determine the clustering: 
#### P_Eplus: Specifies the cluster strength of the E-neurons within a cluster. 
         Can take values from 0 (unclustered network) to Q (fully clustered)
#### R_j: indicates the cluster strength of the I-neurons within a cluster in relation to the E neurons. 
        Takes values between 0 (no I-clustering) to 1 (same cluster strength as E-neurons).
In addition, the type of clustering can be selected, whether the clusters are created by adjusting the weights (weight clustering) 
or by adjusting the connection probabilities (probability clustering).

### Safe_data.py 
...for each specified P_Eplus (multiprocessing) and solves the simulation for one P_Eplus.
The solution for each P_Eplus is saved as a pkl file. Safe_data.py generates input rates from 0 to 1 to which cluster 1 is fixed.
For each input rate, the system is solved using run.py. 

### Run.py 
...uses the halley method to find a good starting value and then the Newton method to find an exact solution. 
If no solution exists for an input, it is slightly increased and the function is repeated recursively until a solution is found
or the input exceeds 1.

### Other helper modules are
### connectivit.py / matrix_builder.py
...creates the connectivity matrix and calculates mean-value and variance of the rates
### fixpoint_iteration.py
...not needed in the actual simulation version, could be used additional to the newton-solver to make the solve more exact
### newton.py / halley.py
...solving the ERF function
### mean_field.py
--- not used in the actual simulation version, can be used alternatively instead of newton-method but very slow. 
    Used Minimization algorithm (sim. to Rost)
# EI Clustering (Analysis)
### main_analysis.py
... used the .pkl-data saved by "safa-data.py" and is able to plot the rates 
    and calculate fixpoints. Saved the plot and also a pickle-file with a dictionary that stores the fixpoints
### fixpoints.py
... first searches for rates where v_in == v_out. These crossings are checked for stability.
