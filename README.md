# EI-Clustering (Simulation)
### main_simulation.py
Runs network simulations across a sweep of `P_Eplus` values.  The defaults now
live in `simulation_config.py`, so the script only wires multiprocessing,
environment overrides, and the call into `safe_data.main`.
There are two main parameters that determine the clustering:
#### P_Eplus: Specifies the cluster strength of the E-neurons within a cluster.
         Can take values from 0 (unclustered network) to Q (fully clustered)
#### R_j: indicates the cluster strength of the I-neurons within a cluster in relation to the E neurons.
        Takes values between 0 (no I-clustering) to 1 (same cluster strength as E-neurons).
#### kappa: Mixing coefficient that interpolates between purely probability-based
        clustering (`kappa = 0`) and purely weight-based clustering (`kappa = 1`).
        Intermediate values implement the mixed rule described in the manuscript.

### safe_data.py
Generates input rates from 0 to 1 (configurable) for a fixed cluster and hands
each configuration to `run.simulation`.  The new `generate_erf_curve` helper
returns the computed ERF without touching the filesystem so scripts or notebooks
can decide what to do with the data.  The legacy `main` function now just wraps
that helper with pickle/plotting logic so `main_simulation.py` keeps working.

### run.py / rate_system.py
`run.simulation` instantiates ``RateSystem`` which combines the connectivity
matrices from `connectivit.py`/`matrix_builder.py` and solves the ERF using
either SciPy’s `optimize.root` or the optional `optimistix`+JAX backend.  When a
solution is not found for the requested input, the solver slightly increases
`v_in` and tries again until success or until the input exceeds one.

### Other helper modules
### connectivit.py / matrix_builder.py
Create the connectivity matrix and calculate mean-value and variance of the rates.
### legacy/
Contains the previous symbolic Halley/Newton solvers (`mean_field.py`,
`newton.py`, `halley.py`, `fixpoint_iteration.py`).  They are kept for reference
but are not part of the current simulation pipeline.
# EI Clustering (Analysis)
### main_analysis.py
... used the .pkl-data saved by "safa-data.py" and is able to plot the rates 
    and calculate fixpoints. Saved the plot and also a pickle-file with a dictionary that stores the fixpoints
### fixpoints.py
... first searches for rates where v_in == v_out. These crossings are checked for stability.
