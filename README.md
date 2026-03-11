# EI-Clustering Toolkit

This repository contains all necessary code to reproduce the figures from the draft manuscript "Metastable Neural Assemblies on a Wiring–Weight Continuum" by FJ Schmitt, FL Müller, and MP Nawrot.

Mean-field solvers, binary-network simulations, figure-generation scripts, and reference NEST code for the clustered EI model live side by side in this repository. The top level only contains the driver scripts and this overview; each subsystem ships with its own README that covers the full set of options and file layouts.

## Quick Start
- `python ei_pipeline.py` runs the mean-field ERF sweep plus fixpoint analysis using `sim_config/default_simulation.yaml`. See `MeanField/README.md` for the full flag reference, solver details, and data layout.
- `python binary_pipeline.py` launches the stochastic binary-network simulation with the same default config. See `BinaryNetwork/README.md` for sampling controls and plotting helpers.
- Configuration defaults are defined under `sim_config/`. Always override parameters through the CLI rather than environment variables so runs remain reproducible.
- Simulation outputs populate `data/<ConnectionType>/RjXX_XX/<config-tag>/` with `params.yaml` snapshots, ERF `.pkl` bundles, and optional binary traces, while plots go to `plots/`.

## Modules
- `MeanField/` packages the reusable solvers (`rate_system.py`, `solver_utils.py`) and the EI specialization (`ei_cluster_network.py`) used by `ei_pipeline.py`.
- `BinaryNetwork/` contains the clustered binary model and utilities consumed by `binary_pipeline.py`.
- `NEST/EI_clustered_network/` holds the spiking reference implementation; consult it when porting the model into NEST-based workflows.

Refer to the module-specific READMEs for implementation notes, diagnostics, and advanced usage.



### Commands to generate Figures:
#### Main Text:
- Figure1:
`python Figure1.py --jobs 2 -O binary.warmup_steps=200000 -O binary.simulation_steps=3000000 --panel-window c1:0:3000000 --panel-window c2:0:3000000 --panel-override c1:kappa=0 --panel-override c2:kappa=1 --raster-neuron-step 4 --panel-override c1:binary.seed=3 --panel-override c2:binary.seed=4 -O R_Eplus=7.2 -O R_j=0.8 -O connection_type=poisson -O p0_ee=0.3 -O p0_ei=0.3 -O p0_ie=0.3 -O p0_ii=0.3`
- Figure2:
`python Figure2.py  --rows 0.3 0.1  --columns 0 0.5 1.0  --r-eplus-start 1  --r-eplus-end 20  --r-eplus-step 0.2  --bif-r-eplus-min 1  --bif-r-eplus-max 20  --bif-bisection-tol 0.05  --bif-rj 0.75  --bif-rj 0.5  --bif-avg-connectivity-range 0.01 0.31 0.005  -O Q=20  -O N_E=8000  -O N_I=2000  -O R_j=0.75  --bif-fixpoint-threshold 2  --marker-focus-count 1  --line-focus-counts 5 4 3 2 1  --line-colormap viridis_r --jobs 1 --bif-jobs 1 --bif-rj 0.8`
- Figure3:
`python Figure3.py --config default_simulation -O N_E=8000 -O N_I=2000 -O Q=20     -O R_Eplus=8.0 -O connection_type=poisson --column-override a:kappa=0 --column-override a:R_j=0.81 --column-override b:kappa=0.5 --column-override b:R_j=0.75 --column-override c:kappa=1 --column-override c:R_j=0.75 --stability-filter any --focus-counts 5 4 3 2 1  --warmup-steps 400000 --simulation-steps 6000000 --raster-stride 2 -O binary.seed=1 --jobs 1 --erf-jobs 1 --output-prefix Figures/Figure3 -O p0_ee=0.3 -O p0_ei=0.3 -O p0_ie=0.3 -O p0_ii=0.3 --column-override a:binary.seed=2`
- Figure4:
`python Figure4.py --config default_simulation --kappas 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 --mean-connectivity 0.2 0.25 0.3 --n-networks 20 --n-inits 20 --focus-counts 1 2 3 4 -O R_Eplus=8. --jobs 120 --warmup-steps 200000 --simulation-steps 12000000 --sample-interval 12000 -O N_E=8000 -O N_I=2000 --no-std-shading -O R_j=0.8`
- Figure5:
`python Figure5.py --column-override 0:spiking.net.rep=5.9 --column-override 1:spiking.net.rep=5.2`
#### Supplements:
- Figure S1 (Eigenvalues):
`FigureS1.py -O R_Eplus=8`
- Figure S2 (Event plots of correlation analysis):
`python FigureS2.py --config default_simulation --kappas 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 --mean-connectivity 0.2 0.25 0.3 --n-networks 20 --n-inits 20 --focus-counts 1 2 3 4 -O R_Eplus=8.0 --warmup-steps 200000 --simulation-steps 12000000 --sample-interval 12000 -O N_E=8000 -O N_I=2000 -O R_j=0.8 --analysis-only --init-index 0 --network-index 3 --raster-stride 20 --show-rates --show-raster-labels`
- Sfig3:
`000  -O N_I=2000  -O R_j=0.75  --bif-fixpoint-threshold 2  --marker-focus-count 1  --line-focus-counts 5 4 3 2 1  --line-colormap viridis_r --jobs 1 --bif-jobs 1 --bif-rj 0.8 -O connection_type=fixed_indegree --output Figures/FigureS3`
- Sfig4:

- Sfig5:
`python Figure5.py -O spiking.net.rep=5.0 -O spiking.net.connection_rule=fixed_indegree --output-prefix Figures/FigureS5`