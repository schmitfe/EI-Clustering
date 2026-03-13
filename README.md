# EI-Clustering Toolkit

This repository contains all necessary code to reproduce the figures from the draft manuscript "Metastable Neural Assemblies on a Wiring–Weight Continuum" by FJ Schmitt, FL Müller, and MP Nawrot.

Mean-field solvers, binary-network simulations, figure-generation scripts, and reference NEST code for the clustered EI model live side by side in this repository. The top level only contains the driver scripts and this overview; each subsystem ships with its own README that covers the full set of options and file layouts.

## Quick Start
- `python -m pipelines.mean_field` runs the mean-field ERF sweep plus fixpoint analysis using `sim_config/default_simulation.yaml`. See `MeanField/README.md` for the full flag reference, solver details, and data layout.
- `python -m pipelines.binary` launches the stochastic binary-network simulation with the same default config. See `BinaryNetwork/README.md` for sampling controls and plotting helpers.
- Configuration defaults are defined under `sim_config/`. Always override parameters through the CLI rather than environment variables so runs remain reproducible.
- Simulation outputs populate `data/<ConnectionType>/RjXX_XX/<config-tag>/` with `params.yaml` snapshots, ERF `.pkl` bundles, and optional binary traces, while plots go to `plots/`.

## Environments
- Conda environment files for the figure workflows live under [`envs/`](envs/README.md).
- Use [`envs/ei-cluster-core.yml`](envs/ei-cluster-core.yml) for the binary/mean-field figures and shared plotting/docs tooling.
- Use [`envs/ei-cluster-nest.yml`](envs/ei-cluster-nest.yml) when you also need the NEST-based spiking path (`Figure5`, `pipelines/spiking.py`, `NEST/EI_clustered_network/`).
- The split is intentional: `nest-simulator` is only required for the spiking workflow and is easiest to keep isolated from the rest of the stack.
- Recommended solver:
  `mamba env create -f envs/ei-cluster-core.yml`
  `mamba env create -f envs/ei-cluster-nest.yml`
- `conda env create ...` remains a fallback, but `mamba` is the preferred path for environment resolution.

## API Documentation
- The generated API docs live under [`docs/`](docs/index.html) with the entry page at [`docs/index.html`](docs/index.html).
- Regenerate the site locally with `python scripts/generate_api_docs.py`.
- The repository now includes a GitHub Pages workflow at [`.github/workflows/docs.yml`](/home/fschmitt/Documents/git/EI-Clustering/.github/workflows/docs.yml) so the static docs can be published directly from the repo to [Github pages](https://schmitfe.github.io/EI-Clustering/).

## Modules
- `MeanField/` packages the reusable solvers (`rate_system.py`, `solver_utils.py`) and the EI specialization (`ei_cluster_network.py`) used by `pipelines/mean_field.py`.
- `BinaryNetwork/` contains the clustered binary model and utilities consumed by `pipelines/binary.py`.
- `pipelines/` contains the maintained entry points for the mean-field, binary, spiking, and figure-helper workflows.
- `NEST/EI_clustered_network/` holds the spiking reference implementation; consult it when porting the model into NEST-based workflows.

Refer to the module-specific READMEs for implementation notes, diagnostics, and advanced usage.



### Commands to generate Figures:
`Figure4.py` and the fixed-indegree `FigureS4` command below are long-running with the manuscript-scale settings. For cluster runs, use [`scripts/submit_figure4_slurm.py`](/home/fschmitt/Documents/git/EI-Clustering/scripts/submit_figure4_slurm.py), which accepts the normal `Figure4.py` arguments plus optional `--slurm-*` wrapper flags.

#### Main Text:
- Figure1:
`python Figure1.py --panel-override c1:kappa=0 --panel-override c2:kappa=1 --raster-neuron-step 2 -O R_Eplus=7.25 -O R_j=0.8`
- Figure2:
`python Figure2.py --line-colormap viridis_r --columns 0:1:0.5 --r-eplus 1:20:0.1 --bif-rj 0.5 0.75 0.8 --bif-avg-connectivity 0.01:0.31:0.005 --line-focus-counts 5:1:-1`
- Figure3:
`python Figure3.py --jobs 3 -O R_Eplus=7.25 --column-override a:kappa=0 --column-override a:R_j=0.81 --column-override b:kappa=0.5 --column-override c:kappa=1 --stability-filter any --focus-counts 5:1:-1 --warmup-steps 400000 --simulation-steps 6000000 --raster-stride 2`
- Figure4:
`python Figure4.py --kappas 0:1:0.125 --mean-connectivity 0.2:0.3:0.05 --focus-counts 1:4:1 -O R_Eplus=8 -O R_j=0.8 --jobs 120 --simulation-steps 12000000 --sample-interval 12000 --no-std-shading`
- Figure5:
`python Figure5.py --column-override 0:spiking.net.rep=5.9 --column-override 1:spiking.net.rep=5.2`
#### Supplements:
- Figure S1 (Eigenvalues):
`python FigureS1.py -O R_Eplus=7.25 -O R_j=0.8`
- Figure S2 (Event plots of correlation analysis):
`python FigureS2.py --kappas 0:1:0.125 --mean-connectivity 0.2:0.3:0.05 --focus-counts 1:4:1 -O R_Eplus=8 -O R_j=0.8 --simulation-steps 12000000 --sample-interval 12000 --analysis-only --init-index 0 --network-index 3 --raster-stride 20 --show-rates --show-raster-labels`
- Sfig3:
`python Figure2.py --columns 0:1:0.5 --r-eplus 1:20:0.1 --bif-rj 0.5 0.75 0.8 --bif-avg-connectivity 0.01:0.31:0.005 --bif-fixpoint-threshold 2 --line-focus-counts 5:1:-1 --line-colormap viridis_r -O connection_type=fixed_indegree --output Figures/FigureS3`
- Sfig4:
`python Figure4.py --kappas 0:1:0.125 --mean-connectivity 0.2:0.3:0.05 --focus-counts 1:4:1 -O R_Eplus=8 -O R_j=0.8 -O connection_type=fixed_indegree --jobs 20 --simulation-steps 12000000 --sample-interval 12000 --no-std-shading --output-prefix Figures/FigureS4`
- Sfig5:
`python Figure5.py -O spiking.net.rep=5.0 -O spiking.net.connection_rule=fixed_indegree --output-prefix Figures/FigureS5`
