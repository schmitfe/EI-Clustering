# EI-Clustering Toolkit

Mean-field solvers, binary-network simulations, and reference NEST code for the clustered EI model live side by side in this repository. The top level only contains the driver scripts and this overview; each subsystem ships with its own README that covers the full set of options and file layouts.

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
