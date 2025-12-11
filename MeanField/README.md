# Mean-Field Solvers

The mean-field workflow couples the reusable solver utilities under this package with the `ei_pipeline.py` driver in the repository root. The pipeline sweeps `R_Eplus`, generates effective response functions (ERFs), persists them alongside YAML snapshots, and immediately analyzes the resulting fixpoints.

## `ei_pipeline.py`
- Sweeps `R_Eplus` either from explicit `--r-eplus` values or by specifying `--r-eplus-start`, `--r-eplus-end`, and `--r-eplus-step`. Without flags it reuses the configuration’s `R_Eplus` (falling back to `Q`).
- Controls the ERF grid via `--v-start`, `--v-end`, and `--v-steps`, and retries difficult runs with `--retry-step`.
- Limits execution to a single stage with `--simulation-only` or `--analysis-only`, chooses between per-ERF or aggregated plots with `--plot`, and sets the worker pool size through `--jobs`.
- Adjusts how many populations remain fixed through `--focus-count`, switches to the classic 2Q−1 system via `--full-focus-system`, and forces regeneration with `--overwrite-simulation`.
- Tags every run deterministically using `sim_config/sim_tag_from_cfg`, stores ERFs below `data/<ConnectionType>/RjXX_XX/<tag>/R_EplusXX_XX.pkl`, writes a `params.yaml` snapshot, and drops plots under `plots/`.
- Loads YAML configs from `sim_config/` via `--config my_case` and receives overrides through repeated `-O dotted.path=value` flags (environment variables are ignored by design).

## `rate_system.py`
Defines `RateSystem`, a generic ERF/fixpoint solver that manages population grouping, optional JAX/optimistix acceleration, interpolation helpers, and file I/O utilities:
- `ensure_output_folder` and `serialize_erf` implement the shared on-disk layout.
- `aggregate_data` merges individual ERF pickles back into `all_data_P_Eplus.pkl`.
- `ERFResult` stores the raw sweep samples and solver status.

Run `python -m py_compile MeanField/rate_system.py MeanField/ei_cluster_network.py ei_pipeline.py` for a fast syntax sanity check.

## `ei_cluster_network.py`
Provides `EIClusterNetwork`, the concrete `RateSystem` specialization that:
- Builds mean and variance matrices (plus bias/tau) from the clustering parameters, interpolating between probability and weight clustering via `kappa`.
- Supports automatic grouping of excitatory/inhibitory populations, with `--full-focus-system` disabling the type grouping.
- Mirrors the manuscript’s connectivity rules while keeping the logic isolated from the generic solver.

## `solver_utils.py`
Contains the symbolic-to-numeric compiler for rapid prototyping. It can differentiate SymPy expressions with either cached symbolic derivatives or JAX autodiff and exposes the compiled residual/Jacobian/Hessian bundles through `prepare_system_functions`.
