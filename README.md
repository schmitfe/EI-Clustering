# EI-Clustering Toolkit

The repository now centers on a compact, scriptable workflow that simulates
effective response functions (ERFs) and evaluates their fixpoints in a single
pass.  All shared solver utilities live in `rate_system.py`, the EI-network
specialization resides in `ei_cluster_network.py`, and the command-line entry
point is `ei_pipeline.py`.

## ei_pipeline.py
Run this script to sweep `R_Eplus`, generate ERFs, persist them under
`data/<ConnectionType>/RjXX_XX/<config-tag>/`, and immediately analyze the
stored curves.  Command-line options control the ERF range (`--v-start`,
`--v-end`, `--v-steps`), select explicit `R_Eplus` values (`--r-eplus`,
`--r-eplus-start`, `--r-eplus-end`, `--r-eplus-step`), pick the number of
parallel workers (`--jobs`), limit execution to the simulation or analysis
stage (`--simulation-only`, `--analysis-only`), adjust how many focus
populations remain fixed (`--focus-count`), and force regeneration via
`--overwrite-simulation`.  Without a flag the script performs both stages: it
writes `.pkl` files for each converged ERF and emits the fixpoint scatter plot
plus a pickle summary inside `plots/` and `data/`.
Select alternative parameter sets via `--config my_case` and override individual
values with repeated `-O path=value` arguments (e.g.,
`-O connection_type=poisson -O kappa=0.0`). Environment variables such as
`connection_type=...` are ignored; use CLI overrides instead so configuration
changes remain explicit.  Each data directory receives a deterministic tag
derived from the chosen parameters (excluding `R_Eplus`), so you can store
multiple configurations without conflicts.  The default number of focus
populations is defined via `focus_count` in the YAML and can be overridden with
`--focus-count` for experiments that need multiple synchronized clusters.

## rate_system.py
Defines the general `RateSystem` solver together with helper utilities for ERF
generation, interpolation, fixpoint detection, and sweep serialization.  The
class supports both SciPy and optional JAX/optimistix backends as well as
population-group constraints so multiple populations share identical rates once
declared.  Helper functions such as `ensure_output_folder`, `serialize_erf`, and
`aggregate_data` are reused by the CLI for a consistent layout.

## ei_cluster_network.py
Implements `EIClusterNetwork`, a concrete `RateSystem` derivative that converts
the manuscript’s clustering rules into mean/variance connectivity matrices.  It
reuses the builder logic previously scattered across `connectivit.py` and
`matrix_builder.py` so the EI model’s details remain isolated from the general
solver.

## legacy/
Unchanged: stores the older symbolic Halley/Newton solvers
(`mean_field.py`, `newton.py`, `halley.py`, `fixpoint_iteration.py`) for
reference only.

## Simulation Parameters
Key controls remain the same:

- `P_Eplus`: excitatory cluster strength (0 = unclustered, `Q` = fully clustered)
- `R_j`: inhibitory cluster strength relative to the excitatory one (0–1)
- `kappa`: interpolation between probability (`0`) and weight (`1`) clustering
- `connection_type`: `"bernoulli"`, `"poisson"`, or `"fixed-indegree"`

Defaults live in `sim_config/default_simulation.yaml` and can be overridden via
`--config` (to select another YAML) and repeated `-O path=value` CLI overrides.
Use `python ei_pipeline.py --help` for the full set of options.
