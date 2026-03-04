# Repository Guidelines

## Project Structure & Module Organization
The repository keeps only the runnable entry scripts (`ei_pipeline.py` for mean-field sweeps and `binary_pipeline.py` for stochastic simulations) at the top level. Reusable solvers and helpers live under `MeanField/` (`rate_system.py`, `ei_cluster_network.py`, `solver_utils.py`), while the binary model code remains in `BinaryNetwork/`. Simulation defaults live under `sim_config/` (YAML files plus loader helpers), and configuration changes must go through CLI overrides. Generated data live in `data/<ConnectionType>/RjXX_XX/<config-tag>/` (each with a `params.yaml` snapshot and ERF `.pkl` bundles), while figures belong in `plots/`. Each configuration specifies a `focus_count` so you can control how many populations are grouped at the fixed input; override it via CLI when needed. Use the module-specific READMEs for extended explanations.

## Build, Test, and Development Commands
- `python ei_pipeline.py` — run both simulation and analysis; add `--simulation-only` or `--analysis-only` to limit the stage, `--jobs N` to control parallelism, `--focus-count N` to pin the first N populations, `--full-focus-system` to revert to the classic 2Q−1 system (no type grouping), `--overwrite-simulation` to force regeneration, and `--r-eplus*` flags to control the sweep.
- `python ei_pipeline.py --analysis-only --folder <data-folder>` — analyze existing `.pkl` files without regenerating them.
- `python ei_pipeline.py --config my_run --overwrite parameter.kappa=0.5` — point to a custom YAML config and override individual values via dotted paths.
- `python binary_pipeline.py` — run the binary-network simulation with the current config; use `--warmup-steps`, `--simulation-steps`, etc., to override the `binary.*` settings.
- `python -m py_compile MeanField/rate_system.py MeanField/ei_cluster_network.py ei_pipeline.py` — fast syntax sanity check for the mean-field stack.
Use Python ≥3.10 with `pip install numpy sympy matplotlib pyyaml` (plus optional `jax`, `optimistix`) until a pinned `requirements.txt` is added.

## Coding Style & Naming Conventions
Target PEP 8: four-space indents, snake_case functions, descriptive module names, and UpperCamelCase reserved for future classes. Group related constants and parameter keys (`"kappa"`, `"connection_type"`, `"tau_e"`, etc.) near the top of each file. Document unusual math in short inline comments rather than docstrings.

## Testing Guidelines
No automated suite exists yet, so rely on deterministic reruns. After behavior changes, rerun `ei_pipeline.py --simulation-only` on at least two `R_Eplus` values and confirm monotonic `v_in`/`v_out` trends in the regenerated `.pkl` series. Extend solvers with assert statements covering vector lengths or convergence flags, and consider adding a `tests/` folder with `pytest` cases that mock simple two-cluster configurations.

## Commit & Pull Request Guidelines
History favors concise, imperative messages (“Add core functionality…”). Keep subject lines ≤72 characters and explain parameter defaults or reshaping choices in the body. Pull requests should describe the scenario, enumerate affected driver scripts/modules (`ei_pipeline.py`, `MeanField/rate_system.py`, `MeanField/ei_cluster_network.py`, `binary_pipeline.py` when relevant), attach representative figures from `plots/`, and link upstream issues. Flag compatibility risks—especially new clustering modes—so reviewers know which `.pkl` artifacts or plots to regenerate.

## Security & Configuration Tips
Avoid embedding stateful globals; always thread configuration via the shared `parameter` dict so sweeps remain reproducible. Store large artifacts under the existing `connection_type/...` hierarchy rather than checking them into Git, and scrub temporary notebooks before submission.
