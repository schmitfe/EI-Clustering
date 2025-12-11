# Repository Guidelines

## Project Structure & Module Organization
This pure-Python tree now revolves around three active modules: `rate_system.py` implements the generic solver plus ERF/fixpoint helpers, `ei_cluster_network.py` derives the EI-cluster specialization (folding in the previous connectivity/matrix builders), and `ei_pipeline.py` orchestrates simulations and analysis from the command line. Simulation defaults live under `sim_config/` (YAML files plus loader helpers), and environment-variable overrides are intentionally unsupported so configuration changes always flow through the CLI. Legacy Halley/Newton prototypes remain under `legacy/`. Generated data now live under `data/<ConnectionType>/RjXX_XX/<config-tag>/` (with SHA-based tags that exclude the swept `R_Eplus`) alongside a `params.yaml` snapshot, while figures belong in `plots/`. Each configuration also specifies a `focus_count` so you can control how many populations are grouped at the fixed input; override it via CLI when needed.

## Build, Test, and Development Commands
- `python ei_pipeline.py` — run both simulation and analysis; add `--simulation-only` or `--analysis-only` to limit the stage, `--jobs N` to control parallelism, `--focus-count N` to pin the first N populations, `--full-focus-system` to revert to the classic 2Q−1 system (no type grouping), `--overwrite-simulation` to force regeneration, and `--r-eplus*` flags to control the sweep.
- `python ei_pipeline.py --analysis-only --folder <data-folder>` — analyze existing `.pkl` files without regenerating them.
- `python ei_pipeline.py --config my_run --overwrite parameter.kappa=0.5` — point to a custom YAML config and override individual values via dotted paths.
- `python -m py_compile rate_system.py ei_cluster_network.py ei_pipeline.py` — fast syntax sanity check.
Use Python ≥3.10 with `pip install numpy sympy matplotlib pyyaml` (plus optional `jax`, `optimistix`) until a pinned `requirements.txt` is added.

## Coding Style & Naming Conventions
Target PEP 8: four-space indents, snake_case functions, descriptive module names, and UpperCamelCase reserved for future classes. Group related constants and parameter keys (`"kappa"`, `"connection_type"`, `"tau_e"`, etc.) near the top of each file. Document unusual math in short inline comments rather than docstrings.

## Testing Guidelines
No automated suite exists yet, so rely on deterministic reruns. After behavior changes, rerun `ei_pipeline.py --simulation-only` on at least two `R_Eplus` values and confirm monotonic `v_in`/`v_out` trends in the regenerated `.pkl` series. Extend solvers with assert statements covering vector lengths or convergence flags, and consider adding a `tests/` folder with `pytest` cases that mock simple two-cluster configurations.

## Commit & Pull Request Guidelines
History favors concise, imperative messages (“Add core functionality…”). Keep subject lines ≤72 characters and explain parameter defaults or reshaping choices in the body. Pull requests should describe the scenario, enumerate affected driver scripts (`ei_pipeline.py`, `rate_system.py`, `ei_cluster_network.py`), attach representative figures from `plots/`, and link upstream issues. Flag compatibility risks—especially new clustering modes—so reviewers know which `.pkl` artifacts or plots to regenerate.

## Security & Configuration Tips
Avoid embedding stateful globals; always thread configuration via the shared `parameter` dict so sweeps remain reproducible. Store large artifacts under the existing `connection_type/...` hierarchy rather than checking them into Git, and scrub temporary notebooks before submission.
