# Binary Network Simulation

`pipelines/binary.py` drives the stochastic binary-network counterpart of the clustered EI model while reusing the same YAML configuration and override semantics as the mean-field solver.

## Running the pipeline
- Invoke `python -m pipelines.binary [-O path=value ...]` using the configs from `sim_config/`. Use the helper flags to override the common binary settings directly: `--warmup-steps`, `--simulation-steps`, `--sample-interval`, `--batch-size`, `--seed`, `--output-name`, and `--plot-activity`.
- The script respects global clustering parameters (`kappa`, `connection_type`, `R_Eplus`, `R_j`, etc.) and mirrors the mixed probability/weight scheme from the mean-field solver. Bernoulli, Poisson, and fixed-indegree synapse generation are all supported.
- Simulation-specific defaults live under the `binary` section of the YAML (`warmup_steps`, `simulation_steps`, `sample_interval`, `batch_size`, `seed`, `output_name`, and optional `plot_activity`). Override nested fields via dotted `-O binary.sample_interval=25` arguments if the dedicated CLI flag is insufficient.

## Outputs
- Downsampled population activities are stored in `data/<ConnectionType>/RjXX_XX/<tag>/binary/` along with a YAML summary reuse the same deterministic folder naming as the mean-field pipeline.
- Each `activity_trace.npz` bundle now follows the legacy binary trace layout: `rates`, `times`, `names`, `initial_state`, `state_updates`, and `state_deltas`.
- The trace stores the recorded state only once at the simulation start; subsequent activity is represented as per-step neuron updates and their `{-1,0,+1}` deltas. This is substantially smaller than storing full state matrices and allows cluster-rate reconstruction directly from the diff log.
- Pass `--plot-activity` (or set `binary.plot_activity` in the config) to reconstruct a state matrix from the diff log and generate a neuron-by-time heatmap on demand.

## Initial conditions
Add an `initial_activity` block to the YAML to align binary simulations with mean-field predictions:

```yaml
initial_activity:
  mode: deterministic  # or "bernoulli"
  excitatory: [0.3, 0.1, 0.1, 0.05]
  inhibitory: 0.02
```

Use scalars or length-`Q` lists. Deterministic mode matches the desired fraction exactly, while Bernoulli samples each neuron independently with the given probability.
