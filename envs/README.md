Use [ei-cluster-core.yml](ei-cluster-core.yml) for the binary/mean-field figures (`Figure1`-`Figure4`, `FigureS1`, `FigureS2`) and [ei-cluster-nest.yml](ei-cluster-nest.yml) when you also need the NEST-based spiking path (`Figure5`, `pipelines/spiking.py`, `NEST/EI_clustered_network`).

Recommended commands:

```bash
mamba env create -f envs/ei-cluster-core.yml
mamba env create -f envs/ei-cluster-nest.yml
```

The NEST environment is separate on purpose:
- the repository only needs `nest-simulator` for the spiking figure path
- official NEST documentation recommends installing NEST in a dedicated conda environment
- `pairwise_poisson` connectivity is available in NEST `3.7+`, so the NEST file requires `nest-simulator>=3.7,<4`
- the JAX stack is installed via `pip` on purpose so the default environment stays CPU-only instead of resolving CUDA builds on GPU-capable machines

The version ranges are intentionally conservative rather than a full export:
- they match the packages used by the current figure code
- they stay compatible with the Python 3.12 module stack used on the cluster
