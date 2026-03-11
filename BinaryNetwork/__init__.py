"""Binary network simulations for clustered E/I models.

The package contains two main layers:

- `BinaryNetwork.BinaryNetwork`: generic binary-network building blocks
- `BinaryNetwork.ClusteredEI_network`: repository-specific clustered E/I model assembly

Typical workflow:

```python
import numpy as np

from BinaryNetwork import ClusteredEI_network

np.random.seed(3)
network = ClusteredEI_network(parameter)
network.initialize(weight_mode="dense")
initial_state = network.state.copy()

network.enable_diff_logging(steps=1000)
network.run(1000, batch_size=128)
updates, deltas = network.consume_diff_log()

rates = network.population_rates_from_diff_logs(
    initial_state,
    updates,
    deltas,
    sample_interval=10,
)
```

The recommended trace format in this repository stores:

- `initial_state`
- `state_updates`
- `state_deltas`
- optionally derived `spike_times` and `spike_ids`

This keeps simulations compact and allows later reconstruction of full sampled
states, onset events, or population rates.

Regenerating docs:

```bash
python scripts/generate_api_docs.py
```
"""

from .BinaryNetwork import (
    BinaryNeuronPopulation,
    BackgroundActivity,
    PairwiseBernoulliSynapse,
    PoissonSynapse,
    FixedIndegreeSynapse,
    AllToAllSynapse,
    BinaryNetwork,
)
from .ClusteredEI_network import ClusteredEI_network

__all__ = [
    "BinaryNeuronPopulation",
    "BackgroundActivity",
    "PairwiseBernoulliSynapse",
    "PoissonSynapse",
    "FixedIndegreeSynapse",
    "AllToAllSynapse",
    "BinaryNetwork",
    "ClusteredEI_network",
]
