"""Mean-field solvers and EI-cluster specializations.

The package exposes a generic fixed-point solver in `RateSystem` and a concrete
clustered E/I specialization in `EIClusterNetwork`.

Minimal example:

```python
from MeanField import EIClusterNetwork

system = EIClusterNetwork(parameter, v_focus=0.2)
x, residual, success = system.solve()
rates = system.full_rates_numpy(x)
```

The helpers in `rate_system` are used by the figure pipelines to trace
event-rate functions, store fixpoint bundles, and reuse cached results.

Regenerating docs:

```bash
python scripts/generate_api_docs.py
```
"""

from .rate_system import (
    ERFResult,
    RateSystem,
    aggregate_data,
    ensure_output_folder,
    serialize_erf,
)
from .ei_cluster_network import EIClusterNetwork

__all__ = [
    "RateSystem",
    "ERFResult",
    "EIClusterNetwork",
    "ensure_output_folder",
    "serialize_erf",
    "aggregate_data",
]
