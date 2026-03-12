"""Population-level summary statistics for binned spike matrices.

Examples
--------
Shared example setup used throughout the documentation:

```python
import numpy as np

from spiketools import gamma_spikes, spiketimes_to_binary
from spiketools.population import synchrony

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

binary, _ = spiketimes_to_binary(spiketimes, tlim=[0.0, 5000.0], dt=50.0)
sync = synchrony(binary)
```
"""

from __future__ import annotations

import pylab

__all__ = ["synchrony"]


def synchrony(spikes, ignore_zero_rows=True):
    r"""
    Calculate the Golomb & Hansel (2000) population synchrony measure.

    Parameters
    ----------
    spikes:
        Binned spike matrix with shape `(n_units, n_time_bins)` or a stack of
        such matrices with trials along the first axis.
    ignore_zero_rows:
        If `True`, units with zero spikes are excluded from the statistic.

    Returns
    -------
    float
        Synchrony estimate between `0` and `1` for typical inputs.

    Definition
    ----------
    If $x_i(t)$ is the binned activity of unit $i$ and $\langle \cdot \rangle_i$
    denotes the population average, this implementation returns

    $$
    \chi =
    \sqrt{
    \frac{\mathrm{Var}_t\left[\langle x_i(t) \rangle_i\right]}
    {\left\langle \mathrm{Var}_t[x_i(t)] \right\rangle_i}
    }.
    $$

    Values near `0` indicate largely independent activity, while larger values
    indicate that the population fluctuates together on the chosen time grid.

    Notes
    -----
    This function expects a dense spike-count matrix, not canonical
    `spiketimes`. Convert first with `spiketimes_to_binary(...)` if needed.

    Examples
    --------
    >>> round(float(synchrony(pylab.array([[1, 0, 1], [0, 1, 0]]))), 3)
    0.0
    """
    if len(spikes.shape) > 2:
        return pylab.array([synchrony(s, ignore_zero_rows) for s in spikes]).mean()
    if ignore_zero_rows:
        mask = spikes.sum(axis=1) > 0
        sync = spikes[mask].mean(axis=0).var() / spikes[mask].var(axis=1).mean()
    else:
        sync = spikes.mean(axis=0).var() / spikes.var(axis=1).mean()
    return sync ** 0.5
