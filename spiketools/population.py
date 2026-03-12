"""Population-level summary statistics for binned spike matrices.

Examples
--------
Shared example setup used throughout the documentation:

```python
from spiketools import gamma_spikes, spiketimes_to_binary
from spiketools.population import synchrony

rates = [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5]
orders = [1, 2, 2, 3, 1, 2, 3, 2, 1, 3]
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

binary, _ = spiketimes_to_binary(spiketimes, tlim=[0.0, 5000.0], dt=50.0)
sync = synchrony(binary)
```
"""

from __future__ import annotations

import pylab

__all__ = ["synchrony"]


def synchrony(spikes, ignore_zero_rows=True):
    """
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
