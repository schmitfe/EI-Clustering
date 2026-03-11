"""Population-level summary statistics for binned spike matrices."""

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
