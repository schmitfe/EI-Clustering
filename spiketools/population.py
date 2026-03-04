from __future__ import annotations

import pylab

__all__ = ["synchrony"]


def synchrony(spikes, ignore_zero_rows=True):
    """
    Calculate the Golomb & Hansel (2000) population synchrony measure.

    If *spikes* has more than two dimensions, trials are assumed along the first
    axis and the average synchrony across trials is returned.
    """
    if len(spikes.shape) > 2:
        return pylab.array([synchrony(s, ignore_zero_rows) for s in spikes]).mean()
    if ignore_zero_rows:
        mask = spikes.sum(axis=1) > 0
        sync = spikes[mask].mean(axis=0).var() / spikes[mask].var(axis=1).mean()
    else:
        sync = spikes.mean(axis=0).var() / spikes.var(axis=1).mean()
    return sync ** 0.5
