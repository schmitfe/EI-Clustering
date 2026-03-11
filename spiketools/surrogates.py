"""Spike-train surrogate generation utilities."""

from __future__ import annotations

import numpy as np
import pylab

from .conversion import binary_to_spiketimes

__all__ = ["gamma_spikes"]


def gamma_spikes(rates, order=[1], tlim=[0.0, 1000.0], dt=0.1):
    """Generate surrogate spike trains from a homogeneous gamma process.

    Parameters
    ----------
    rates:
        Scalar or per-train firing rates in spikes/s.
    order:
        Gamma-process order parameter per train. `1` corresponds to a Poisson
        process.
    tlim:
        Two-element time interval `[tmin, tmax]` in ms.
    dt:
        Simulation step in ms.

    Returns
    -------
    np.ndarray
        Canonical `spiketimes` array.

    Examples
    --------
    >>> np.random.seed(0)
    >>> spikes = gamma_spikes([10.0], order=[1], tlim=[0.0, 10.0], dt=1.0)
    >>> spikes.shape[0]
    2
    """
    time = pylab.arange(tlim[0], tlim[1] + dt, dt)
    if len(rates) == 1:
        rates = rates[0] * order[0]
    else:
        if len(order) == 1:
            rates = [r * order[0] for r in rates]
        else:
            rates = [r * o for r, o in zip(rates, order)]
        rates = pylab.tile(pylab.array(rates)[:, pylab.newaxis], (1, len(time)))

    spikes = 1.0 * pylab.rand(rates.shape[0], rates.shape[1]) < rates / 1000.0 * dt

    if len(order) == 1:
        order *= spikes.shape[0]
    for i, o in enumerate(order):
        if o == 1:
            continue
        inds = np.nonzero(spikes[i, :])[0]
        selection = range(0, len(inds), int(o))
        spikes[i, :] = 0.0
        spikes[i, inds[list(selection)]] = 1

    return binary_to_spiketimes(spikes, time)
