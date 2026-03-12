"""Spike-train surrogate generation utilities.

Examples
--------
Generate the shared documentation dataset:

```python
import numpy as np

from spiketools import gamma_spikes

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)
```
"""

from __future__ import annotations

import numpy as np
import pylab

__all__ = ["gamma_spikes"]


def gamma_spikes(rates, order=[1], tlim=[0.0, 1000.0], dt=0.1):
    r"""Generate surrogate spike trains from homogeneous gamma-renewal processes.

    Parameters
    ----------
    rates:
        Scalar or per-train firing rates in spikes/s.
    order:
        Gamma-process shape parameter per train. `1` corresponds to a Poisson
        process, values larger than `1` are more regular, and values below `1`
        are more bursty.
    tlim:
        Two-element time interval `[tmin, tmax]` in ms.
    dt:
        Output quantization step in ms. Generated spike times are rounded to
        this grid for backwards compatibility.

    Returns
    -------
    np.ndarray
        Canonical `spiketimes` array.

    Definition
    ----------
    For each spike train, consecutive inter-spike intervals are drawn
    independently as

    $$
    I_n \sim \mathrm{Gamma}(k, \theta),
    \qquad
    k = \mathrm{order},
    \qquad
    \theta = \frac{1000}{\mathrm{rate}\,k}
    $$

    so that

    $$
    \mathrm{E}[I_n] = \frac{1000}{\mathrm{rate}},
    \qquad
    \mathrm{CV}^2 = \frac{1}{k}.
    $$

    Examples
    --------
    >>> np.random.seed(0)
    >>> spikes = gamma_spikes([10.0], order=[0.5], tlim=[0.0, 10.0], dt=1.0)
    >>> spikes.shape[0]
    2
    """
    rates_arr = np.atleast_1d(np.asarray(rates, dtype=float))
    order_arr = np.atleast_1d(np.asarray(order, dtype=float))

    if rates_arr.size == 1 and order_arr.size > 1:
        rates_arr = np.full(order_arr.shape, rates_arr.item(), dtype=float)
    elif order_arr.size == 1 and rates_arr.size > 1:
        order_arr = np.full(rates_arr.shape, order_arr.item(), dtype=float)
    elif rates_arr.size != order_arr.size:
        raise ValueError("`rates` and `order` must have matching lengths or be scalar.")

    if np.any(rates_arr <= 0.0):
        raise ValueError("All firing rates must be positive.")
    if np.any(order_arr <= 0.0):
        raise ValueError("All gamma orders must be positive.")

    tmin, tmax = float(tlim[0]), float(tlim[1])
    if tmax <= tmin:
        raise ValueError("`tlim` must satisfy tmax > tmin.")
    if dt <= 0.0:
        raise ValueError("`dt` must be positive.")

    spiketimes = [[], []]
    for trial_id, (rate_hz, shape) in enumerate(zip(rates_arr, order_arr)):
        scale_ms = 1000.0 / (rate_hz * shape)
        t = tmin + np.random.gamma(shape=shape, scale=scale_ms)
        generated = []
        while t < tmax:
            rounded = np.round((t - tmin) / dt) * dt + tmin
            if rounded < tmax:
                generated.append(float(rounded))
            t += np.random.gamma(shape=shape, scale=scale_ms)

        if generated:
            spiketimes[0].extend(generated)
            spiketimes[1].extend([float(trial_id)] * len(generated))
        else:
            spiketimes[0].append(pylab.nan)
            spiketimes[1].append(float(trial_id))

    return pylab.array(spiketimes, dtype=float)
