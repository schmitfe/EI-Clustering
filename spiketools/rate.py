"""Rate estimation utilities for spike trains.

All functions accept spike trains in the shared `spiketimes` representation
described in :mod:`spiketools.conversion`.
"""

from __future__ import annotations

import numpy as np
import pylab
from scipy.signal import convolve2d

from .conversion import (
    get_time_limits,
    spiketimes_to_binary,
)

__all__ = [
    "gaussian_kernel",
    "kernel_rate",
    "rate_integral",
    "sliding_counts",
    "triangular_kernel",
]


def gaussian_kernel(sigma, dt=1.0, nstd=3.0):
    """Return a normalized Gaussian kernel for rate smoothing.

    Parameters
    ----------
    sigma:
        Kernel width in ms.
    dt:
        Temporal resolution of the target grid in ms.
    nstd:
        Half-width of the kernel in units of `sigma`.

    Examples
    --------
    >>> kernel = gaussian_kernel(2.0, dt=1.0, nstd=1.0)
    >>> round(kernel.sum(), 3)
    1.0
    """
    t = pylab.arange(-nstd * sigma, nstd * sigma + dt, dt)
    gauss = pylab.exp(-t ** 2 / sigma ** 2)
    gauss /= gauss.sum() * dt
    return gauss


def triangular_kernel(sigma, dt=1):
    """Return a normalized triangular smoothing kernel.

    Parameters
    ----------
    sigma:
        Target width parameter in ms.
    dt:
        Temporal resolution of the target grid in ms.

    Examples
    --------
    >>> kernel = triangular_kernel(1.0, dt=1.0)
    >>> round(kernel.sum(), 3)
    1.0
    """
    half_base = pylab.around(sigma * pylab.sqrt(6))
    half_kernel = pylab.linspace(0.0, 1.0, half_base + 1)
    kernel = pylab.append(half_kernel, half_kernel[:-1][::-1])
    kernel /= dt * kernel.sum()
    return kernel


def kernel_rate(spiketimes, kernel, tlim=None, dt=1.0, pool=True):
    """Estimate smoothed firing rates from `spiketimes`.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation with times in ms.
    kernel:
        One-dimensional kernel normalized to unit integral in seconds.
    tlim:
        Optional `[tmin, tmax]` interval in ms.
    dt:
        Bin width in ms used for discretization before convolution.
    pool:
        If `True`, average across trials or units before convolution and return
        a single population rate trace. If `False`, return one trace per row.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(rates, time)` where `rates` is in spikes/s.

    Examples
    --------
    >>> spikes = np.array([[0.0, 2.0], [0.0, 0.0]])
    >>> kernel = gaussian_kernel(1.0, dt=1.0, nstd=1.0)
    >>> rates, time = kernel_rate(spikes, kernel, tlim=[0.0, 4.0], dt=1.0, pool=False)
    >>> rates.shape, time.shape
    ((1, 2), (2,))
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)

    binary, time = spiketimes_to_binary(spiketimes, tlim, dt)

    if pool:
        binary = binary.mean(axis=0)[pylab.newaxis, :]

    rates = convolve2d(binary, kernel[pylab.newaxis, :], "same")
    kwidth = len(kernel)
    rates = rates[:, int(kwidth / 2) : -int(kwidth / 2)]
    time = time[int(kwidth / 2) : -int(kwidth / 2)]
    return rates * 1000.0, time


def sliding_counts(spiketimes, window, dt=1.0, tlim=None):
    """Count spikes inside a sliding window.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Window width in ms.
    dt:
        Step size of the discretized binary representation in ms.
    tlim:
        Optional `[tmin, tmax]` interval in ms.

    Examples
    --------
    >>> spikes = np.array([[0.0, 2.0], [0.0, 0.0]])
    >>> counts, time = sliding_counts(spikes, window=2.0, dt=1.0, tlim=[0.0, 4.0])
    >>> counts.astype(int).tolist()
    [[1, 1, 1]]
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)
    binary, time = spiketimes_to_binary(spiketimes, dt=dt, tlim=tlim)

    kernel = pylab.ones((1, int(window // dt)))
    counts = convolve2d(binary, kernel, "valid")

    dif = time.shape[0] - counts.shape[1]
    time = time[int(np.ceil(dif / 2)) : int(-dif / 2)]

    return counts, time


def rate_integral(rate, dt):
    """Integrate a rate trace in spikes/s to expected spike counts.

    Parameters
    ----------
    rate:
        One-dimensional rate trace in spikes/s.
    dt:
        Sampling interval in ms.

    Examples
    --------
    >>> rate_integral(np.array([500.0, 500.0]), dt=1.0).tolist()
    [0.5, 1.0]
    """
    return pylab.cumsum(rate / 1000.0) * dt
