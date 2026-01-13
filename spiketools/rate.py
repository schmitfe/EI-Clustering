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
    """Return a normalized Gaussian kernel."""
    t = pylab.arange(-nstd * sigma, nstd * sigma + dt, dt)
    gauss = pylab.exp(-t ** 2 / sigma ** 2)
    gauss /= gauss.sum() * dt
    return gauss


def triangular_kernel(sigma, dt=1):
    """Return a normalized triangular kernel."""
    half_base = pylab.around(sigma * pylab.sqrt(6))
    half_kernel = pylab.linspace(0.0, 1.0, half_base + 1)
    kernel = pylab.append(half_kernel, half_kernel[:-1][::-1])
    kernel /= dt * kernel.sum()
    return kernel


def kernel_rate(spiketimes, kernel, tlim=None, dt=1.0, pool=True):
    """Compute kernel-smoothed firing rates."""
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
    """Count spikes inside a sliding window of width *window*."""
    if tlim is None:
        tlim = get_time_limits(spiketimes)
    binary, time = spiketimes_to_binary(spiketimes, dt=dt, tlim=tlim)

    kernel = pylab.ones((1, int(window // dt)))
    counts = convolve2d(binary, kernel, "valid")

    dif = time.shape[0] - counts.shape[1]
    time = time[int(np.ceil(dif / 2)) : int(-dif / 2)]

    return counts, time


def rate_integral(rate, dt):
    """Integrate a rate in spikes/s to obtain expected spike counts."""
    return pylab.cumsum(rate / 1000.0) * dt
