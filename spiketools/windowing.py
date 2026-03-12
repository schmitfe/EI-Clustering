"""Windowed analyses on canonical `spiketimes` arrays.

Examples
--------
Shared example setup used throughout the documentation:

```python
import numpy as np

from spiketools import gamma_spikes, time_resolved
from spiketools.variability import cv2

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

values, window_time = time_resolved(
    spiketimes,
    window=1000.0,
    func=cv2,
    kwargs={"pool": True},
    tlim=[0.0, 5000.0],
    tstep=250.0,
)
```
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import pylab

from .conversion import (
    binary_to_spiketimes,
    cut_spiketimes,
    get_time_limits,
    spiketimes_to_binary,
)
from .rate import gaussian_kernel, kernel_rate
from .transforms import time_warp

__all__ = [
    "rate_warped_analysis",
    "time_resolved",
    "time_resolved_new",
]


def time_resolved(
    spiketimes,
    window,
    func: Callable,
    kwargs=None,
    tlim: Optional[Sequence[float]] = None,
    tstep=1.0,
):
    r"""
    Apply a function to successive windows of a spike train.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Window size in ms.
    func:
        Callable receiving the cropped `spiketimes` of each window.
    kwargs:
        Optional keyword arguments passed to `func`.
    tlim:
        Optional analysis interval `[tmin, tmax]` in ms.
    tstep:
        Window step size in ms.

    Returns
    -------
    tuple[list, np.ndarray]
        Function outputs and window-center times.

    Definition
    ----------
    If $S$ denotes the spike train and $f$ the analysis function, then

    $$
    y_n = f\left(S \cap [t_n, t_n + W)\right),
    \qquad
    t_n^\mathrm{center} = t_n + \frac{W}{2},
    $$

    where $W$ is the window width and consecutive windows are offset by
    `tstep`.

    Examples
    --------
    >>> spikes = pylab.array([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    >>> values, times = time_resolved(spikes, window=2.0, func=lambda s: s.shape[1], tlim=[0.0, 5.0], tstep=1.0)
    >>> values
    [1, 1, 1, 1]
    >>> times.tolist()
    [0.5, 1.5, 2.5, 3.5]
    """
    if kwargs is None:
        kwargs = {}
    if tlim is None:
        tlim = get_time_limits(spiketimes)
    cut_spikes = cut_spiketimes(spiketimes, tlim)

    tmin = tlim[0]
    tmax = tmin + window
    tcenter = tmin + 0.5 * (tmax - tstep - tmin)

    time = []
    func_out = []
    while tmax <= tlim[1]:
        windowspikes = cut_spiketimes(cut_spikes, [tmin, tmax])
        time.append(tcenter)
        func_out.append(func(windowspikes, **kwargs))

        tmin += tstep
        tmax += tstep
        tcenter += tstep

    return func_out, pylab.array(time)


def time_resolved_new(
    spiketimes,
    window,
    func: Callable,
    kwargs=None,
    tlim: Optional[Sequence[float]] = None,
    tstep=1.0,
):
    r"""
    Windowed evaluation using an intermediate binary representation.

    This variant can be faster for repeated window extraction because the spike
    train is binned once before the loop.

    Definition
    ----------
    This function evaluates the same windowed statistic as `time_resolved(...)`,
    but it first bins the spike train into a count matrix $B$ and reconstructs
    each window from the relevant slice:

    Let $k_{n} = \lfloor t_{n} \rfloor$ be the left window index on the binned
    grid and let $K = \lfloor W \rfloor$ be the corresponding window width in
    bins. Let $B_{n}$ denote the corresponding slice of the count matrix and let
    $\mathcal{C}$ denote the conversion from a binary window back to spike
    times. The statistic is then

    $$
    y_n = f\left(\mathcal{C}(B_{n})\right).
    $$

    Examples
    --------
    >>> spikes = pylab.array([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    >>> values, times = time_resolved_new(spikes, window=2.0, func=lambda s: s.shape[1], tlim=[0.0, 5.0], tstep=1.0)
    >>> values
    [1, 1, 1, 1]
    """
    if kwargs is None:
        kwargs = {}
    if tlim is None:
        tlim = get_time_limits(spiketimes)

    binary, btime = spiketimes_to_binary(spiketimes, tlim=tlim)

    tmin = tlim[0]
    tmax = tmin + window
    tcenter = 0.5 * (tmax + tmin)

    time = []
    func_out = []
    while tmax <= tlim[1]:
        windowspikes = binary_to_spiketimes(
            binary[:, int(tmin) : int(tmax)], btime[int(tmin) : int(tmax)]
        )
        time.append(tcenter)
        func_out.append(func(windowspikes, **kwargs))

        tmin += tstep
        tmax += tstep
        tcenter += tstep
    return func_out, pylab.array(time)


def rate_warped_analysis(
    spiketimes,
    window,
    step=1.0,
    tlim=None,
    rate=None,
    func=lambda x: x.shape[1] / x[1].max(),
    kwargs=None,
    rate_kernel=None,
    dt=1.0,
):
    r"""
    Apply a function after warping time according to instantaneous rate.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Window size in warped time. Use `"full"` to analyze the complete warped
        recording at once.
    step:
        Step size in warped time.
    tlim:
        Optional `[tmin, tmax]` interval in ms.
    rate:
        Optional precomputed `(rate, time)` tuple as returned by
        `kernel_rate(...)`.
    func:
        Statistic to evaluate on the warped spike train.
    kwargs:
        Optional keyword arguments forwarded to `func`.
    rate_kernel:
        Kernel used when `rate` is not provided.
    dt:
        Sampling interval in ms used for the rate estimate.

    Definition
    ----------
    The idea is to replace real time $t$ by warped time $\tau(t)$,

    $$
    \tau(t) = \int_{t_0}^{t} \frac{r(u)}{1000}\,du,
    $$

    where $r(u)$ is an estimated firing rate in spikes/s. Equal distances on
    the warped axis therefore correspond to equal expected spike counts. High-
    rate epochs are stretched, low-rate epochs are compressed, and the chosen
    statistic is then evaluated on this transformed spike train. If windowed
    output is requested, the resulting window centers are mapped back to real
    time before returning.

    Examples
    --------
    >>> spikes = pylab.array([[0.0, 3.0, 6.0], [0.0, 0.0, 0.0]])
    >>> result, warped_duration = rate_warped_analysis(
    ...     spikes,
    ...     window="full",
    ...     func=lambda s: s.shape[1],
    ...     rate=(pylab.array([[500.0, 500.0, 500.0]]), pylab.array([0.0, 1.0, 2.0])),
    ...     dt=1.0,
    ... )
    >>> int(result), round(float(warped_duration), 3)
    (3, 1.5)
    """
    if kwargs is None:
        kwargs = {}
    if rate_kernel is None:
        rate_kernel = gaussian_kernel(50.0)
    if rate is None:
        rate = kernel_rate(spiketimes, rate_kernel, tlim)

    rate, trate = rate
    rate_tlim = [trate.min(), trate.max() + 1]
    spiketimes = cut_spiketimes(spiketimes, rate_tlim)

    ot = pylab.cumsum(rate) / 1000.0 * dt
    w_spiketimes = spiketimes.copy()
    w_spiketimes[0, :] = time_warp(spiketimes[0, :], trate, ot)

    if window == "full":
        return func(w_spiketimes, **kwargs), ot.max()

    result, tresult = time_resolved(
        w_spiketimes, window, func, kwargs, tstep=step
    )
    tresult = time_warp(tresult, ot, trate)
    return result, tresult
