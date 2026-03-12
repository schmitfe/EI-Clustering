"""Rate estimation utilities for spike trains.

All functions accept spike trains in the shared `spiketimes` representation
described in :mod:`spiketools.conversion`.

Examples
--------
Shared example setup used throughout the documentation:

```python
import numpy as np

from spiketools import (
    gamma_spikes,
    gaussian_kernel,
    kernel_rate,
    rate_integral,
    sliding_counts,
    triangular_kernel,
)

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)
```

Inspect both smoothing kernels directly:

```python
gauss = gaussian_kernel(25.0, dt=5.0, nstd=2.0)
# Match the Gaussian cutoff at +/- 50 ms.
tri = triangular_kernel(50.0 / np.sqrt(6.0), dt=5.0)

print(np.round(gauss[:5], 6))
print(np.round(tri[:5], 6))
```

Example output:

```text
[0.000415, 0.000886, 0.00175, 0.003188, 0.005362]
[0.0, 8e-05, 0.00016, 0.00024, 0.00032]
```

Run the rate analysis on the same spike trains:

```python
pooled_rate, rate_time = kernel_rate(spiketimes, gauss, tlim=[0.0, 5000.0], dt=5.0, pool=True)
counts, count_time = sliding_counts(spiketimes, window=250.0, dt=5.0, tlim=[0.0, 5000.0])
integrated = rate_integral(pooled_rate[0], dt=5.0)

print(np.round(pooled_rate[0, :5], 3))
print(np.round(counts.mean(axis=0)[:5], 3))
print(np.round(integrated[:5], 3))
```

Example output:

```text
[13.121, 13.105, 12.739, 12.063, 11.223]
[2.6, 2.5, 2.6, 2.4, 2.5]
[0.066, 0.131, 0.195, 0.255, 0.311]
```
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
    r"""Return a normalized Gaussian kernel for rate smoothing.

    Parameters
    ----------
    sigma:
        Kernel width in ms.
    dt:
        Temporal resolution of the target grid in ms.
    nstd:
        Half-width of the kernel in units of `sigma`.

    Definition
    ----------
    On a grid $t_k = k\,dt$ centered at zero, this function samples the
    truncated Gaussian

    $$
    G_{\sigma, n_\mathrm{std}}(t) =
    \begin{cases}
    \frac{\exp\left(-(t / \sigma)^2\right)}
    {\int_{-n_\mathrm{std}\sigma}^{n_\mathrm{std}\sigma} \exp\left(-(u / \sigma)^2\right)\,du}
    & \text{for } |t| \le n_\mathrm{std}\,\sigma, \\
    0 & \text{otherwise,}
    \end{cases}
    $$

    and rescales the sampled support so that `kernel.sum() * dt = 1`.

    <img src="spiketools_assets/gaussian_kernel_example.png"
         alt="Gaussian kernel example"
         onerror="this.onerror=null;this.src='../spiketools_assets/gaussian_kernel_example.png';" />

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
    r"""Return a normalized triangular smoothing kernel.

    Parameters
    ----------
    sigma:
        Target width parameter in ms.
    dt:
        Temporal resolution of the target grid in ms.

    Definition
    ----------
    Let $a = \sigma \sqrt{6}$. The underlying continuous kernel is

    $$
    T_\sigma(t) = \frac{\max\left(1 - |t| / a, 0\right)}
    {\int_{-\infty}^{\infty} \max\left(1 - |u| / a, 0\right)\,du}
    $$

    and the discrete samples are normalized so that `kernel.sum() * dt = 1`.

    <img src="spiketools_assets/triangular_kernel_example.png"
         alt="Triangular kernel example"
         onerror="this.onerror=null;this.src='../spiketools_assets/triangular_kernel_example.png';" />

    Examples
    --------
    >>> kernel = triangular_kernel(1.0, dt=1.0)
    >>> round(kernel.sum(), 3)
    1.0
    """
    half_base = int(pylab.around(sigma * pylab.sqrt(6)))
    half_kernel = pylab.linspace(0.0, 1.0, half_base + 1)
    kernel = pylab.append(half_kernel, half_kernel[:-1][::-1])
    kernel /= dt * kernel.sum()
    return kernel


def kernel_rate(spiketimes, kernel, tlim=None, dt=1.0, pool=True):
    r"""Estimate smoothed firing rates from `spiketimes`.

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

    Definition
    ----------
    If `b_i[k]` is the discretized spike train on row `i`, then the returned
    estimate is the discrete kernel convolution

    $$
    r_i[n] = 1000 \sum_k b_i[k]\,K[n-k]
    $$

    where $K$ is normalized so that $\sum_k K[k]\,dt = 1$. If `pool=True`, the
    mean spike train across rows is convolved before returning the result.

    <img src="spiketools_assets/kernel_rate_example.png"
         alt="Kernel rate example"
         onerror="this.onerror=null;this.src='../spiketools_assets/kernel_rate_example.png';" />

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
    r"""Count spikes inside a sliding window.

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

    Definition
    ----------
    With $W = \text{window} / dt$ bins and binned spike train `b_i[k]`, this function
    returns

    $$
    C_i[n] = \sum_{k = 0}^{W - 1} b_i[n + k]
    $$

    for each row `i`.

    <img src="spiketools_assets/sliding_counts_example.png"
         alt="Sliding counts example"
         onerror="this.onerror=null;this.src='../spiketools_assets/sliding_counts_example.png';" />

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
    r"""Integrate a rate trace in spikes/s to expected spike counts.

    Parameters
    ----------
    rate:
        One-dimensional rate trace in spikes/s.
    dt:
        Sampling interval in ms.

    Definition
    ----------
    The cumulative expected spike count is

    $$
    I[n] = \sum_{k \le n} r[k]\,dt / 1000
    $$

    which is the discrete-time version of integrating the rate over time.

    <img src="spiketools_assets/rate_integral_example.png"
         alt="Rate integral example"
         onerror="this.onerror=null;this.src='../spiketools_assets/rate_integral_example.png';" />

    Examples
    --------
    >>> rate_integral(np.array([500.0, 500.0]), dt=1.0).tolist()
    [0.5, 1.0]
    """
    return pylab.cumsum(rate / 1000.0) * dt
