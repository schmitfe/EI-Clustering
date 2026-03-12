"""Variability metrics for spike trains.

Unless noted otherwise, functions operate on canonical `spiketimes` arrays with
times in row `0` and trial or unit indices in row `1`.

Examples
--------
Shared example setup used throughout the documentation:

```python
import numpy as np

from spiketools import gamma_spikes
from spiketools.variability import cv2, cv_two, ff

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)
```

Use one trial from the shared dataset for interval variability. Here `trial_id = 13`
has gamma `order = 3`, so it should be more regular than a Poisson-like trial and
we therefore expect `cv2 < 1`.

```python
trial_id = 13
trial = spiketimes[:, spiketimes[1] == trial_id].copy()
trial[1] = 0

print(f"Trial {trial_id} uses gamma order {orders[trial_id]}; expect cv2 < 1.")
print(round(float(cv2(trial)), 3))
print(round(float(cv_two(trial, min_vals=2)), 3))
```

Example output:

```text
Trial 13 uses gamma order 3.0; expect cv2 < 1.
0.292
0.536
```

For the Fano factor, use only trials `0-9`. These have the same rate but a
bursty gamma `order = 0.2`, so we expect `ff(...) > 1`:

```python
ff_trials = spiketimes[:, spiketimes[1] < 10].copy()

print("Trials 0-9 use rate 6 spikes/s and gamma order 0.2; expect ff > 1.")
print(round(float(ff(ff_trials, tlim=[0.0, 5000.0])), 3))
```

Example output:

```text
Trials 0-9 use rate 6 spikes/s and gamma order 0.2; expect ff > 1.
2.891
```
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pylab

try:
    from Cspiketools import time_resolved_cv_two as _time_resolved_cv_two  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional acceleration
    def _time_resolved_cv_two(*args, **kwargs):
        raise ModuleNotFoundError(
            "The optional C extension 'Cspiketools' is not available. "
            "Compile it via 'python src/setupCspiketools.py build_ext --inplace'."
        )
from .conversion import (
    cut_spiketimes,
    get_time_limits,
    spiketimes_to_binary,
    spiketimes_to_list,
)
from .rate import gaussian_kernel, kernel_rate, rate_integral, sliding_counts
from .transforms import time_warp

__all__ = [
    "cv2",
    "cv_two",
    "ff",
    "lv",
    "kernel_fano",
    "time_resolved_cv2",
    "time_resolved_cv_two",
    "time_warped_cv2",
]


def ff(spiketimes, mintrials=None, tlim=None):
    r"""Compute the Fano factor across trials or units.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    mintrials:
        Optional minimum number of rows required for a valid estimate.
    tlim:
        Optional `[tmin, tmax]` counting window in ms.

    Definition
    ----------
    If `N` is the spike count in the analysis window, the Fano factor is

    $$
    \mathrm{FF} = \frac{\mathrm{Var}[N]}{\mathrm{E}[N]}
    $$

    Examples
    --------
    >>> ff(np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]), tlim=[0.0, 2.0])
    0.0
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)
    dt = tlim[1] - tlim[0]
    binary, _ = spiketimes_to_binary(spiketimes, tlim=tlim, dt=dt)
    counts = binary.sum(axis=1)
    if (counts == 0).all():
        return pylab.nan
    if mintrials is not None and len(counts) < mintrials:
        return pylab.nan
    return counts.var() / counts.mean()


def kernel_fano(spiketimes, window, dt=1.0, tlim=None, components=False):
    """Estimate a sliding-window Fano factor from binned spike counts.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Counting window in ms.
    dt:
        Bin width in ms.
    tlim:
        Optional analysis interval in ms.
    components:
        If `True`, return variance and mean separately instead of their ratio.

    Examples
    --------
    >>> fano, time = kernel_fano(
    ...     np.array([[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
    ...     window=2.0,
    ...     dt=1.0,
    ...     tlim=[0.0, 4.0],
    ... )
    >>> fano.shape, time.shape
    ((3,), (3,))
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)
    if tlim[1] - tlim[0] == window:
        binary, time = spiketimes_to_binary(spiketimes, tlim=tlim)
        counts = binary.sum(axis=1)
        return pylab.array([counts.var() / counts.mean()]), pylab.array(time.mean())

    counts, time = sliding_counts(spiketimes, window, dt, tlim)

    vars_ = counts.var(axis=0)
    means = counts.mean(axis=0)
    fano = pylab.zeros_like(vars_) * pylab.nan
    if components:
        return vars_, means, time
    fano[means > 0] = vars_[means > 0] / means[means > 0]
    return fano, time


def cv2(
    spiketimes,
    pool=True,
    return_all=False,
    bessel_correction=False,
    minvals=0,
):
    r"""Compute the coefficient-of-variation statistic from inter-spike intervals.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    pool:
        If `True`, pool intervals across trials or units before computing the
        statistic. Otherwise compute per row and average.
    return_all:
        If `True`, return per-row values instead of their mean.
    bessel_correction:
        Use `ddof=1` in the interval variance.
    minvals:
        Minimum number of finite intervals required for a per-row value.

    Definition
    ----------
    Despite the historical function name, this returns the squared coefficient
    of variation of the inter-spike intervals $I_n$:

    $$
    \mathrm{CV}^2 = \frac{\mathrm{Var}[I_n]}{\mathrm{E}[I_n]^2}
    $$

    Examples
    --------
    >>> round(float(cv2(np.array([[0.0, 2.0, 5.0], [0.0, 0.0, 0.0]]))), 3)
    0.04
    """
    if spiketimes.shape[1] < 3:
        if return_all:
            return pylab.array([pylab.nan])
        return pylab.nan
    ddof = 1 if bessel_correction else 0
    spikelist = spiketimes_to_list(spiketimes)
    maxlen = max(len(sl) for sl in spikelist)
    if maxlen < 3:
        return pylab.nan
    spikearray = pylab.zeros((len(spikelist), maxlen)) * pylab.nan
    spike_counts = []
    for i, sl in enumerate(spikelist):
        spikearray[i, : len(sl)] = sl
        spike_counts.append(len(sl))
    spike_counts = pylab.array(spike_counts)
    spikearray = spikearray[spike_counts > 2]
    intervals = pylab.diff(spikearray, axis=1)

    if pool:
        var = pylab.nanvar(intervals, ddof=ddof)
        mean_squared = pylab.nanmean(intervals) ** 2
        return var / mean_squared

    intervals[pylab.isfinite(intervals).sum(axis=1) < minvals, :] = pylab.nan
    var = pylab.nanvar(intervals, axis=1, ddof=ddof)
    mean_squared = pylab.nanmean(intervals, axis=1) ** 2
    if return_all:
        return var / mean_squared
    return pylab.nanmean(var / mean_squared)


def _consecutive_intervals(spiketimes):
    order = pylab.argsort(spiketimes[0])
    sorted_spiketimes = spiketimes[:, order]
    order = pylab.argsort(sorted_spiketimes[1], kind="mergesort")
    sorted_spiketimes = sorted_spiketimes[:, order]
    isis = pylab.diff(sorted_spiketimes[0])
    trial = pylab.diff(sorted_spiketimes[1])
    isis[trial != 0] = pylab.nan
    inds = pylab.array([range(i, i + 2) for i in range(len(isis) - 1)])
    try:
        consecutive_isis = isis[inds]
    except Exception:
        consecutive_isis = pylab.zeros((1, 2)) * pylab.nan
    consecutive_isis = consecutive_isis[pylab.isfinite(consecutive_isis.sum(axis=1))]
    return consecutive_isis


def cv_two(spiketimes, min_vals=20):
    r"""Compute the local Cv2 measure from consecutive inter-spike intervals.

    Definition
    ----------
    For consecutive inter-spike intervals $I_n$ and $I_{n+1}$, the local Cv2
    statistic is

    $$
    \mathrm{Cv2} =
    \left\langle
    \frac{2\,|I_{n+1} - I_n|}{I_{n+1} + I_n}
    \right\rangle_n
    $$

    where the average is taken across all finite neighboring interval pairs.

    Examples
    --------
    >>> round(float(cv_two(np.array([[0.0, 2.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0]]), min_vals=2)), 3)
    0.343
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = (
        2
        * pylab.absolute(consecutive_isis[:, 0] - consecutive_isis[:, 1])
        / (consecutive_isis[:, 0] + consecutive_isis[:, 1])
    )
    if len(ms) >= min_vals:
        return pylab.mean(ms)
    return pylab.nan


def lv(spiketimes, min_vals=20):
    r"""Compute the Shinomoto local variation metric.

    Definition
    ----------
    Using the same consecutive inter-spike interval notation as `cv_two(...)`,
    the local variation is

    $$
    \mathrm{LV} =
    \left\langle
    \frac{3\,(I_{n+1} - I_n)^2}{(I_{n+1} + I_n)^2}
    \right\rangle_n
    $$

    Unlike $\mathrm{CV}^2$, this statistic is local in time and is therefore
    less sensitive to slow rate drifts.

    Examples
    --------
    >>> round(float(lv(np.array([[0.0, 2.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0]]), min_vals=2)), 3)
    0.091
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = (
        3
        * (consecutive_isis[:, 0] - consecutive_isis[:, 1]) ** 2
        / (consecutive_isis[:, 0] + consecutive_isis[:, 1]) ** 2
    )
    if len(ms) >= min_vals:
        return pylab.mean(ms)
    return pylab.nan


def time_resolved_cv2(
    spiketimes,
    window=None,
    ot=5.0,
    tlim=None,
    pool=False,
    tstep=1.0,
    bessel_correction=False,
    minvals=0,
    return_all=False,
):
    r"""Estimate Cv2 in sliding windows over time.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Window width in ms. If `None`, it is inferred from the mean ISI.
    ot:
        Multiplicative factor used when inferring `window`.
    tlim:
        Optional analysis interval in ms.
    pool:
        Whether to pool intervals across trials or units.
    tstep:
        Window step size in ms.
    bessel_correction:
        Use `ddof=1` in the interval variance.
    minvals:
        Minimum number of intervals per row.
    return_all:
        Return per-row values for each window instead of a single mean.

    Definition
    ----------
    For a window of width $W$ starting at $t$, this function computes

    $$
    \mathrm{CV}^2(t) =
    \mathrm{CV}^2\left(\{I_n : \text{both spikes of } I_n \in [t, t + W)\}\right)
    $$

    and reports it at the window center. If `window=None`, the code first
    estimates a characteristic window from the mean inter-spike interval and
    the factor `ot`.

    Examples
    --------
    >>> values, time = time_resolved_cv2(
    ...     np.array([[0.0, 2.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0]]),
    ...     window=5.0,
    ...     tlim=[0.0, 10.0],
    ...     tstep=2.0,
    ... )
    >>> values.shape, time.shape
    ((3,), (3,))
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)

    if window is None:
        spikelist = spiketimes_to_list(spiketimes)
        isis = [pylab.diff(spikes) for spikes in spikelist]
        meanisi = [isi.mean() for isi in isis if pylab.isnan(isi.mean()) == False]
        if meanisi:
            window = sum(meanisi) / float(len(meanisi)) * ot
        else:
            window = tlim[1] - tlim[0]

    time = []
    cvs = []
    tmin = tlim[0]
    tmax = tlim[0] + window
    order = pylab.argsort(spiketimes[0])
    ordered_spiketimes = spiketimes[:, order]
    while tmax < tlim[1]:
        start_ind = bisect_right(ordered_spiketimes[0], tmin)
        end_ind = bisect_right(ordered_spiketimes[0], tmax)
        window_spikes = ordered_spiketimes[:, start_ind:end_ind]
        cvs.append(
            cv2(
                window_spikes,
                pool,
                bessel_correction=bessel_correction,
                minvals=minvals,
                return_all=return_all,
            )
        )
        time.append(0.5 * (tmin + tmax))
        tmin += tstep
        tmax += tstep
    if return_all:
        if len(cvs) > 0:
            maxlen = max(len(cv) for cv in cvs)
            cvs = [cv.tolist() + [pylab.nan] * (maxlen - len(cv)) for cv in cvs]
        else:
            cvs = [[]]
    return pylab.array(cvs), pylab.array(time)


def time_warped_cv2(
    spiketimes,
    window=None,
    ot=5.0,
    tstep=1.0,
    tlim=None,
    rate=None,
    kernel_width=50.0,
    nstd=3.0,
    pool=True,
    dt=1.0,
    inspection_plots=False,
    kernel_type="gaussian",
    bessel_correction=False,
    interpolate=False,
    minvals=0,
    return_all=False,
):
    """Estimate Cv2 after compensating for slow rate fluctuations by time warping.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    window:
        Window width in warped time. If `None`, infer from the mean ISI.
    ot:
        Multiplicative factor used for automatic window selection.
    tstep:
        Window step size in ms.
    tlim:
        Optional `[tmin, tmax]` analysis interval.
    rate:
        Optional precomputed `(rate, time)` tuple.
    kernel_width:
        Gaussian kernel width in ms when `rate` is not provided.
    pool:
        Whether to pool intervals across rows in the Cv2 estimate.
    dt:
        Rate-estimation sampling interval in ms.
    inspection_plots:
        Plot the forward and backward time-warp mappings for debugging.
    interpolate:
        If `True`, interpolate the output back onto a regular time grid.
    minvals:
        Minimum number of intervals per row.
    return_all:
        Return per-row values for each window instead of a single mean.

    Examples
    --------
    >>> values, time = time_warped_cv2(
    ...     np.array([[0.0, 3.0, 7.0, 12.0], [0.0, 0.0, 0.0, 0.0]]),
    ...     window=3.0,
    ...     tlim=[0.0, 13.0],
    ...     pool=True,
    ...     dt=1.0,
    ... )
    >>> time.ndim
    1
    """
    del nstd  # Unused but kept for API compatibility.
    del kernel_type  # Unused but kept for API compatibility.

    if tlim is None:
        tlim = get_time_limits(spiketimes)

    if rate is None:
        kernel = gaussian_kernel(kernel_width, dt)
        rate, trate = kernel_rate(spiketimes, kernel, tlim=tlim, dt=dt)
    else:
        trate = rate[1]
        rate = rate[0]

    rate = rate.flatten()
    rate[rate == 0] = 1e-4
    notnaninds = pylab.isnan(rate) == False
    rate = rate[notnaninds]
    trate = trate[notnaninds]

    tdash = rate_integral(rate, dt)
    tdash -= tdash.min()
    tdash /= tdash.max()
    tdash *= trate.max() - trate.min()
    tdash += trate.min()

    transformed_spikes = spiketimes.copy().astype(float)
    transformed_spikes[0, :] = time_warp(transformed_spikes[0, :], trate, tdash)

    if inspection_plots:
        pylab.figure()
        pylab.subplot(1, 2, 1)
        stepsize = max(1, int(spiketimes.shape[1] / 100))
        for i in list(range(0, spiketimes.shape[1], stepsize)) + [-1]:
            if pylab.isnan(transformed_spikes[0, i]) or pylab.isnan(spiketimes[0, i]):
                continue
            idx_real = int(np.argmin(np.abs(trate - spiketimes[0, i])))
            idx_warp = int(np.argmin(np.abs(tdash - transformed_spikes[0, i])))
            pylab.plot(
                [spiketimes[0, i]] * 2,
                [0, tdash[idx_real]],
                "--k",
                linewidth=0.5,
            )
            pylab.plot(
                [0, trate[idx_warp]],
                [transformed_spikes[0, i]] * 2,
                "--k",
                linewidth=0.5,
            )
        pylab.plot(trate, tdash, linewidth=2.0)
        pylab.xlabel("real time")
        pylab.ylabel("transformed time")
        pylab.title("transformation of spiketimes")

    cv2_vals, tcv2 = time_resolved_cv2(
        transformed_spikes,
        window,
        ot,
        None,
        pool,
        tstep,
        bessel_correction=bessel_correction,
        minvals=minvals,
        return_all=return_all,
    )
    if inspection_plots:
        pylab.subplot(1, 2, 2)
        stepsize = max(1, int(len(tcv2) / 50))
        tcv22 = time_warp(tcv2, tdash, trate)
        for i in list(range(0, len(tcv2), stepsize)) + [-1]:
            idx_warp = int(np.argmin(np.abs(tdash - tcv2[i])))
            idx_real = int(np.argmin(np.abs(trate - tcv22[i])))
            pylab.plot(
                [tcv2[i]] * 2,
                [0, trate[idx_warp]],
                "--k",
                linewidth=0.5,
            )
            pylab.plot(
                [0, tdash[idx_real]],
                [tcv22[i]] * 2,
                "--k",
                linewidth=0.5,
            )
        pylab.plot(tdash, trate, linewidth=2)
        pylab.title("re-transformation of cv2")
        pylab.xlabel("transformed time")
        pylab.ylabel("real time")

    tcv2 = time_warp(tcv2, tdash, trate)

    if interpolate:
        time = pylab.arange(tlim[0], tlim[1], dt)
        if len(cv2_vals) == 0 or (return_all and cv2_vals.shape[1] == 0):
            cv2_vals = pylab.zeros_like(time) * pylab.nan
        else:
            if return_all:
                cv2_vals = pylab.array(
                    [
                        pylab.interp(time, tcv2, cv2_vals[:, i], left=pylab.nan, right=pylab.nan)
                        for i in range(cv2_vals.shape[1])
                    ]
                )
            else:
                cv2_vals = pylab.interp(time, tcv2, cv2_vals, left=pylab.nan, right=pylab.nan)
        return cv2_vals, time
    return cv2_vals, tcv2


def time_resolved_cv_two(spiketimes, window=400, tlim=None, min_vals=10, tstep=1):
    r"""Compute rolling Cv2 using the optional C extension.

    Definition
    ----------
    This is the compiled analogue of applying `cv_two(...)` in successive
    windows. For each window $[t, t + W)$ it collects all finite neighboring
    interval pairs $(I_n, I_{n+1})$ whose spikes lie inside the window and
    evaluates

    $$
    \mathrm{Cv2}(t) =
    \left\langle
    \frac{2\,|I_{n+1} - I_n|}{I_{n+1} + I_n}
    \right\rangle_n
    $$

    if at least `min_vals` pairs are available. The window is then shifted by
    `tstep`.

    Notes
    -----
    This function requires the optional `Cspiketools` extension module.

    Expected output
    ---------------
    Returns `(values, time)` with the same window semantics as
    `time_resolved_cv2(...)` when the compiled extension is available.
    """
    return _time_resolved_cv_two(spiketimes, window, tlim, min_vals, tstep)
