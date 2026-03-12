"""Transform spike trains or time-resolved signals onto new time axes.

Examples
--------
Shared example setup used throughout the documentation:

```python
from spiketools import gamma_spikes, gaussian_kernel, kernel_rate, time_warp

rates = [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5]
orders = [1, 2, 2, 3, 1, 2, 3, 2, 1, 3]
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

kernel = gaussian_kernel(25.0, dt=1.0)
rates, rate_time = kernel_rate(spiketimes, kernel, tlim=[0.0, 5000.0], dt=1.0, pool=True)
warped_spikes = time_warp(spiketimes[0], rate_time, rate_time)
```
"""

from __future__ import annotations

import pylab

__all__ = [
    "resample",
    "time_stretch",
    "time_warp",
]


def time_warp(events, told, tnew):
    """Map event times from one timeline onto another by interpolation.

    Parameters
    ----------
    events:
        Event times to transform.
    told:
        Original reference time axis.
    tnew:
        New reference axis aligned to `told`.

    Returns
    -------
    np.ndarray
        Warped event times. Values outside the interpolation range become `nan`.

    Examples
    --------
    >>> time_warp([0.0, 5.0, 10.0], [0.0, 10.0], [0.0, 20.0]).tolist()
    [0.0, 10.0, 20.0]
    """
    return pylab.interp(events, told, tnew, left=pylab.nan, right=pylab.nan)


def time_stretch(spiketimes, stretchstart, stretchend, endtime=None):
    """Stretch trial-wise spike times between two markers to a common duration.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation. The function modifies and returns this
        array in place.
    stretchstart:
        One start time per trial.
    stretchend:
        One end time per trial.
    endtime:
        Common target end time. Defaults to the mean of `stretchend`.

    Examples
    --------
    >>> spikes = pylab.array([[1.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
    >>> stretched = time_stretch(spikes.copy(), pylab.array([0.0, 0.0]), pylab.array([4.0, 2.0]), endtime=4.0)
    >>> stretched[0].tolist()
    [1.0, 3.0, 4.0]
    """
    if endtime is None:
        endtime = stretchend.mean()

    trials = pylab.unique(spiketimes[1, :])

    for i, trial in enumerate(trials):
        trialmask = spiketimes[1, :] == trial
        trialspikes = spiketimes[0, trialmask]
        trialspikes -= stretchstart[i]
        se = stretchend[i] - stretchstart[i]
        trialspikes /= se
        trialspikes *= endtime - stretchstart[i]
        trialspikes += stretchstart[i]
        spiketimes[0, trialmask] = trialspikes

    return spiketimes


def resample(vals, time, new_time):
    """Interpolate a time-resolved signal onto a new time axis.

    Parameters
    ----------
    vals:
        Values sampled on `time`.
    time:
        Original sampling points.
    new_time:
        Target sampling points.

    Examples
    --------
    >>> resample([0.0, 1.0], [0.0, 2.0], [0.0, 1.0, 2.0]).tolist()
    [0.0, 0.5, 1.0]
    """
    if len(vals) > 0:
        return pylab.interp(new_time, time, vals, right=pylab.nan, left=pylab.nan)
    return pylab.ones(new_time.shape) * pylab.nan
