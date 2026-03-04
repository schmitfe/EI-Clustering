from __future__ import annotations

import pylab

__all__ = [
    "resample",
    "time_stretch",
    "time_warp",
]


def time_warp(events, told, tnew):
    """Transform events from axis *told* into *tnew*."""
    return pylab.interp(events, told, tnew, left=pylab.nan, right=pylab.nan)


def time_stretch(spiketimes, stretchstart, stretchend, endtime=None):
    """Stretch spike times between start and end markers to a common scale."""
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
    """Interpolate time-resolved quantities onto *new_time*."""
    if len(vals) > 0:
        return pylab.interp(new_time, time, vals, right=pylab.nan, left=pylab.nan)
    return pylab.ones(new_time.shape) * pylab.nan
