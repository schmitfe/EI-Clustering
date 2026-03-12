"""Conversion utilities for the canonical `spiketimes` representation.

The package-wide spike format is a `float` array of shape `(2, n_spikes)`:

- row `0`: spike times in milliseconds
- row `1`: trial, unit, or neuron indices

Examples
--------
Shared example setup used throughout the documentation:

```python
from spiketools import gamma_spikes, spiketimes_to_binary, spiketimes_to_list

rates = [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5]
orders = [1, 2, 2, 3, 1, 2, 3, 2, 1, 3]
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

binary, time = spiketimes_to_binary(spiketimes, tlim=[0.0, 5000.0], dt=50.0)
trains = spiketimes_to_list(spiketimes)
```
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Optional, Sequence

import numpy as np
import pylab

__all__ = [
    "binary_to_spiketimes",
    "cut_spiketimes",
    "get_time_limits",
    "spiketimes_to_binary",
    "spiketimes_to_list",
]


def get_time_limits(spiketimes: np.ndarray) -> list[float]:
    """
    Infer time limits from a `spiketimes` array.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation with shape `(2, n_spikes)`.

    Returns
    -------
    list[float]
        `[tmin, tmax]` with inclusive start and exclusive end. If inference
        fails, `[0.0, 1.0]` is returned as a safe fallback.

    Examples
    --------
    >>> spikes = np.array([[5.0, 8.0], [0.0, 1.0]])
    >>> get_time_limits(spikes)
    [5.0, 9.0]
    """
    try:
        tlim = [
            float(pylab.nanmin(spiketimes[0, :])),
            float(pylab.nanmax(spiketimes[0, :]) + 1),
        ]
    except Exception:
        tlim = [0.0, 1.0]
    if pylab.isnan(tlim).any():
        tlim = [0.0, 1.0]
    return tlim


def spiketimes_to_binary(
    spiketimes: np.ndarray,
    tlim: Optional[Sequence[float]] = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert `spiketimes` into a spike-count matrix on a regular time grid.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation. Row `0` stores times in ms, row `1`
        stores trial or unit indices.
    tlim:
        Two-element sequence defining [tmin, tmax] in ms. Defaults to inferred bounds.
    dt:
        Temporal resolution in ms.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(binary, time)` where `binary.shape == (n_trials, n_bins)` and
        `time` contains the left bin edges in ms.

    Notes
    -----
    The output is not strictly binary if multiple spikes land in the same bin.
    In that case the matrix stores spike counts per bin.

    Examples
    --------
    >>> spikes = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    >>> binary, time = spiketimes_to_binary(spikes, tlim=[0.0, 3.0], dt=1.0)
    >>> binary.astype(int).tolist()
    [[1, 1, 0], [0, 1, 0]]
    >>> time.tolist()
    [-0.5, 0.5, 1.5]
    """
    if tlim is None:
        tlim = get_time_limits(spiketimes)

    time = pylab.arange(tlim[0], tlim[1] + dt, dt).astype(float)
    if dt <= 1:
        time -= 0.5 * float(dt)

    trials = pylab.array([-1] + list(range(int(spiketimes[1, :].max() + 1)))) + 0.5

    tlim_spikes = cut_spiketimes(spiketimes, tlim)
    tlim_spikes = tlim_spikes[:, pylab.isnan(tlim_spikes[0, :]) == False]

    if tlim_spikes.shape[1] > 0:
        binary = pylab.histogram2d(
            tlim_spikes[0, :], tlim_spikes[1, :], [time, trials]
        )[0].T
    else:
        binary = pylab.zeros((len(trials) - 1, len(time) - 1))
    return binary, time[:-1]


def binary_to_spiketimes(binary: np.ndarray, time: Iterable[float]) -> np.ndarray:
    """
    Convert a binned spike matrix into canonical `spiketimes`.

    Parameters
    ----------
    binary:
        Array with shape `(n_trials, n_bins)`. Values larger than `1` are
        interpreted as multiple spikes in the same bin.
    time:
        One time value per bin, typically the left bin edges in ms.

    Returns
    -------
    np.ndarray
        `float` array with shape `(2, n_spikes_or_markers)`.

    Notes
    -----
    Trials with no spikes are retained via placeholder columns
    `[nan, trial_id]`.

    This function assumes that each non-zero entry denotes one or more spike
    events in that time bin. It is not intended for persistent state matrices
    where a neuron stays at `1` across consecutive bins until the next update.

    Examples
    --------
    Convert a spike-count raster on a regular grid:

    >>> binary_to_spiketimes(
    ...     np.array([[1, 0, 2], [0, 0, 0]]),
    ...     [0.0, 1.0, 2.0],
    ... ).tolist()
    [[0.0, 2.0, 2.0, nan], [0.0, 0.0, 0.0, 1.0]]
    """
    time = pylab.array(time)
    spiketimes: list[list[float]] = [[], []]
    max_count = binary.max()
    spikes = binary.copy()
    while max_count > 0:
        if max_count == 1:
            trial, t_index = spikes.nonzero()
        else:
            trial, t_index = pylab.where(spikes == max_count)
        spiketimes[0] += time[t_index].tolist()
        spiketimes[1] += trial.tolist()
        spikes[spikes == max_count] -= 1
        max_count -= 1
    spiketimes_array = pylab.array(spiketimes)

    all_trials = set(range(binary.shape[0]))
    found_trials = set(spiketimes_array[1, :])
    missing_trials = list(all_trials.difference(found_trials))
    for mt in missing_trials:
        spiketimes_array = pylab.append(
            spiketimes_array, pylab.array([[pylab.nan], [mt]]), axis=1
        )
    return spiketimes_array.astype(float)


def spiketimes_to_list(spiketimes: np.ndarray) -> list[np.ndarray]:
    """
    Convert canonical `spiketimes` into one array per trial or unit.

    Parameters
    ----------
    spiketimes:
        Spike array with times in row `0` and indices in row `1`.

    Returns
    -------
    list[np.ndarray]
        `result[i]` contains the spike times of trial or unit `i`.

    Examples
    --------
    >>> spiketimes_to_list(np.array([[1.0, 2.0, 4.0], [0.0, 1.0, 1.0]]))
    [array([1.]), array([2., 4.])]
    """
    if spiketimes.shape[1] == 0:
        return []
    trials = range(int(spiketimes[1, :].max() + 1))
    orderedspiketimes = spiketimes.copy()
    orderedspiketimes = orderedspiketimes[
        :, pylab.isnan(orderedspiketimes[0]) == False
    ]
    spike_order = pylab.argsort(orderedspiketimes[0], kind="mergesort")
    orderedspiketimes = orderedspiketimes[:, spike_order]
    trial_order = pylab.argsort(orderedspiketimes[1], kind="mergesort")
    orderedspiketimes = orderedspiketimes[:, trial_order]

    spikelist: list[Optional[np.ndarray]] = [None] * len(trials)
    start = 0
    for trial in trials:
        end = bisect_right(orderedspiketimes[1], trial)
        trialspikes = orderedspiketimes[0, start:end]
        start = end
        spikelist[trial] = trialspikes
    return [spikes if spikes is not None else pylab.array([]) for spikes in spikelist]


def cut_spiketimes(spiketimes: np.ndarray, tlim: Sequence[float]) -> np.ndarray:
    """
    Restrict `spiketimes` to a time interval.

    Parameters
    ----------
    spiketimes:
        Canonical spike representation.
    tlim:
        Two-element sequence `[tmin, tmax]` in ms. The interval is inclusive at
        the start and exclusive at the end.

    Returns
    -------
    np.ndarray
        Cropped spike array. Empty trials are still represented by
        `[nan, trial_id]` markers.

    Examples
    --------
    >>> cut_spiketimes(
    ...     np.array([[1.0, 3.0, 5.0], [0.0, 0.0, 1.0]]),
    ...     [2.0, 5.0],
    ... ).tolist()
    [[3.0, nan], [0.0, 1.0]]
    """
    alltrials = list(set(spiketimes[1, :]))
    cut_spikes = spiketimes[:, pylab.isfinite(spiketimes[0])]
    cut_spikes = cut_spikes[:, cut_spikes[0, :] >= tlim[0]]

    if cut_spikes.shape[1] > 0:
        cut_spikes = cut_spikes[:, cut_spikes[0, :] < tlim[1]]
    for trial in alltrials:
        if trial not in cut_spikes[1, :]:
            cut_spikes = pylab.append(
                cut_spikes, pylab.array([[pylab.nan], [trial]]), axis=1
            )
    return cut_spikes


_get_tlim = get_time_limits
