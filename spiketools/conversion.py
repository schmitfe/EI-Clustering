from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Sequence

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
    Return [tmin, tmax] bounds inferred from spike times (inclusive start, exclusive end).
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
    tlim: Sequence[float] | None = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an array of spiketimes into a binary spike matrix (trials x time).

    Parameters
    ----------
    spiketimes:
        Array where row 0 holds spike times in ms and row 1 holds trial indices.
    tlim:
        Two-element sequence defining [tmin, tmax] in ms. Defaults to inferred bounds.
    dt:
        Temporal resolution in ms.
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
    Convert a binary spike matrix and corresponding time axis into spiketimes.
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
    Convert the spiketimes array into a list-of-arrays representation per trial.
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

    spikelist: list[np.ndarray | None] = [None] * len(trials)
    start = 0
    for trial in trials:
        end = bisect_right(orderedspiketimes[1], trial)
        trialspikes = orderedspiketimes[0, start:end]
        start = end
        spikelist[trial] = trialspikes
    return [spikes if spikes is not None else pylab.array([]) for spikes in spikelist]


def cut_spiketimes(spiketimes: np.ndarray, tlim: Sequence[float]) -> np.ndarray:
    """
    Restrict spiketimes to the provided bounds while preserving empty-trial markers.
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
