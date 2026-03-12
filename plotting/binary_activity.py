"""Helpers for visualizing binary-network activity as onset rasters.

These functions work with binary state traces or diff-log traces. In both
cases the plotting path extracts 0->1 transitions and then reuses the generic
spike-raster plotting backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from .spike_raster import RasterGroup, RasterLabels, plot_spike_raster

__all__ = [
    "BinaryStateSource",
    "collect_binary_onset_events",
    "plot_binary_raster",
]


@dataclass
class BinaryStateSource:
    """Describe how binary neuron states are provided for raster plotting.

    Parameters
    ----------
    inline_states:
        Optional in-memory state matrix with shape `(steps, neurons)`.
    chunk_files:
        Optional paths to `.npy` or `.npz` chunks storing state matrices.
    neuron_count:
        Total number of neurons represented by the source.
    update_log, delta_log:
        Optional diff-log representation where each column corresponds to a
        simulation step.
    initial_state:
        Optional initial state for diff-log based sources.

    Examples
    --------
    Wrap an in-memory state matrix:

    >>> states = np.array([[0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    >>> source = BinaryStateSource.from_array(states)
    >>> source.neuron_count
    2

    Wrap diff logs directly:

    >>> updates = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    >>> deltas = np.array([[1, -1], [0, 1]], dtype=np.int8)
    >>> source = BinaryStateSource.from_diff_logs(updates, deltas, neuron_count=2)
    >>> source.update_log.shape
    (2, 2)
    """

    inline_states: Optional[np.ndarray] = None
    chunk_files: Sequence[Path] = field(default_factory=tuple)
    neuron_count: int = 0
    update_log: Optional[np.ndarray] = None
    delta_log: Optional[np.ndarray] = None
    initial_state: Optional[np.ndarray] = None

    def iter_chunks(self) -> Iterator[np.ndarray]:
        """Yield state chunks with shape `(steps, neurons)`."""
        if self.inline_states is not None and self.inline_states.size:
            yield np.asarray(self.inline_states, dtype=np.uint8)
            return
        for entry in self.chunk_files:
            path = Path(entry)
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            yield np.asarray(data, dtype=np.uint8)

    @classmethod
    def from_array(cls, states: Union[np.ndarray, Iterable[Sequence[int]]]) -> "BinaryStateSource":
        """Convenience helper for wrapping an in-memory state matrix.

        Examples
        --------
        >>> source = BinaryStateSource.from_array([[0, 1], [1, 1]])
        >>> source.neuron_count
        2
        """
        array = np.asarray(states)
        neuron_count = int(array.shape[1]) if array.ndim == 2 else 0
        return cls(inline_states=array, chunk_files=tuple(), neuron_count=neuron_count)

    @classmethod
    def from_diff_logs(
        cls,
        updates: np.ndarray,
        deltas: np.ndarray,
        *,
        neuron_count: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> "BinaryStateSource":
        """Construct a source from diff-log traces.

        Examples
        --------
        >>> source = BinaryStateSource.from_diff_logs(
        ...     np.array([[0, 1]], dtype=np.uint16),
        ...     np.array([[1, -1]], dtype=np.int8),
        ...     neuron_count=2,
        ... )
        >>> source.neuron_count
        2
        """
        update_arr = np.asarray(updates, dtype=np.uint16)
        delta_arr = np.asarray(deltas, dtype=np.int8)
        init_state = None
        if initial_state is not None:
            init_state = np.asarray(initial_state, dtype=np.uint8)
        return cls(
            inline_states=None,
            chunk_files=tuple(),
            neuron_count=int(neuron_count),
            update_log=update_arr,
            delta_log=delta_arr,
            initial_state=init_state,
        )


def collect_binary_onset_events(
    state_source: BinaryStateSource,
    sample_interval: int,
    *,
    window: Optional[Tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect onset events from a binary state stream.

    Parameters
    ----------
    state_source:
        Binary state source, either as full states or diff logs.
    sample_interval:
        Time step between successive recorded samples.
    window:
        Optional `(start, end)` filter in the same units as `sample_interval`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Event times and neuron ids suitable for `plot_spike_raster(...)`.

    Examples
    --------
    >>> source = BinaryStateSource.from_array(np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8))
    >>> times, ids = collect_binary_onset_events(source, sample_interval=5)
    >>> times.tolist()
    [5.0, 10.0]
    >>> ids.tolist()
    [0, 1]
    """
    sample_interval = max(1, int(sample_interval))
    window_start = float(window[0]) if window else None
    window_end = float(window[1]) if window else None
    if state_source.update_log is not None and state_source.delta_log is not None:
        return _collect_from_diff_log(state_source, sample_interval, window=window)
    times: list[np.ndarray] = []
    neurons: list[np.ndarray] = []
    prev_state: Optional[np.ndarray] = None
    sample_index = 0
    for chunk in state_source.iter_chunks():
        block = np.asarray(chunk, dtype=np.uint8)
        if block.ndim != 2 or block.shape[0] == 0:
            continue
        for row in block:
            sample_index += 1
            if prev_state is None:
                prev_state = row.copy()
                continue
            transitions = (prev_state == 0) & (row == 1)
            transition_time = (sample_index - 1) * sample_interval
            prev_state = row.copy()
            if window_end is not None and transition_time > window_end:
                return _finalize(times, neurons)
            if not transitions.any():
                continue
            if window_start is not None and transition_time < window_start:
                continue
            idx = np.flatnonzero(transitions)
            if idx.size == 0:
                continue
            times.append(np.full(idx.size, transition_time, dtype=np.float64))
            neurons.append(idx.astype(np.int64))
    return _finalize(times, neurons)


def plot_binary_raster(
    ax: Axes,
    *,
    state_source: BinaryStateSource,
    sample_interval: int,
    n_exc: int,
    n_inh: Optional[int] = None,
    total_neurons: Optional[int] = None,
    window: Optional[Tuple[float, float]] = None,
    time_scale: float = 1.0,
    stride: int = 1,
    labels: Optional[RasterLabels] = None,
    groups: Optional[Sequence[RasterGroup]] = None,
    marker: str = ".",
    marker_size: float = 4.0,
    empty_text: str = "No neuron onset events",
    **raster_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot a binary-network onset raster.

    The function first extracts onset events from the provided binary state
    source and then forwards those events to `plot_spike_raster(...)`.

    Examples
    --------
    ```python
    fig, ax = plt.subplots(figsize=(4, 2))
    source = BinaryStateSource.from_array(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    )
    times, ids = plot_binary_raster(
        ax,
        state_source=source,
        sample_interval=10,
        n_exc=1,
        n_inh=1,
    )
    ```

    Expected output
    ---------------
    `times` is `array([10., 20.])` and `ids` is `array([0, 1])`. The axes
    shows two onset markers at those coordinates.

    ![Binary raster example](plotting_assets/binary_raster_example.png)
    """
    events = collect_binary_onset_events(state_source, sample_interval, window=window)
    spike_times, spike_ids = events
    if spike_times.size == 0 or spike_ids.size == 0:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return events
    safe_scale = time_scale if time_scale > 0 else 1.0
    scaled_times = spike_times / safe_scale
    t_start = (window[0] / safe_scale) if window else None
    t_end = (window[1] / safe_scale) if window else None

    total = total_neurons or state_source.neuron_count or 0
    n_exc = max(0, int(n_exc))
    if n_inh is None:
        n_inh = max(0, total - n_exc)
    n_inh = max(0, int(n_inh))

    plot_spike_raster(
        ax=ax,
        spike_times_ms=scaled_times,
        spike_ids=spike_ids,
        n_exc=n_exc,
        n_inh=n_inh,
        groups=groups,
        stride=max(1, int(stride)),
        t_start=t_start,
        t_end=t_end,
        marker=marker,
        marker_size=marker_size,
        labels=labels,
        **raster_kwargs,
    )
    if t_start is not None or t_end is not None:
        xmin = t_start if t_start is not None else ax.get_xlim()[0]
        xmax = t_end if t_end is not None else ax.get_xlim()[1]
        if xmin == xmax:
            xmax = xmin + 1.0
        ax.set_xlim(xmin, xmax)
    return events


def _collect_from_diff_log(
    state_source: BinaryStateSource,
    sample_interval: int,
    *,
    window: Optional[Tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    updates = np.asarray(state_source.update_log)
    deltas = np.asarray(state_source.delta_log)
    if updates.ndim != 2 or deltas.shape != updates.shape:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    steps = updates.shape[1]
    per_step = updates.shape[0]
    if steps == 0 or per_step == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    sample_interval = max(1, int(sample_interval))
    window_start = float(window[0]) if window else None
    window_end = float(window[1]) if window else None
    times: list[np.ndarray] = []
    neurons: list[np.ndarray] = []
    for idx in range(steps):
        current_time = idx * sample_interval
        if window_end is not None and current_time > window_end:
            break
        if window_start is not None and current_time < window_start:
            continue
        delta_col = deltas[:, idx]
        onset_mask = delta_col > 0
        if not onset_mask.any():
            continue
        updated_units = updates[:, idx][onset_mask].astype(np.int64, copy=False)
        times.append(np.full(updated_units.size, current_time, dtype=np.float64))
        neurons.append(updated_units)
    if not times:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(times), np.concatenate(neurons)


def _finalize(times: list[np.ndarray], neurons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not times:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(times), np.concatenate(neurons)
