from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "RasterGroup",
    "RasterLabels",
    "plot_spike_raster",
]

GroupIndexer = Union[
    slice,
    range,
    Sequence[int],
    np.ndarray,
    Callable[[np.ndarray], np.ndarray],
]


@dataclass(frozen=True)
class RasterGroup:
    """
    Definition of a neuron group for spike raster plotting.

    Parameters
    ----------
    name:
        Unique identifier for the group. Used to derive default labels.
    ids:
        Specification of neuron membership. May be a slice, range, iterable of ids,
        NumPy array, or callable returning a boolean mask when applied to neuron ids.
    color:
        Matplotlib-compatible color specification for the group's spikes.
    marker:
        Marker symbol passed to ``Axes.scatter``.
    size:
        Marker size passed to ``Axes.scatter``.
    label:
        Optional display label overriding automatic label resolution.

    Examples
    --------
    ```python
    groups = [
        RasterGroup("exc_a", ids=range(0, 5), color="#1f77b4", label="Exc A"),
        RasterGroup("inh", ids=range(5, 7), color="#8B0000", label="Inh"),
    ]
    ```

    Expected output
    ---------------
    Passing `groups` into `plot_spike_raster(...)` draws each group with its own
    color, marker, and label.
    """

    name: str
    ids: GroupIndexer
    color: str = "black"
    marker: str = "."
    size: float = 4.0
    label: Optional[str] = None


@dataclass
class RasterLabels:
    """
    Configuration for annotating neuron groups within a raster plot.

    Parameters
    ----------
    show:
        Whether to annotate groups.
    mapping:
        Explicit mapping of group name -> label text.
    excitatory:
        Fallback text for groups whose name starts with ``"exc"``.
    inhibitory:
        Fallback text for groups whose name starts with ``"inh"``.
    location:
        ``\"right\"`` (default) or ``\"left\"`` indicating where labels should be placed
        relative to the plot area.
    offset:
        Fraction of the x-range used to offset labels from the axis boundary.
    kwargs:
        Additional keyword arguments forwarded to ``Axes.text``.

    Examples
    --------
    ```python
    labels = RasterLabels(
        mapping={"exc_a": "Exc A", "inh": "Inh"},
        location="right",
        kwargs={"fontsize": 9},
    )
    ```

    Expected output
    ---------------
    The raster receives one text label per group at the chosen side of the
    axes.
    """

    show: bool = True
    mapping: Mapping[str, str] = field(default_factory=dict)
    excitatory: Optional[str] = None
    inhibitory: Optional[str] = None
    location: str = "right"
    offset: float = 0.02
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def resolve_label(self, group: RasterGroup) -> Optional[str]:
        if not self.show:
            return None
        if group.name in self.mapping:
            return self.mapping[group.name]
        if group.label is not None:
            return group.label
        gname = group.name.lower()
        if self.excitatory and gname.startswith("exc"):
            return self.excitatory
        if self.inhibitory and gname.startswith("inh"):
            return self.inhibitory
        return None


def plot_spike_raster(
    ax: plt.Axes,
    spike_times_ms: Sequence[float],
    spike_ids: Sequence[int],
    *,
    n_exc: Optional[int] = None,
    n_inh: Optional[int] = None,
    groups: Optional[Sequence[RasterGroup]] = None,
    stride: int = 1,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    align_time: Optional[float] = None,
    time_reference: str = "absolute",
    reference_time: Optional[float] = None,
    marker: str = ".",
    marker_size: float = 4.0,
    exc_color: str = "black",
    inh_color: str = "#8B0000",
    labels: Optional[RasterLabels] = None,
) -> plt.Axes:
    """
    Plot a configurable spike raster on *ax*.

    Parameters
    ----------
    ax:
        Target Matplotlib axes.
    spike_times_ms, spike_ids:
        1-D sequences of spike times (ms) and corresponding neuron ids.
    n_exc, n_inh:
        Sizes of excitatory and inhibitory populations used when *groups* is not provided.
    groups:
        Optional explicit group definitions overriding *n_exc* / *n_inh*.
    stride:
        Keep every ``stride``-th neuron id (e.g., ``stride=10`` shows every 10th neuron).
    t_start, t_end:
        Optional temporal window (after alignment) to display.
    align_time:
        Time (ms) subtracted from all spike times prior to plotting.
    time_reference:
        Either ``\"absolute\"`` (default) or ``\"relative\"``. When ``\"relative\"`` and
        *reference_time* is ``None``, the minimum time after alignment is used as reference.
    reference_time:
        Time (ms) used as reference when ``time_reference=\"relative\"``.
    marker, marker_size:
        Defaults for marker appearance when *groups* is not provided.
    exc_color, inh_color:
        Default colors for excitatory/inhibitory groups.
    labels:
        Optional :class:`RasterLabels` controlling group annotations.

    Examples
    --------
    ```python
    fig, ax = plt.subplots(figsize=(4, 2))
    groups = [
        RasterGroup("exc_a", ids=range(0, 3), color="#1f77b4", label="Exc A"),
        RasterGroup("exc_b", ids=range(3, 5), color="#2ca02c", label="Exc B"),
        RasterGroup("inh", ids=range(5, 7), color="#8B0000", label="Inh"),
    ]
    plot_spike_raster(
        ax,
        spike_times_ms=[5, 8, 11, 13, 21, 23, 29],
        spike_ids=[0, 1, 2, 3, 4, 5, 6],
        groups=groups,
        labels=RasterLabels(location="right", kwargs={"fontsize": 8}),
    )
    ```

    Expected output
    ---------------
    The axes contains one grouped raster with group-specific colors and three
    group labels at the right margin.
    """
    times = np.asarray(spike_times_ms, dtype=float)
    neuron_ids = np.asarray(spike_ids, dtype=int)
    if times.shape != neuron_ids.shape:
        raise ValueError("spike_times_ms and spike_ids must have matching shapes.")

    if align_time is not None:
        times = times - float(align_time)

    time_reference = time_reference.lower()
    if time_reference not in {"absolute", "relative"}:
        raise ValueError("time_reference must be 'absolute' or 'relative'.")
    if time_reference == "relative":
        ref = reference_time
        if ref is None:
            ref = times.min() if times.size else 0.0
        times = times - float(ref)

    # Temporal selection
    mask = np.ones(times.shape, dtype=bool)
    if t_start is not None:
        mask &= times >= float(t_start)
    if t_end is not None:
        mask &= times <= float(t_end)

    # Stride selection
    stride = max(int(stride), 1)
    if stride > 1:
        mask &= (neuron_ids % stride) == 0

    times = times[mask]
    neuron_ids = neuron_ids[mask]

    resolved_groups = _resolve_groups(
        groups=groups,
        n_exc=n_exc,
        n_inh=n_inh,
        marker=marker,
        marker_size=marker_size,
        exc_color=exc_color,
        inh_color=inh_color,
    )

    for group in resolved_groups:
        group_mask = _evaluate_group_mask(group.ids, neuron_ids)
        if not np.any(group_mask):
            continue
        ax.scatter(
            times[group_mask],
            neuron_ids[group_mask],
            s=float(group.size) ** 2,
            c=group.color,
            marker=group.marker,
            linewidths=0,
            edgecolors="none",
        )

    # Limits
    if times.size:
        xmin = float(times.min() if t_start is None else t_start)
        xmax = float(times.max() if t_end is None else t_end)
        if t_start is not None:
            xmin = float(t_start)
        if t_end is not None:
            xmax = float(t_end)
        if xmin == xmax:
            xmax = xmin + 1.0
        ax.set_xlim(xmin, xmax)
    elif t_start is not None or t_end is not None:
        xmin = float(t_start if t_start is not None else 0.0)
        xmax = float(t_end if t_end is not None else xmin + 1.0)
        if xmin == xmax:
            xmax = xmin + 1.0
        ax.set_xlim(xmin, xmax)

    if neuron_ids.size:
        ymin = neuron_ids.min()
        ymax = neuron_ids.max()
    else:
        ymin, ymax = _group_global_bounds(resolved_groups)
    if ymin == ymax:
        ymax = ymin + 1
    padding = max(1, int((ymax - ymin) * 0.02))
    ax.set_ylim(ymin - padding, ymax + padding)

    if labels is not None and labels.show:
        _apply_group_labels(ax, resolved_groups, labels)

    return ax


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_groups(
    *,
    groups: Optional[Sequence[RasterGroup]],
    n_exc: Optional[int],
    n_inh: Optional[int],
    marker: str,
    marker_size: float,
    exc_color: str,
    inh_color: str,
) -> Sequence[RasterGroup]:
    if groups is not None:
        return list(groups)
    if n_exc is None:
        raise ValueError("n_exc must be provided when groups is None.")
    n_exc = int(n_exc)
    if n_inh is None:
        n_inh = 0
    n_inh = int(n_inh)

    resolved = [
        RasterGroup(
            name="exc",
            ids=slice(0, n_exc),
            color=exc_color,
            marker=marker,
            size=marker_size,
            label=None,
        )
    ]
    if n_inh > 0:
        resolved.append(
            RasterGroup(
                name="inh",
                ids=slice(n_exc, n_exc + n_inh),
                color=inh_color,
                marker=marker,
                size=marker_size,
                label=None,
            )
        )
    return resolved


def _evaluate_group_mask(ids_spec: GroupIndexer, neuron_ids: np.ndarray) -> np.ndarray:
    if callable(ids_spec):
        mask = np.asarray(ids_spec(neuron_ids), dtype=bool)
    elif isinstance(ids_spec, slice):
        start = ids_spec.start if ids_spec.start is not None else -np.inf
        stop = ids_spec.stop if ids_spec.stop is not None else np.inf
        step = ids_spec.step if ids_spec.step is not None else 1
        mask = (neuron_ids >= start) & (neuron_ids < stop)
        if step != 1:
            mask &= ((neuron_ids - (start if np.isfinite(start) else neuron_ids)) % step) == 0
    elif isinstance(ids_spec, range):
        start = ids_spec.start
        stop = ids_spec.stop
        step = ids_spec.step
        mask = (neuron_ids >= start) & (neuron_ids < stop) & (((neuron_ids - start) % step) == 0)
    else:
        ids_array = np.asarray(list(ids_spec), dtype=int)
        if ids_array.size == 0:
            return np.zeros_like(neuron_ids, dtype=bool)
        mask = np.isin(neuron_ids, ids_array)
    return mask


def _group_global_bounds(groups: Sequence[RasterGroup]) -> tuple[int, int]:
    bounds = []
    for group in groups:
        b = _group_bounds_from_spec(group.ids)
        if b is not None:
            bounds.append(b)
    if not bounds:
        return (0, 1)
    mins, maxs = zip(*bounds)
    return int(min(mins)), int(max(maxs))


def _group_bounds_from_spec(ids_spec: GroupIndexer) -> Optional[tuple[float, float]]:
    if isinstance(ids_spec, slice):
        start = ids_spec.start if ids_spec.start is not None else 0
        stop = ids_spec.stop if ids_spec.stop is not None else start
        step = ids_spec.step if ids_spec.step is not None else 1
        if stop <= start:
            return None
        return float(start), float(stop - step)
    if isinstance(ids_spec, range):
        if ids_spec.stop <= ids_spec.start:
            return None
        return float(ids_spec.start), float(ids_spec.stop - ids_spec.step)
    if callable(ids_spec):
        return None
    ids_array = np.asarray(list(ids_spec), dtype=float)
    if ids_array.size == 0:
        return None
    return float(np.nanmin(ids_array)), float(np.nanmax(ids_array))


def _apply_group_labels(
    ax: plt.Axes,
    groups: Sequence[RasterGroup],
    labels: RasterLabels,
) -> None:
    x0, x1 = ax.get_xlim()
    span = x1 - x0 if x1 != x0 else 1.0
    offset = labels.offset * span
    if labels.location.lower() == "left":
        x_pos = x0 - offset
        ha = "right"
    else:
        x_pos = x1 + offset
        ha = "left"

    for group in groups:
        text = labels.resolve_label(group)
        if not text:
            continue
        bounds = _group_bounds_from_spec(group.ids)
        if bounds is None:
            continue
        y0, y1 = bounds
        y_center = (y0 + y1) / 2.0
        kwargs = dict(labels.kwargs)
        if "ha" not in kwargs:
            kwargs["ha"] = ha
        if "va" not in kwargs:
            kwargs["va"] = "center"
        if "clip_on" not in kwargs:
            kwargs["clip_on"] = False
        ax.text(x_pos, y_center, text, **kwargs)
