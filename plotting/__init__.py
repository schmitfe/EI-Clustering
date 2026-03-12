"""Reusable plotting helpers for figures in this repository.

The package provides three main layers:

- spike rasters via `plot_spike_raster(...)`
- binary-network onset rasters via `plot_binary_raster(...)`
- shared figure styling via `FontCfg`, `style_axes(...)`, and the public palette helpers

Examples
--------

Grouped spike raster example generated from this repository:

![Grouped spike raster example](plotting_assets/grouped_spike_raster_example.png)

Composite showcase generated from this repository:

![Plotting showcase](plotting_assets/plotting_showcase.png)

Using `RasterGroup` and `RasterLabels` explicitly:

```python
import matplotlib.pyplot as plt

from plotting import FontCfg, RasterGroup, RasterLabels, plot_spike_raster, style_axes

groups = [
    RasterGroup("exc_a", ids=range(0, 3), color="#1f77b4", label="Exc A"),
    RasterGroup("exc_b", ids=range(3, 5), color="#2ca02c", label="Exc B"),
    RasterGroup("inh", ids=range(5, 7), color="#8B0000", label="Inh"),
]

fig, ax = plt.subplots(figsize=(4.4, 2.4))
plot_spike_raster(
    ax,
    spike_times_ms=[5, 8, 11, 13, 21, 23, 29],
    spike_ids=[0, 1, 2, 3, 4, 5, 6],
    groups=groups,
    labels=RasterLabels(location="right", kwargs={"fontsize": 8}),
)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Neuron index")
style_axes(ax, FontCfg().resolve())
fig.tight_layout()
```

For binary-network traces, use `BinaryStateSource` together with
`plot_binary_raster(...)`. The generated showcase above includes a full
two-by-two example with grouped spike rasters, binary onset rasters, a discrete
colorbar, and image embedding.

Regenerating docs:

```bash
python scripts/generate_api_docs.py
```
"""

from __future__ import annotations

from .spike_raster import RasterGroup, RasterLabels, plot_spike_raster
from .binary_activity import BinaryStateSource, collect_binary_onset_events, plot_binary_raster
from .image import add_image_ax
from .font import (
    FontCfg,
    add_corner_tag,
    add_panel_label,
    add_panel_labels_column_left_of_ylabel,
    style_axes,
    style_legend,
    style_colorbar,
)
from .time_axis import _time_axis_scale
from .palette import (
    LINE_COLORS,
    DEFAULT_LINE_COLOR,
    _cycle_palette,
    _sample_cmap_colors,
    _prepare_line_color_map,
    _prepare_value_color_map,
    compute_discrete_boundaries,
    draw_listed_colorbar,
)

__pdoc__ = {
    "_time_axis_scale": False,
    "_cycle_palette": False,
    "_sample_cmap_colors": False,
    "_prepare_line_color_map": False,
    "_prepare_value_color_map": False,
}

__all__ = [
    "RasterGroup",
    "RasterLabels",
    "plot_spike_raster",
    "BinaryStateSource",
    "collect_binary_onset_events",
    "plot_binary_raster",
    "add_image_ax",
    "FontCfg",
    "add_corner_tag",
    "add_panel_label",
    "add_panel_labels_column_left_of_ylabel",
    "style_axes",
    "style_colorbar",
    "style_legend",
    "LINE_COLORS",
    "DEFAULT_LINE_COLOR",
    "compute_discrete_boundaries",
    "draw_listed_colorbar",
]
