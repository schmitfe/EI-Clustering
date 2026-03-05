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
    "_time_axis_scale",
    "LINE_COLORS",
    "DEFAULT_LINE_COLOR",
    "_cycle_palette",
    "_sample_cmap_colors",
    "_prepare_line_color_map",
    "_prepare_value_color_map",
    "compute_discrete_boundaries",
    "draw_listed_colorbar",
]
