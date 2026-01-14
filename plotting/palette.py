from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .font import FontCfg

__all__ = [
    "LINE_COLORS",
    "DEFAULT_LINE_COLOR",
    "_cycle_palette",
    "_sample_cmap_colors",
    "_prepare_line_color_map",
    "compute_discrete_boundaries",
    "draw_listed_colorbar",
]

LISTED_CATEGORICAL_LIMIT = 32


def _cmyk_to_rgb_hex(c: float, m: float, y: float, k: float) -> str:
    r = 1.0 - min(1.0, c + k)
    g = 1.0 - min(1.0, m + k)
    b = 1.0 - min(1.0, y + k)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


LINE_COLORS = (
    _cmyk_to_rgb_hex(0.8, 0.1, 0.0, 0.1),
    _cmyk_to_rgb_hex(0.0, 0.6, 0.2, 0.1),
    _cmyk_to_rgb_hex(0.1, 0.2, 0.8, 0.1),
    _cmyk_to_rgb_hex(0.0, 0.4, 0.8, 0.2),
    _cmyk_to_rgb_hex(0.6, 0.0, 0.1, 0.2),
)
DEFAULT_LINE_COLOR = LINE_COLORS[0]


def _cycle_palette(palette: Sequence[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if not palette:
        raise ValueError("Cannot cycle an empty palette.")
    repeats = (count + len(palette) - 1) // len(palette)
    return list(palette * repeats)[:count]


def _sample_cmap_colors(colormap: str, count: int) -> List[str]:
    if count <= 0:
        return []
    try:
        cmap = plt.get_cmap(colormap)
    except ValueError as exc:
        raise SystemExit(f"Unknown matplotlib colormap '{colormap}'.") from exc
    categorical_colors = getattr(cmap, "colors", None)
    use_categorical = (
        isinstance(cmap, mcolors.ListedColormap)
        and categorical_colors is not None
        and len(categorical_colors) <= LISTED_CATEGORICAL_LIMIT
    )
    if use_categorical:
        base_colors = list(categorical_colors)
        repeats = (count + len(base_colors) - 1) // len(base_colors)
        selected = (base_colors * repeats)[:count]
    else:
        if count == 1:
            positions = [0.5]
        else:
            positions = np.linspace(0.0, 1.0, count)
        selected = [cmap(float(pos)) for pos in positions]
    return [mcolors.to_hex(color) for color in selected]


def _prepare_line_color_map(
    focus_counts: Sequence[int],
    *,
    colormap: str | None = None,
    palette: Sequence[str] | None = None,
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    mapping: Dict[int, str] = {}
    entries: List[Tuple[int, str]] = []
    ordered_counts = sorted({int(fc) for fc in focus_counts})
    if not ordered_counts:
        return mapping, entries
    palette_source: Sequence[str] = palette if palette is not None else LINE_COLORS
    if colormap:
        colors = _sample_cmap_colors(colormap, len(ordered_counts))
    else:
        colors = _cycle_palette(palette_source, len(ordered_counts))
    for focus_count, color in zip(ordered_counts, colors):
        mapping[int(focus_count)] = color
        entries.append((int(focus_count), color))
    return mapping, entries


def compute_discrete_boundaries(values: Sequence[float]) -> List[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return []
    ordered = sorted(dict.fromkeys(numeric))
    if len(ordered) == 1:
        val = float(ordered[0])
        return [val - 0.5, val + 0.5]
    boundaries = [ordered[0] - 0.5]
    for prev_val, next_val in zip(ordered[:-1], ordered[1:]):
        midpoint = (prev_val + next_val) / 2.0
        boundaries.append(midpoint)
    boundaries.append(ordered[-1] + 0.5)
    return boundaries


def draw_listed_colorbar(
    fig: "Figure",
    axis: "Axes",
    entries: Sequence[Tuple[float, str]],
    *,
    font_cfg: "FontCfg",
    label: str,
    orientation: str = "vertical",
    height_fraction: float | None = None,
    use_parent_axis: bool = False,
    label_kwargs: Mapping[str, Any] | None = None,
) -> None:
    if not entries:
        axis.set_axis_off()
        return
    ticks = [float(value) for value, _ in entries]
    colors = [color for _, color in entries]
    cmap = mcolors.ListedColormap(colors)
    boundaries = compute_discrete_boundaries(ticks)
    if len(boundaries) < 2:
        single = ticks[0] if ticks else 0.0
        boundaries = [single - 0.5, single + 0.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_kwargs: Dict[str, Any] = {
        "ticks": ticks,
        "boundaries": boundaries,
        "orientation": orientation,
    }
    if use_parent_axis:
        colorbar_kwargs["ax"] = axis
    else:
        axis.set_axis_off()
        target_axis: "Axes" = axis
        if height_fraction is not None and 0.0 < height_fraction < 1.0:
            inset_height = height_fraction
            inset_y = (1.0 - inset_height) / 2.0
            target_axis = axis.inset_axes([0.0, inset_y, 1.0, inset_height])
        colorbar_kwargs["cax"] = target_axis
    colorbar = fig.colorbar(scalar, **colorbar_kwargs)
    colorbar.ax.tick_params(labelsize=font_cfg.tick)
    if label:
        params = dict(label_kwargs or {})
        if orientation == "vertical":
            colorbar.ax.set_ylabel(label, fontsize=font_cfg.label, **params)
        else:
            colorbar.ax.set_xlabel(label, fontsize=font_cfg.label, **params)
