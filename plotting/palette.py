"""Color-palette and discrete-colorbar helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

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
    "compute_discrete_boundaries",
    "draw_listed_colorbar",
]

__pdoc__ = {
    "_cycle_palette": False,
    "_sample_cmap_colors": False,
    "_prepare_line_color_map": False,
    "_prepare_value_color_map": False,
}

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
    """Repeat a finite palette until `count` colors have been produced."""
    if count <= 0:
        return []
    if not palette:
        raise ValueError("Cannot cycle an empty palette.")
    repeats = (count + len(palette) - 1) // len(palette)
    return list(palette * repeats)[:count]


def _sample_cmap_colors(colormap: str, count: int) -> List[str]:
    """Sample `count` colors from a Matplotlib colormap."""
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
    colormap: Optional[str] = None,
    palette: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    """Build a stable color mapping for integer categories."""
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


def _prepare_value_color_map(
    values: Sequence[float],
    *,
    colormap: Optional[str] = None,
    palette: Optional[Sequence[str]] = None,
) -> Tuple[Dict[float, str], List[Tuple[float, str]]]:
    """Build a stable color mapping for numeric values."""
    mapping: Dict[float, str] = {}
    entries: List[Tuple[float, str]] = []
    ordered_values = sorted({float(value) for value in values if value is not None})
    if not ordered_values:
        return mapping, entries
    palette_source: Sequence[str] = palette if palette is not None else LINE_COLORS
    if colormap:
        colors = _sample_cmap_colors(colormap, len(ordered_values))
    else:
        colors = _cycle_palette(palette_source, len(ordered_values))
    for value, color in zip(ordered_values, colors):
        mapping[float(value)] = color
        entries.append((float(value), color))
    return mapping, entries


def compute_discrete_boundaries(values: Sequence[float]) -> List[float]:
    """Compute colorbar boundaries for a discrete set of values.

    Examples
    --------
    >>> compute_discrete_boundaries([1.0, 2.0, 4.0])
    [0.5, 1.5, 3.0, 5.0]
    """
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return []
    ordered = sorted(dict.fromkeys(numeric))
    if len(ordered) == 1:
        val = float(ordered[0])
        return [val - 0.5, val + 0.5]
    gaps = [next_val - prev_val for prev_val, next_val in zip(ordered[:-1], ordered[1:])]
    first_gap = gaps[0]
    last_gap = gaps[-1]
    boundaries = [ordered[0] - first_gap / 2.0]
    for prev_val, next_val in zip(ordered[:-1], ordered[1:]):
        midpoint = (prev_val + next_val) / 2.0
        boundaries.append(midpoint)
    boundaries.append(ordered[-1] + last_gap / 2.0)
    return boundaries


def draw_listed_colorbar(
    fig: "Figure",
    axis: "Axes",
    entries: Sequence[Tuple[float, str]],
    *,
    font_cfg: "FontCfg",
    label: str,
    orientation: str = "vertical",
    height_fraction: Optional[float] = None,
    width_fraction: Optional[float] = None,
    use_parent_axis: bool = False,
    label_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Draw a discrete listed colorbar from `(value, color)` entries.

    Examples
    --------
    ```python
    fig, (ax, cax) = plt.subplots(
        1,
        2,
        figsize=(4, 1.5),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    ax.plot([0, 1], [0, 1], color=LINE_COLORS[0])
    draw_listed_colorbar(
        fig,
        cax,
        entries=[(1.0, LINE_COLORS[0]), (2.0, LINE_COLORS[1])],
        font_cfg=FontCfg().resolve(),
        label="Focused clusters",
    )
    ```

    Expected output
    ---------------
    The second axes contains a two-level discrete colorbar with ticks at `1.0`
    and `2.0`.
    """
    if not entries:
        axis.set_axis_off()
        return None
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
        inset = [0.0, 0.0, 1.0, 1.0]
        use_inset = False
        if height_fraction is not None and 0.0 < height_fraction < 1.0:
            inset_height = height_fraction
            inset[1] = (1.0 - inset_height) / 2.0
            inset[3] = inset_height
            use_inset = True
        if width_fraction is not None and 0.0 < width_fraction < 1.0:
            inset[0] = 0.0
            inset[2] = width_fraction
            use_inset = True
        if use_inset:
            target_axis = axis.inset_axes(inset)
        colorbar_kwargs["cax"] = target_axis
    colorbar = fig.colorbar(scalar, **colorbar_kwargs)
    colorbar.ax.tick_params(labelsize=font_cfg.tick)
    if label:
        params = dict(label_kwargs or {})
        if orientation == "vertical":
            colorbar.ax.set_ylabel(label, fontsize=font_cfg.label, **params)
        else:
            colorbar.ax.set_xlabel(label, fontsize=font_cfg.label, **params)
    return colorbar
