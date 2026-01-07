from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import matplotlib as mpl

MM_TO_INCH = 1.0 / 25.4
MAX_WIDTH_MM = 180.0
MAX_HEIGHT_MM = 215.0


def _cmyk_to_rgb_hex(c: float, m: float, y: float, k: float) -> str:
    r = 1.0 - min(1.0, c + k)
    g = 1.0 - min(1.0, m + k)
    b = 1.0 - min(1.0, y + k)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


@dataclass
class PlotConfig:
    """Reusable plotting configuration aligned with Nature figure specs."""

    figure_width: float = MAX_WIDTH_MM * MM_TO_INCH
    figure_height: float | None = None
    base_font_size: float = 8.0
    title_size: float = 8.0
    label_size: float = 8.0
    tick_size: float = 7.0
    palette: Dict[str, str] = field(
        default_factory=lambda: {
            "focus_stable": _cmyk_to_rgb_hex(0.0, 0.0, 0.0, 1.0),
            "focus_unstable": _cmyk_to_rgb_hex(0.0, 0.0, 0.0, 1.0),
            "line": _cmyk_to_rgb_hex(0.6, 0.0, 0.6, 0.2),
        }
    )
    line_colors: Sequence[str] = (
        _cmyk_to_rgb_hex(0.8, 0.1, 0.0, 0.1),
        _cmyk_to_rgb_hex(0.0, 0.6, 0.2, 0.1),
        _cmyk_to_rgb_hex(0.1, 0.2, 0.8, 0.1),
        _cmyk_to_rgb_hex(0.0, 0.4, 0.8, 0.2),
        _cmyk_to_rgb_hex(0.6, 0.0, 0.1, 0.2),
    )
    panel_label_coords: tuple[float, float] = (-0.12, 1.02)
    panel_label_align: tuple[str, str] = ("right", "bottom")
    panel_label_above_coords: tuple[float, float] = (0.0, 1.02)
    panel_label_above_align: tuple[str, str] = ("center", "bottom")

    def apply(self) -> None:
        """Apply matplotlib styling for consistent publication-ready figures."""
        max_height = MAX_HEIGHT_MM * MM_TO_INCH
        default_height = self.figure_width * 0.8
        height = min(max_height, self.figure_height or default_height)
        mpl.rcParams.update(
            {
                "figure.figsize": (self.figure_width, height),
                "font.size": self.base_font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "legend.fontsize": self.label_size,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titlepad": 4.0,
                "axes.labelpad": 2.0,
            }
        )


DEFAULT_PLOT_CONFIG = PlotConfig()
