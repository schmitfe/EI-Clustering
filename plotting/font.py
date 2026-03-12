"""Shared typography and annotation helpers for Matplotlib figures."""

__all__ = [
    "FontCfg",
    "add_corner_tag",
    "add_panel_label",
    "add_panel_labels_column_left_of_ylabel",
    "style_axes",
    "style_colorbar",
    "style_legend"]

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import transforms
from typing import List


@dataclass
class FontCfg:
    """Font-size configuration shared across multi-panel figures.

    Call `resolve()` once after construction to derive unset sizes from the
    `base` and `scale` values.

    Examples
    --------
    >>> cfg = FontCfg(base=10.0, scale=1.2).resolve()
    >>> round(cfg.label, 1), round(cfg.letter, 1)
    (12.0, 13.2)
    """
    base: float = 12.0
    scale: float = 1.4
    title: float = None
    label: float = None
    tick: float = None
    legend: float = None
    panel: float = None
    labelpad: float = 6.0
    letter: float = None          # NEW: subplot letter size

    def resolve(self):
        """Fill unset size fields from the base configuration."""
        if self.title  is None: self.title  = self.base * self.scale * 1.20
        if self.label  is None: self.label  = self.base * self.scale * 1.00
        if self.tick   is None: self.tick   = self.base * self.scale * 0.95
        if self.legend is None: self.legend = self.base * self.scale * 0.95
        if self.panel  is None: self.panel  = self.base * self.scale * 0.95
        if self.letter is None: self.letter = self.base * self.scale * 1.10
        return self


def add_corner_tag(ax, text, color, fc: FontCfg, *, x=0.985, y=0.985):
    """Add a boxed annotation tag in the upper-right corner of an axes.

    Expected output
    ---------------
    The axes receives one bold text box near its upper-right corner.

    ![add_corner_tag example](plotting_assets/add_corner_tag_example.png)
    """
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fc.label, fontweight="bold", color=color,
        bbox=dict(
            facecolor="white",
            edgecolor="darkgrey",
            linewidth=0.9,
            boxstyle="square,pad=0.25"
        ),
        zorder=10,
        clip_on=False
    )

def add_panel_label(ax, text, fc: FontCfg, *, x=-0.12, y=1.03):
    """Add a bold panel label in axes coordinates.

    Expected output
    ---------------
    One bold label appears slightly above and left of the axes, matching the
    figure panel style used across this repository.

    ![add_panel_label example](plotting_assets/add_panel_label_example.png)
    """
    ax.text(x, y, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=fc.letter, fontweight="bold", clip_on=False)
def add_panel_labels_column_left_of_ylabel(
    axs: List[plt.Axes],
    texts: List[str],
    fc: FontCfg,
    *,
    pad_pts: float = 6.0,   # how far left of the y-label/axes edge (in points)
    y_axes: float = 0.99    # vertical position within each axes (axes coords)
):
    """Place panel labels (texts) in a vertical column left of the y-labels.

    Alignment is computed in figure coordinates from the left-most of:
      - the y-label bbox (if present), else
      - the axes bbox (if no y-label).

    Expected output
    ---------------
    Each axes receives one label, and all labels are aligned in a shared
    vertical column left of the y-axis labels.

    ![Shared panel-label column example](plotting_assets/add_panel_labels_column_left_of_ylabel_example.png)
    """
    if not axs:
        return
    fig = axs[0].figure

    # Ensure we have a renderer (positions are known)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Find a shared x position (in pixels) across the group
    x_candidates_px = []
    for ax in axs:
        label_text = ax.yaxis.label.get_text()
        if label_text:
            bb = ax.yaxis.label.get_window_extent(renderer=renderer)
            x_candidates_px.append(bb.x0)
        else:
            bb = ax.get_window_extent(renderer=renderer)
            x_candidates_px.append(bb.x0)
    x_target_px = min(x_candidates_px)

    # Convert pad (points) to figure fraction
    pad_in = pad_pts / 72.0
    fig_w_in = fig.get_size_inches()[0]
    pad_fig = pad_in / fig_w_in

    # Convert x from pixels to figure fraction and subtract pad
    x_target_fig = fig.transFigure.inverted().transform((x_target_px, 0))[0] - pad_fig

    # Use blended transform: x in figure coords, y in axes coords
    for ax, text in zip(axs, texts):
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        ax.text(
            x_target_fig, y_axes, text,
            transform=trans, ha="right", va="top",
            fontsize=fc.letter, fontweight="bold", clip_on=False
        )

def style_axes(ax, fc: FontCfg, *, set_xlabel=True, set_ylabel=True):
    """Apply consistent label and tick font sizes to one axes.

    Examples
    --------
    ```python
    fig, ax = plt.subplots()
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Rate")
    style_axes(ax, FontCfg().resolve())
    ```

    Expected output
    ---------------
    The x-label, y-label, and tick labels use the sizes defined by `fc`.

    ![style_axes comparison](plotting_assets/style_axes_comparison.png)
    """
    if set_xlabel and ax.xaxis.label is not None:
        ax.xaxis.label.set_size(fc.label)
        ax.xaxis.labelpad = fc.labelpad
    if set_ylabel and ax.yaxis.label is not None:
        ax.yaxis.label.set_size(fc.label)
    ax.tick_params(axis='both', labelsize=fc.tick)

def style_colorbar(cbar, fc: FontCfg, *, set_label=True):
    """Apply consistent font sizes to a Matplotlib colorbar.

    Expected output
    ---------------
    The colorbar label and tick labels use the sizes defined by `fc`.

    ![style_colorbar comparison](plotting_assets/style_colorbar_comparison.png)
    """
    if cbar is None:
        return
    ax = getattr(cbar, "ax", None)
    if ax is None:
        return

    orient = getattr(cbar, "orientation", None)
    if set_label:
        if orient == "horizontal":
            if ax.xaxis.label is not None:
                ax.xaxis.label.set_size(fc.label)
                ax.xaxis.labelpad = fc.labelpad
        else:
            if ax.yaxis.label is not None:
                ax.yaxis.label.set_size(fc.label)

    ax.tick_params(axis="both", labelsize=fc.tick)


def style_legend(ax, fc: FontCfg):
    """Apply the configured legend font size to an axes legend, if present.

    Expected output
    ---------------
    All legend labels on the axes use `fc.legend`.

    ![style_legend comparison](plotting_assets/style_legend_comparison.png)
    """
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(fc.legend)
