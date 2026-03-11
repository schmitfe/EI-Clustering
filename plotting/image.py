"""Utilities for embedding bitmap or PDF-backed images into Matplotlib axes."""

__all__ = [
    "add_image_ax",
]

import os
from typing import Optional
import numpy as np
import matplotlib.image as mpimg
from .font import FontCfg

def _imread_any(path: str, *, pdf_page: int = 0, pdf_zoom: float = 2.0) -> np.ndarray:
    """
    Read an image from disk. If `path` is a PDF, render page `pdf_page` (0-based)
    using PyMuPDF at the given zoom factor (1.0 = native, 2.0 ≈ 2x).
    Returns an array suitable for matplotlib.imshow.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(path) as doc:
            if pdf_page < 0 or pdf_page >= len(doc):
                raise ValueError(f"Requested page {pdf_page} not in {path} (has {len(doc)} pages)")
            page = doc.load_page(pdf_page)
            mat = fitz.Matrix(pdf_zoom, pdf_zoom)
            pix = page.get_pixmap(matrix=mat, alpha=True)  # keep alpha for imshow
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return img
    else:
        return mpimg.imread(path)

def add_image_ax(ax, path, label=None, fc: Optional[FontCfg]=None, *, pdf_page: int = 0, pdf_zoom: float = 2.0):
    """Render an image into an axes and optionally add a panel label.

    Parameters
    ----------
    ax:
        Target Matplotlib axes.
    path:
        Image or PDF path.
    label:
        Optional panel label placed near the upper-left corner.
    fc:
        Optional `FontCfg` used for the label size.
    pdf_page, pdf_zoom:
        PDF rendering options used when `path` points to a PDF.

    Examples
    --------
    ```python
    fig, ax = plt.subplots(figsize=(3, 2))
    add_image_ax(ax, "docs/plotting_assets/spike_raster_example.png", label="A")
    ```

    Expected output
    ---------------
    The axes displays the image and, when `label` is provided, a bold panel
    label is drawn near the upper-left corner.
    """
    ax.axis('off')
    try:
        img = _imread_any(path, pdf_page=pdf_page, pdf_zoom=pdf_zoom)
        h, w = img.shape[0], img.shape[1]
        ax.imshow(img)
        ax.set_box_aspect(h / w if w else 1)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Missing image:\n{path}", ha='center', va='center', fontsize=10)
        ax.set_box_aspect(1)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading:\n{os.path.basename(path)}\n{e}", ha='center', va='center', fontsize=10)
        ax.set_box_aspect(1)
    if label:
        fs = (fc.letter if fc is not None else 12)
        ax.text(-0.05, 0.99, label, transform=ax.transAxes,
                va='top', ha='left', fontsize=fs, fontweight='bold')
    return ax
