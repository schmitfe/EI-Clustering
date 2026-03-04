from __future__ import annotations

from . import conversion, population, rate, surrogates, transforms, variability, windowing
from .conversion import (
    binary_to_spiketimes,
    cut_spiketimes,
    get_time_limits,
    spiketimes_to_binary,
    spiketimes_to_list,
)
from .population import synchrony
from .rate import (
    gaussian_kernel,
    kernel_rate,
    rate_integral,
    sliding_counts,
    triangular_kernel,
)
from .surrogates import gamma_spikes
from .transforms import resample, time_stretch, time_warp
from .variability import (
    cv2,
    cv_two,
    ff,
    lv,
    kernel_fano,
    time_resolved_cv2,
    time_resolved_cv_two,
    time_warped_cv2,
)
from .windowing import rate_warped_analysis, time_resolved, time_resolved_new

__all__ = [
    # Submodules
    "conversion",
    "population",
    "rate",
    "surrogates",
    "transforms",
    "variability",
    "windowing",
    # Conversion utilities
    "binary_to_spiketimes",
    "cut_spiketimes",
    "get_time_limits",
    "spiketimes_to_binary",
    "spiketimes_to_list",
    # Rate analysis
    "gaussian_kernel",
    "kernel_rate",
    "rate_integral",
    "sliding_counts",
    "triangular_kernel",
    # Windowed analyses
    "rate_warped_analysis",
    "time_resolved",
    "time_resolved_new",
    # Variability analysis
    "cv2",
    "cv_two",
    "ff",
    "kernel_fano",
    "lv",
    "time_resolved_cv2",
    "time_resolved_cv_two",
    "time_warped_cv2",
    # Transforms and helpers
    "resample",
    "time_stretch",
    "time_warp",
    # Population-level metrics
    "synchrony",
    # Surrogate generation
    "gamma_spikes",
]
