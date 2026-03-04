from __future__ import annotations

import math
from typing import Tuple

__all__ = ["_time_axis_scale"]


def _time_axis_scale(start: float, end: float) -> Tuple[float, str]:
    """
    Determine a compact time-axis scaling factor and label.

    Values below 10^3 keep the default axis label, while larger ranges
    are scaled by the corresponding power of ten.
    """
    start_val = float(start)
    end_val = float(end)
    max_value = max(start_val, end_val)
    if max_value <= 0.0:
        return 1.0, "Time [a.u.]"
    exponent = int(math.floor(math.log10(max_value)))
    if exponent < 3:
        return 1.0, "Time [a.u.]"
    scale_value = 10 ** exponent
    label = f"Time [a.u.]/$10^{{{exponent}}}$"
    return float(scale_value), label
