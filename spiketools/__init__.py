"""Utilities for converting, analyzing, and transforming spike trains.

The package uses one canonical spike-train representation throughout:

`spiketimes.shape == (2, n_spikes)`
    `spiketimes[0]` contains spike times in milliseconds.
    `spiketimes[1]` contains integer trial or unit indices.

The representation is intentionally simple so it can bridge different
simulation backends, as long as the source data already represents spike
events rather than persistent activity states:

- NEST events:
  `np.vstack([events["times"], events["senders"] - sender_offset])`
- Binned spike-count matrices:
  `binary_to_spiketimes(spike_count_matrix, time_axis_ms)`
- BinaryNetwork diff logs in this repository:
  `BinaryNetwork.extract_spike_events_from_diff_logs(...)`
- Trial-wise Python lists:
  `spiketimes_to_list(spiketimes)`

Empty trials are preserved with placeholder columns `[nan, trial_id]`. This
allows downstream trial-wise statistics to keep the original population size
even when some trials or neurons do not spike.

Persistent binary network states are a different datatype. A state matrix keeps
neurons at `1` until they switch off again, whereas `spiketimes` encodes only
event onsets. Converting state matrices to spike trains therefore requires an
explicit onset extraction step and is not what `binary_to_spiketimes(...)`
does.

Shared Example Dataset
----------------------

The examples below reuse one synthetic spike-train dataset: `10` trials, each
`5` seconds long, generated from homogeneous gamma processes with rates near
`6 spikes/s` and mild trial-to-trial differences in regularity.

```python
import numpy as np

from spiketools import gamma_spikes

rates = np.array([5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([1, 2, 2, 3, 1, 2, 3, 2, 1, 3], dtype=int)
spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)
```

![Shared spike raster](spiketools_assets/shared_example_raster.png)

Cookbook
--------

Convert the shared example to a binned raster and a per-trial list:

```python
import numpy as np

from spiketools import spiketimes_to_binary, spiketimes_to_list

binary, time = spiketimes_to_binary(spiketimes, tlim=[0.0, 5000.0], dt=50.0)
trials = spiketimes_to_list(spiketimes)
```

Estimate smoothed rates from the same spike trains:

```python
from spiketools import gaussian_kernel, kernel_rate

kernel = gaussian_kernel(25.0, dt=1.0)
rates, rate_time = kernel_rate(
    spiketimes,
    kernel,
    tlim=[0.0, 5000.0],
    dt=1.0,
    pool=False,
)
```

Compute trial-wise variability metrics:

```python
from spiketools.variability import cv_two, ff

fano = ff(spiketimes, tlim=[0.0, 5000.0])
local_cv2 = cv_two(spiketimes)
```

Run a sliding-window analysis on the same dataset:

```python
from spiketools import time_resolved
from spiketools.variability import cv2

values, window_time = time_resolved(
    spiketimes,
    window=1000.0,
    func=cv2,
    kwargs={"pool": True},
    tlim=[0.0, 5000.0],
    tstep=250.0,
)
```

Regenerating docs:

```bash
python scripts/generate_api_docs.py
```
"""

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
