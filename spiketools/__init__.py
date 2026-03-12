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

The examples below reuse one synthetic spike-train dataset: `20` trials, each
`5` seconds long. Trials `0-9` share the same rate and a bursty gamma order
`0.2` for controlled Fano-factor examples, while trials `10-19` use rates near
`6 spikes/s` and more regular orders for the interval-variability examples.

```python
import numpy as np

from spiketools import gamma_spikes

np.random.seed(0)
rates = np.array([6.0] * 10 + [5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
orders = np.array([0.2] * 10 + [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0], dtype=float)
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
from spiketools import gaussian_kernel, kernel_rate, rate_integral, sliding_counts, triangular_kernel

gauss = gaussian_kernel(25.0, dt=5.0, nstd=2.0)
# Match the Gaussian cutoff at +/- 50 ms.
tri = triangular_kernel(50.0 / np.sqrt(6.0), dt=5.0)
rates, rate_time = kernel_rate(
    spiketimes,
    gauss,
    tlim=[0.0, 5000.0],
    dt=5.0,
    pool=True,
)
counts, count_time = sliding_counts(spiketimes, window=250.0, dt=5.0, tlim=[0.0, 5000.0])
integrated = rate_integral(rates[0], dt=5.0)
```

![Cookbook rate analysis](spiketools_assets/cookbook_rate_tools_matched_support.png)

Compute trial-wise variability metrics:

```python
from spiketools.variability import cv2, cv_two, ff

trial_id = 13
trial = spiketimes[:, spiketimes[1] == trial_id].copy()
trial[1] = 0
ff_trials = spiketimes[:, spiketimes[1] < 10].copy()

print(f"Trial {trial_id} uses gamma order {orders[trial_id]}; expect cv2 < 1.")
print(round(float(cv2(trial)), 3))
print(round(float(cv_two(trial, min_vals=2)), 3))
print("Trials 0-9 use rate 6 spikes/s and gamma order 0.2; expect ff > 1.")
print(round(float(ff(ff_trials, tlim=[0.0, 5000.0])), 3))
```

Example output:

```text
Trial 13 uses gamma order 3.0; expect cv2 < 1.
0.292
0.536
Trials 0-9 use rate 6 spikes/s and gamma order 0.2; expect ff > 1.
2.891
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

print(np.round(values[:5], 3))
print(np.round(window_time[:5], 1))
```

Example output:

```text
[1.276 1.404 1.427 1.102 0.955]
[ 375.  625.  875. 1125. 1375.]
```

![Time-resolved cv2 example](spiketools_assets/time_resolved_cv2_example.png)

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
