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

Cookbook
--------

NEST spike-recorder events to `spiketimes` and smoothed rates:

```python
import numpy as np

from spiketools.rate import gaussian_kernel, kernel_rate

events = spike_recorder.get("events")
senders = events["senders"]
times = events["times"]

sender_offset = senders.min()
spiketimes = np.vstack([
    times.astype(float),
    (senders - sender_offset).astype(float),
])

kernel = gaussian_kernel(25.0, dt=1.0)
rates, rate_time = kernel_rate(
    spiketimes,
    kernel,
    tlim=[0.0, float(times.max()) + 1.0],
    dt=1.0,
    pool=False,
)
```

Binned spike-count matrices to `spiketimes` and variability metrics:

```python
import numpy as np

from spiketools import binary_to_spiketimes
from spiketools.variability import cv_two, ff

# spike_counts.shape == (n_units, n_time_bins)
# entries are spike counts per discretized time bin
dt_ms = 1.0
time_axis = np.arange(spike_counts.shape[1], dtype=float) * dt_ms

spiketimes = binary_to_spiketimes(spike_counts, time_axis)
fano = ff(spiketimes, tlim=[0.0, time_axis[-1] + dt_ms])
local_cv2 = cv_two(spiketimes)
```

BinaryNetwork simulation output to `spiketimes`:

```python
import numpy as np

data = np.load(trace_path)
spiketimes = np.vstack([
    data["spike_times"].astype(float),
    data["spike_ids"].astype(float),
])
```

If only diff logs are available:

```python
import numpy as np

from BinaryNetwork.BinaryNetwork import BinaryNetwork

data = np.load(trace_path)
times, ids = BinaryNetwork.extract_spike_events_from_diff_logs(
    data["state_updates"],
    data["state_deltas"],
)
spiketimes = np.vstack([times.astype(float), ids.astype(float)])
```

BinaryNetwork diff logs to direct population rates:

```python
import numpy as np

from BinaryNetwork.ClusteredEI_network import ClusteredEI_network

data = np.load(trace_path)
network = ClusteredEI_network(parameter)
network.initialize()

rates = network.population_rates_from_diff_logs(
    data["initial_state"],
    data["state_updates"],
    data["state_deltas"],
    sample_interval=int(data["sample_interval"]),
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
