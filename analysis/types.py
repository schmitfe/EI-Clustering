from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd


SourceType = Literal["binary", "snn"]


@dataclass
class AnalysisInput:
    """Common analysis payload used by all state-inference methods."""

    dt: float
    source_type: SourceType
    X_counts: Optional[np.ndarray] = None
    X_binary: Optional[np.ndarray] = None
    X_rate: Optional[np.ndarray] = None
    cluster_ids: Optional[Sequence[int]] = None
    cluster_names: Optional[Sequence[str]] = None
    cluster_cell_types: Optional[Sequence[str]] = None
    cluster_group_ids: Optional[Sequence[int]] = None
    true_labels: Optional[np.ndarray] = None
    true_active_clusters: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_timepoints(self) -> int:
        for arr in (self.X_counts, self.X_binary, self.X_rate):
            if arr is not None:
                return int(arr.shape[0])
        return 0

    @property
    def n_clusters(self) -> int:
        for arr in (self.X_counts, self.X_binary, self.X_rate):
            if arr is not None:
                return int(arr.shape[1])
        return 0

    def preferred_matrix(self) -> np.ndarray:
        for arr in (self.X_rate, self.X_counts, self.X_binary):
            if arr is not None:
                return np.asarray(arr)
        raise ValueError("AnalysisInput does not contain any feature matrix.")


@dataclass
class StateInferenceResult:
    """Unified result container for all inference methods."""

    method: str
    labels: np.ndarray
    segments: pd.DataFrame
    dwell_times: Dict[int, np.ndarray]
    transition_matrix: np.ndarray
    state_means: np.ndarray
    state_templates: Optional[np.ndarray]
    state_occupancy: Dict[int, float]
    posterior_probs: Optional[np.ndarray] = None
    log_likelihood: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return int(self.state_means.shape[0]) if self.state_means.ndim == 2 else 0
