from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from analysis.episode_inference import (
    active_set_configs,
    canonicalize_active_set_result,
    select_analysis_populations,
)
from analysis.types import AnalysisInput
from analysis.utils import extract_segments


def _full_input(source_type: str = "binary") -> AnalysisInput:
    rates = np.array(
        [
            [0.8, 0.1, 0.3, 0.2],
            [0.9, 0.1, 0.3, 0.2],
            [0.8, 0.1, 0.3, 0.2],
            [0.1, 0.8, 0.2, 0.3],
            [0.1, 0.9, 0.2, 0.3],
            [0.1, 0.8, 0.2, 0.3],
        ]
    )
    return AnalysisInput(
        X_rate=rates,
        dt=0.01,
        source_type=source_type,
        cluster_names=["E1", "E2", "I1", "I2"],
        cluster_cell_types=["E", "E", "I", "I"],
    )


def test_population_selection_is_independent_of_simulator_source() -> None:
    selected = select_analysis_populations(_full_input("snn"), population_source="excitatory")
    assert selected.source_type == "snn"
    assert selected.cluster_names == ["E1", "E2"]
    assert selected.X_rate.shape == (6, 2)


def test_active_set_config_matches_long_run_options() -> None:
    options = SimpleNamespace(
        segmentation="pelt", fixed_width=1, pelt_penalty=10, changepoint_backend="skchange",
        changepoint_method="pelt", pelt_min_size=2, pelt_jump=2, pelt_refine=True,
        changepoint_bandwidth=20, changepoint_max_interval_length=200,
        changepoint_parallel_backend="None", changepoint_parallel_jobs=1, pelt_smooth_width=1,
        kmax=None, em_max_iter=30, merge_after_em=True, beta_merge=0,
        min_flicker_duration=3, flicker_max_hamming=2, merge_max_iter=10,
    )
    preprocessing, method = active_set_configs(options, 0.01)
    assert preprocessing["dt"] == 0.01
    assert method["pelt_min_size"] == 2
    assert method["merge_after_em"] is True
    assert method["min_flicker_duration"] == 3


def test_canonicalization_preserves_full_population_emissions() -> None:
    full = _full_input()
    analyzed = select_analysis_populations(full, population_source="excitatory")
    raw = SimpleNamespace(
        n_states=2,
        segments=extract_segments(np.array([0, 0, 0, 1, 1, 1]), full.dt),
        metadata={
            "status": "ok",
            "episodes": [
                {"start": 0, "stop": 3, "mask": np.array([True, False])},
                {"start": 3, "stop": 6, "mask": np.array([False, True])},
            ],
        },
    )
    canonical, episodes, _summary, inventory, emissions = canonicalize_active_set_result(
        raw, analyzed, full, condition="test", title="test", seed=1, exclude_edges=False,
        kmax=2, z_threshold=3.0, similarity=0.5, noise_floor=1e-6,
    )
    assert canonical.metadata["canonical_state_keys"] == ["E1", "E2"]
    assert [row["K"] for row in inventory] == [1, 1]
    assert {row["state_key"] for row in episodes} == {"E1", "E2"}
    assert "rate_I2" in emissions[0]
