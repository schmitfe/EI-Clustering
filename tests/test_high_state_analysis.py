from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from Figure4_HighState import _accepted_segments, _network_balanced
from analysis.high_state import episode_statistics, population_statistics, variance_decomposition


def test_variance_decomposition_is_temporal_plus_quenched() -> None:
    values = np.array([[1.0, 3.0], [3.0, 5.0]])
    mean, temporal, quenched, total = variance_decomposition(values)
    assert mean == 3.0
    assert temporal == 1.0
    assert quenched == 1.0
    assert total == 2.0


def test_population_variance_vectors_resolve_each_population() -> None:
    values = np.array([[1.0, 3.0, 10.0], [3.0, 5.0, 14.0]])
    stats = population_statistics(values, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(stats["input_mean_population"], [3.0, 12.0])
    np.testing.assert_allclose(stats["input_var_temporal_population"], [1.0, 4.0])
    np.testing.assert_allclose(stats["input_var_quenched_population"], [1.0, 0.0])
    np.testing.assert_allclose(stats["input_var_total_population"], [2.0, 4.0])


def test_episode_statistics_align_active_and_inactive_populations() -> None:
    signal = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    states = np.column_stack([signal, signal, 1 - signal, 1 - signal, signal, 1 - signal])
    fields = states * 2.0 + np.arange(states.shape[1], dtype=np.float32)
    rates = np.column_stack(
        [np.full(6, 0.8), np.full(6, 0.2), np.full(6, 0.3), np.full(6, 0.1)]
    )
    membership = np.array([0, 0, 1, 1, 2, 3])
    pairs = {
        "pairs_all_within": np.array([[0, 1], [2, 3]]),
        "pairs_all_across": np.array([[0, 2]]),
        "pairs_active_0_within": np.array([[0, 1]]),
        "pairs_active_0_across": np.array([[0, 2]]),
    }
    result = episode_statistics(
        states=states,
        fields=fields,
        rates=rates,
        membership=membership,
        active_population=0,
        q=2,
        pairs=pairs,
    )
    assert result["activity_active"] == np.float32(0.8)
    assert result["output_active_within_r"] > 0.99
    assert result["output_active_across_r"] < -0.99
    assert result["output_active_within_n"] == 1
    assert result["input_mean_inactive"] == result["input_mean_population"][1]


def test_episode_filter_excludes_edges_and_short_segments() -> None:
    segments = pd.DataFrame(
        {
            "K": [1, 1, 1, 1],
            "duration_bins": [50, 10, 25, 50],
            "start_bin": [0, 50, 60, 85],
            "stop_bin": [50, 60, 85, 135],
        }
    )
    selected = _accepted_segments(SimpleNamespace(segments=segments), minimum_bins=20, exclude_edges=True)
    assert selected["episode_index"].tolist() == [2]


def test_network_balancing_collapses_episodes_then_initializations() -> None:
    rows = []
    for value in [10.0, 10.0, 10.0, 10.0]:
        rows.append({"connectivity": 0.2, "kappa": 0.0, "network_index": 0, "init_index": 0, **{key: value for key in __import__("Figure4_HighState").SCALAR_METRICS}})
    rows.append({"connectivity": 0.2, "kappa": 0.0, "network_index": 1, "init_index": 0, **{key: 0.0 for key in __import__("Figure4_HighState").SCALAR_METRICS}})
    balanced = _network_balanced(pd.DataFrame(rows))
    assert balanced.shape[0] == 2
    assert sorted(balanced["activity_active"].tolist()) == [0.0, 10.0]
