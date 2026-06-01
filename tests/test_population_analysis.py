from __future__ import annotations

import numpy as np
try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - local smoke fallback
    class _PytestShim:
        @staticmethod
        def skip(message: str) -> None:
            raise RuntimeError(message)

    pytest = _PytestShim()

from analysis.active_set import active_set_em_multi_init, simulate_active_set_data
from analysis.methods import run_active_set_em, run_changepoint_kmeans, run_hmm, run_kmeans_filter, run_threshold_filter
from analysis.pipeline import run_population_state_analysis
from analysis.preprocessing import apply_cluster_selection, apply_population_filter, bin_spikes_by_cluster
from analysis.types import AnalysisInput, StateInferenceResult
from analysis.utils import compute_dwell_times, compute_transition_matrix
from analysis.model_selection import detect_knee, gaussian_sigma_bins_to_cutoff_hz


def _default_analysis_cfg() -> dict:
    return {
        "preprocessing": {
            "binary_threshold_mode": "percentile",
            "binary_threshold_percentile": 75,
            "smoothing_sigma_bins": 1,
            "temporal_window_bins": 0,
            "hysteresis": {"enabled": False, "low_percentile": 60, "high_percentile": 80},
        },
        "methods": {
            "threshold_filter": {
                "enabled": True,
                "threshold_mode": "auto",
                "threshold_percentile": 75,
                "min_active_bins": 1,
                "min_inactive_bins": 1,
                "min_dwell_bins": 2,
                "max_gap_bins": 1,
            },
            "kmeans_filter": {
                "enabled": True,
                "n_states": 3,
                "feature_type": "binary",
                "n_init": 10,
                "random_state": 0,
                "min_dwell_bins": 2,
            },
            "changepoint_kmeans": {
                "enabled": True,
                "algorithm": "pelt",
                "cost": "l2",
                "penalty": 3.0,
                "n_states": 3,
                "feature_type": "rates",
                "segment_features": {"mean": True, "max": False, "active_fraction": True, "duration": False},
                "random_state": 0,
                "min_dwell_bins": 2,
            },
            "active_set_em": {
                "enabled": True,
                "source": "auto",
                "transform": "auto",
                "segmentation": "fixed",
                "fixed_width": 5,
                "lambda_comb": 0.1,
                "min_separation": 0.05,
                "var_floor": 1e-4,
            },
            "hmm": {
                "enabled": True,
                "emission": "poisson",
                "n_states": 3,
                "num_iters": 5,
                "num_seeds": 1,
                "random_state": 0,
            },
        },
        "evaluation": {"compute_ground_truth_metrics": True, "boundary_tolerance_bins": [1, 2], "compute_method_agreement": True},
        "plotting": {"enabled": False},
    }


def _synthetic_binary_input() -> AnalysisInput:
    patterns = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    labels = np.repeat(np.arange(3), [12, 10, 14])
    X_binary = patterns[labels]
    return AnalysisInput(
        X_binary=X_binary,
        X_rate=X_binary.astype(float),
        dt=1.0,
        source_type="binary",
        true_labels=labels,
        true_active_clusters=X_binary.copy(),
        cluster_names=["C1", "C2", "C3"],
    )


def _synthetic_snn_input() -> AnalysisInput:
    rng = np.random.default_rng(0)
    templates = np.array(
        [
            [8.0, 1.0, 1.0],
            [1.0, 8.0, 1.0],
            [1.0, 1.0, 8.0],
        ]
    )
    labels = np.repeat(np.arange(3), [20, 18, 22])
    counts = np.vstack([rng.poisson(templates[label]) for label in labels]).astype(np.int64)
    rates = counts.astype(float)
    return AnalysisInput(
        X_counts=counts,
        X_rate=rates,
        dt=1.0,
        source_type="snn",
        true_labels=labels,
        cluster_names=["E1", "E2", "E3"],
    )


def test_binary_and_snn_inputs_validate() -> None:
    binary = _synthetic_binary_input()
    snn = _synthetic_snn_input()
    assert binary.n_timepoints == binary.true_labels.shape[0]
    assert snn.n_clusters == 3


def test_bin_spikes_by_cluster_uses_cluster_space_and_pads_empty_clusters() -> None:
    counts = bin_spikes_by_cluster(
        spike_times=np.array([0.1, 0.9, 1.2, 2.0]),
        spike_ids=np.array([0, 1, 0, 2]),
        neuron_to_cluster=[0, 0, 2],
        dt=1.0,
        t_start=0.0,
        t_stop=3.0,
        n_clusters=4,
    )
    expected = np.array(
        [
            [2, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(counts, expected)


def test_cluster_selection_excitatory_only() -> None:
    data = AnalysisInput(
        X_rate=np.arange(24, dtype=float).reshape(4, 6),
        dt=1.0,
        source_type="binary",
        cluster_names=["E1", "E2", "E3", "I1", "I2", "I3"],
        cluster_cell_types=["E", "E", "E", "I", "I", "I"],
        cluster_group_ids=[0, 1, 2, 0, 1, 2],
    )
    subset = apply_cluster_selection(data, {"enabled": True, "mode": "excitatory"})
    assert subset.X_rate.shape == (4, 3)
    assert subset.cluster_names == ["E1", "E2", "E3"]


def test_population_filter_removes_flat_populations() -> None:
    state = np.concatenate([np.zeros(20), np.ones(20)])
    data = AnalysisInput(
        X_rate=np.column_stack(
            [
                state * 2.0 + 0.1,
                state * 1.5 + 0.1,
                np.full_like(state, 0.2),
                np.full_like(state, 0.25),
            ]
        ),
        dt=1.0,
        source_type="binary",
        cluster_names=["E1", "E2", "E3", "E4"],
        cluster_cell_types=["E", "E", "E", "E"],
        cluster_group_ids=[0, 1, 2, 3],
    )
    filtered, diagnostics = apply_population_filter(
        data,
        {
            "enabled": True,
            "source": "rate",
            "network_state_method": "kmeans2",
            "smoothing_sigma_bins": 0,
            "min_mean_delta": 0.3,
            "min_effect_size": 0.5,
            "min_correlation": 0.1,
            "min_keep": 1,
        },
    )
    assert diagnostics is not None
    assert filtered.n_clusters == 2
    assert filtered.cluster_names == ["E1", "E2"]


def test_dwell_times_and_transition_matrix() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 2, 2], dtype=np.int64)
    dwell = compute_dwell_times(labels, dt=2.0)
    matrix = compute_transition_matrix(labels, n_states=3)
    assert np.allclose(dwell[0], np.array([4.0, 2.0]))
    assert np.isclose(matrix[0, 1], 0.5)
    assert np.isclose(matrix[0, 2], 0.5)


def test_knee_detection_prefers_elbow() -> None:
    knee = detect_knee([2, 3, 4, 5, 6], [10.0, 6.0, 4.0, 3.5, 3.2])
    assert knee["x"] in {3.0, 4.0}


def test_cutoff_frequency_is_monotonic() -> None:
    low = gaussian_sigma_bins_to_cutoff_hz(1.0, dt=0.1)
    high = gaussian_sigma_bins_to_cutoff_hz(4.0, dt=0.1)
    assert high < low


def test_threshold_pipeline_returns_result() -> None:
    cfg = _default_analysis_cfg()
    result = run_threshold_filter(_synthetic_binary_input(), cfg["preprocessing"], cfg["methods"]["threshold_filter"])
    assert isinstance(result, StateInferenceResult)
    assert result.labels.shape[0] == 36


def test_kmeans_pipeline_returns_result() -> None:
    cfg = _default_analysis_cfg()
    result = run_kmeans_filter(_synthetic_snn_input(), cfg["preprocessing"], cfg["methods"]["kmeans_filter"])
    assert isinstance(result, StateInferenceResult)
    assert result.transition_matrix.shape[0] >= 1


def test_changepoint_pipeline_returns_result() -> None:
    cfg = _default_analysis_cfg()
    result = run_changepoint_kmeans(_synthetic_snn_input(), cfg["preprocessing"], cfg["methods"]["changepoint_kmeans"])
    assert isinstance(result, StateInferenceResult)
    assert result.segments.shape[0] >= 1


def test_active_set_em_recovers_synthetic_masks_and_k_dependent_rates() -> None:
    X, L, true_masks = simulate_active_set_data(M=300, Q=10, noise=0.03, seed=3)
    result = active_set_em_multi_init(
        X,
        L,
        Kmax=3,
        lambda_comb=0.1,
        min_separation=0.05,
        var_floor=1e-4,
    )
    assert np.mean(result.masks == true_masks) > 0.95
    assert result.mu1[1] > result.mu1[2] > result.mu1[3]
    assert result.converged


def test_active_set_pipeline_returns_result_and_diagnostics() -> None:
    cfg = _default_analysis_cfg()
    result = run_active_set_em(_synthetic_binary_input(), cfg["preprocessing"], cfg["methods"]["active_set_em"])
    assert isinstance(result, StateInferenceResult)
    assert result.labels.shape[0] == 36
    assert result.state_templates is not None
    assert "mu1_by_K" in result.metadata
    assert "cluster_labels" in result.metadata


def test_hmm_pipeline_graceful_if_missing() -> None:
    cfg = _default_analysis_cfg()
    try:
        result = run_hmm(_synthetic_snn_input(), cfg["preprocessing"], cfg["methods"]["hmm"])
    except ModuleNotFoundError:
        pytest.skip("dynamax/jax not available in the local environment.")
    assert isinstance(result, StateInferenceResult)
    assert result.posterior_probs is None or result.posterior_probs.shape[0] == result.labels.shape[0]


def test_repository_runner_smoke() -> None:
    cfg = _default_analysis_cfg()
    output = run_population_state_analysis(_synthetic_snn_input(), cfg)
    assert "kmeans_filter" in output["results"]
    assert "changepoint_kmeans" in output["results"]
    assert "active_set_em" in output["results"]
