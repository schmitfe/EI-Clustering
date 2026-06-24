"""Population-state analysis tools for binary and spiking simulations."""

from .evaluation import compare_results, evaluate_result
from .active_set import (
    active_set_em,
    active_set_em_multi_init,
    classify_clusters,
    detect_population_states,
    robust_run_scale,
    simulate_active_set_data,
)
from .io import (
    analysis_input_from_binary_trace,
    analysis_input_from_spiking_payload,
    load_analysis_input,
)
from .model_selection import run_state_count_sweep
from .pipeline import run_population_state_analysis
from .preprocessing import (
    apply_population_filter,
    bin_spikes_by_cluster,
    binarize_activity,
    compute_cluster_rates,
    make_temporal_window_features,
    smooth_rates,
    sqrt_transform_counts,
    validate_analysis_input,
    zscore_features,
)
from .types import AnalysisInput, StateInferenceResult
from .utils import (
    active_cluster_patterns_to_labels,
    compute_dwell_times,
    compute_state_means,
    compute_state_templates,
    compute_transition_matrix,
    extract_segments,
    fill_short_gaps_binary,
    labels_to_active_cluster_patterns,
    median_filter_labels,
    relabel_states_by_activity,
    remove_short_segments,
)

__all__ = [
    "AnalysisInput",
    "StateInferenceResult",
    "run_population_state_analysis",
    "active_set_em",
    "active_set_em_multi_init",
    "classify_clusters",
    "detect_population_states",
    "robust_run_scale",
    "simulate_active_set_data",
    "run_state_count_sweep",
    "load_analysis_input",
    "analysis_input_from_binary_trace",
    "analysis_input_from_spiking_payload",
    "apply_population_filter",
    "bin_spikes_by_cluster",
    "compute_cluster_rates",
    "smooth_rates",
    "sqrt_transform_counts",
    "zscore_features",
    "binarize_activity",
    "make_temporal_window_features",
    "validate_analysis_input",
    "extract_segments",
    "compute_dwell_times",
    "compute_transition_matrix",
    "compute_state_means",
    "compute_state_templates",
    "relabel_states_by_activity",
    "remove_short_segments",
    "median_filter_labels",
    "fill_short_gaps_binary",
    "active_cluster_patterns_to_labels",
    "labels_to_active_cluster_patterns",
    "evaluate_result",
    "compare_results",
]
