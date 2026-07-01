"""Episode-level statistics for single-active-cluster network states."""

from __future__ import annotations

from typing import Any

import numpy as np


FISHER_EPS = 1e-6


def fisher_summary(values: np.ndarray) -> tuple[float, float, int]:
    """Return correlation mean, Fisher-z mean, and finite sample count."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(arr, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS))
    mean_z = float(np.mean(z))
    return float(np.tanh(mean_z)), mean_z, int(arr.size)


def variance_decomposition(values: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, temporal variance, quenched variance, and their pooled total."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] == 0:
        return np.nan, np.nan, np.nan, np.nan
    neuron_means = np.nanmean(arr, axis=0)
    temporal = float(np.nanmean(np.nanvar(arr, axis=0)))
    quenched = float(np.nanvar(neuron_means))
    return float(np.nanmean(neuron_means)), temporal, quenched, temporal + quenched


def population_statistics(
    values: np.ndarray,
    membership: np.ndarray,
    n_populations: int,
) -> dict[str, np.ndarray]:
    """Resolve mean and variance components for every population."""
    means = np.full(n_populations, np.nan, dtype=np.float32)
    temporal = np.full(n_populations, np.nan, dtype=np.float32)
    quenched = np.full(n_populations, np.nan, dtype=np.float32)
    total = np.full(n_populations, np.nan, dtype=np.float32)
    for population in range(n_populations):
        mask = np.asarray(membership) == population
        if not np.any(mask):
            continue
        means[population], temporal[population], quenched[population], total[population] = (
            variance_decomposition(np.asarray(values)[:, mask])
        )
    return {
        "input_mean_population": means,
        "input_var_temporal_population": temporal,
        "input_var_quenched_population": quenched,
        "input_var_total_population": total,
    }


def _random_pairs(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_pairs: int,
    rng: np.random.Generator,
    same: bool,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if (same and first.size < 2) or (not same and (first.size == 0 or second.size == 0)):
        return np.zeros((0, 2), dtype=np.int64)
    possible = first.size * (first.size - 1) // 2 if same else first.size * second.size
    count = min(max(0, int(max_pairs)), int(possible))
    if count == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.empty((count, 2), dtype=np.int64)
    for index in range(count):
        if same:
            pairs[index] = rng.choice(first, size=2, replace=False)
        else:
            pairs[index, 0] = rng.choice(first)
            pairs[index, 1] = rng.choice(second)
    return pairs


def sample_episode_pairs(
    membership: np.ndarray,
    *,
    q: int,
    max_pairs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create fixed excitatory pair sets for Figure-4 and active-centred metrics."""
    membership = np.asarray(membership, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    per_population = max(1, int(max_pairs) // max(1, q))
    within_parts = []
    active_within = []
    active_across = []
    excitatory = np.flatnonzero(membership < q)
    for population in range(q):
        selected = np.flatnonzero(membership == population)
        pairs = _random_pairs(selected, selected, max_pairs=per_population, rng=rng, same=True)
        within_parts.append(pairs)
        active_within.append(pairs)
        other = excitatory[membership[excitatory] != population]
        active_across.append(
            _random_pairs(selected, other, max_pairs=max_pairs, rng=rng, same=False)
        )
    all_within = np.concatenate(within_parts) if within_parts else np.zeros((0, 2), dtype=np.int64)
    # Draw across-population pairs directly and reject same-population pairs.
    across: list[tuple[int, int]] = []
    attempts = 0
    limit = max(10_000, 8 * int(max_pairs))
    while len(across) < int(max_pairs) and attempts < limit and excitatory.size >= 2:
        first, second = rng.choice(excitatory, size=2, replace=False)
        if membership[first] != membership[second]:
            across.append((int(first), int(second)))
        attempts += 1
    result = {
        "pairs_all_within": all_within[: int(max_pairs)],
        "pairs_all_across": np.asarray(across, dtype=np.int64).reshape(-1, 2),
    }
    for population in range(q):
        result[f"pairs_active_{population}_within"] = active_within[population]
        result[f"pairs_active_{population}_across"] = active_across[population]
    return result


def pair_correlations(values: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute temporal Pearson correlations for fixed column pairs."""
    arr = np.asarray(values, dtype=np.float32)
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[0] < 2 or pair_arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    columns = np.unique(pair_arr)
    subset = arr[:, columns]
    means = np.mean(subset, axis=0)
    std = np.std(subset, axis=0)
    lookup = {int(column): index for index, column in enumerate(columns)}
    valid = std > 0
    standardized = np.zeros_like(subset, dtype=np.float32)
    standardized[:, valid] = (subset[:, valid] - means[valid]) / std[valid]
    output = []
    for first, second in pair_arr:
        i, j = lookup[int(first)], lookup[int(second)]
        if valid[i] and valid[j]:
            output.append(float(np.mean(standardized[:, i] * standardized[:, j])))
    return np.asarray(output, dtype=np.float32)


def episode_statistics(
    *,
    states: np.ndarray,
    fields: np.ndarray,
    rates: np.ndarray,
    membership: np.ndarray,
    active_population: int,
    q: int,
    pairs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Calculate all scalar and population-vector metrics for one episode."""
    result: dict[str, Any] = {}
    state_values = np.asarray(states, dtype=np.float32)
    field_values = np.asarray(fields, dtype=np.float32)
    rate_values = np.asarray(rates, dtype=np.float32)
    membership = np.asarray(membership, dtype=np.int64)
    for prefix, values in (("output", state_values), ("input", field_values)):
        for scope, pair_key in (
            ("all_within", "pairs_all_within"),
            ("all_across", "pairs_all_across"),
            ("active_within", f"pairs_active_{active_population}_within"),
            ("active_across", f"pairs_active_{active_population}_across"),
        ):
            corr = pair_correlations(values, pairs[pair_key])
            mean_r, mean_z, count = fisher_summary(corr)
            result[f"{prefix}_{scope}_r"] = mean_r
            result[f"{prefix}_{scope}_z"] = mean_z
            result[f"{prefix}_{scope}_n"] = count

    population_count = int(np.max(membership)) + 1 if membership.size else 0
    result.update(population_statistics(field_values, membership, population_count))
    result["activity_mean_population"] = np.nanmean(rate_values, axis=0).astype(np.float32)
    excitatory_mask = membership < q
    global_mean, global_temporal, global_quenched, global_total = variance_decomposition(
        field_values[:, excitatory_mask]
    )
    result.update(
        input_mean_global=global_mean,
        input_var_temporal_global=global_temporal,
        input_var_quenched_global=global_quenched,
        input_var_total_global=global_total,
    )
    inactive = np.asarray([idx for idx in range(q) if idx != active_population], dtype=int)
    mean_vector = result["input_mean_population"]
    temporal_vector = result["input_var_temporal_population"]
    quenched_vector = result["input_var_quenched_population"]
    total_vector = result["input_var_total_population"]
    activity_vector = result["activity_mean_population"]
    result["activity_active"] = float(activity_vector[active_population])
    for name, vector in (
        ("input_mean", mean_vector),
        ("input_var_temporal", temporal_vector),
        ("input_var_quenched", quenched_vector),
        ("input_var_total", total_vector),
    ):
        result[f"{name}_active"] = float(vector[active_population])
        result[f"{name}_inactive"] = float(np.nanmean(vector[inactive])) if inactive.size else np.nan
    return result
