# Created by Felix J. Schmitt on 05/30/2023.
# Class for binary networks (state is 0 or 1)
"""Core binary-network data structures and simulation engine.

The module provides generic neuron populations, synapse generators, and the
`BinaryNetwork` simulation class used by the repository-specific clustered E/I
wrapper.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from numba import njit

try:  # pragma: no cover - optional dependency
    import scipy.sparse as sp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sp = None

__pdoc__ = {
    "NetworkElement": False,
    "Neuron": False,
    "Synapse": False,
}

__all__ = [
    "BinaryNeuronPopulation",
    "BackgroundActivity",
    "PairwiseBernoulliSynapse",
    "PoissonSynapse",
    "FixedIndegreeSynapse",
    "AllToAllSynapse",
    "BinaryNetwork",
    "warm_numba_caches",
]

_NUMBA_WARMED = False


@njit(cache=True)
def _dense_batch_kernel(
    neurons,
    state,
    field,
    thresholds,
    weights,
    log_states,
    log_enabled,
    log_offset,
    update_log,
    delta_log,
    diff_enabled,
    diff_offset,
):
    """
    Update neurons in-place for dense weight matrices.
    Parameters
    ----------
    neurons : np.ndarray
        Indices that should be updated (int64).
    state : np.ndarray
        Binary neuron state vector (int8).
    field : np.ndarray
        Cached synaptic field h = W @ state (float32).
    thresholds : np.ndarray
        Per-neuron thresholds (float32).
    weights : np.ndarray
        Dense weight matrix (float32).
    """
    neuron_count = weights.shape[0]
    for idx in range(neurons.size):
        neuron = neurons[idx]
        old_state = state[neuron]
        potential = field[neuron]
        new_state = 1 if potential > thresholds[neuron] else 0
        delta = 0
        if new_state != old_state:
            delta = new_state - old_state
            state[neuron] = new_state
            for target in range(neuron_count):
                field[target] += delta * weights[target, neuron]
        if diff_enabled:
            update_log[diff_offset + idx] = neuron
            delta_log[diff_offset + idx] = delta
        if log_enabled:
            log_states[log_offset + idx, :] = state


@njit(cache=True)
def _sparse_batch_kernel(
    neurons,
    state,
    field,
    thresholds,
    data,
    indices,
    indptr,
    log_states,
    log_enabled,
    log_offset,
    update_log,
    delta_log,
    diff_enabled,
    diff_offset,
):
    """
    Update neurons for sparse CSC weights.
    Parameters
    ----------
    data, indices, indptr : np.ndarray
        CSC column representation (float32, int32, int64/int32).
    """
    for idx in range(neurons.size):
        neuron = neurons[idx]
        old_state = state[neuron]
        potential = field[neuron]
        new_state = 1 if potential > thresholds[neuron] else 0
        delta = 0
        if new_state != old_state:
            delta = new_state - old_state
            state[neuron] = new_state
            start = indptr[neuron]
            end = indptr[neuron + 1]
            for ptr in range(start, end):
                row = indices[ptr]
                field[row] += delta * data[ptr]
        if diff_enabled:
            update_log[diff_offset + idx] = neuron
            delta_log[diff_offset + idx] = delta
        if log_enabled:
            log_states[log_offset + idx, :] = state


@njit(cache=True)
def _reconstruct_states_kernel(initial_state, updates, deltas, stride):
    steps = updates.shape[1]
    if steps == 0:
        return np.zeros((0, initial_state.size), dtype=np.uint8)
    sample_count = ((steps - 1) // stride) + 1
    state = initial_state.copy()
    samples = np.zeros((sample_count, state.size), dtype=np.uint8)
    sample_idx = 0
    for step in range(steps):
        for row in range(updates.shape[0]):
            delta = deltas[row, step]
            if delta == 0:
                continue
            unit = updates[row, step]
            state_value = int(state[unit]) + int(delta)
            state[unit] = 1 if state_value > 0 else 0
        if step % stride == 0:
            samples[sample_idx, :] = state
            sample_idx += 1
    return samples


@njit(cache=True)
def _population_rates_kernel(cluster_sums, sizes, cluster_of, updates, deltas, stride):
    steps = updates.shape[1]
    pop_count = cluster_sums.size
    if steps == 0 or pop_count == 0:
        return np.zeros((0, pop_count), dtype=np.float32)
    sample_count = ((steps - 1) // stride) + 1
    samples = np.zeros((sample_count, pop_count), dtype=np.float32)
    running = cluster_sums.copy()
    sample_idx = 0
    for step in range(steps):
        for row in range(updates.shape[0]):
            delta = deltas[row, step]
            if delta == 0:
                continue
            unit = updates[row, step]
            cluster_idx = cluster_of[unit]
            running[cluster_idx] += int(delta)
        if step % stride == 0:
            for pop_idx in range(pop_count):
                samples[sample_idx, pop_idx] = running[pop_idx] / sizes[pop_idx]
            sample_idx += 1
    return samples


def warm_numba_caches() -> None:
    """Compile and cache the numba kernels once per process."""
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    neurons = np.array([0], dtype=np.int64)
    state = np.zeros(2, dtype=np.int8)
    field = np.zeros(2, dtype=np.float32)
    thresholds = np.zeros(2, dtype=np.float32)
    weights = np.zeros((2, 2), dtype=np.float32)
    log_states = np.zeros((1, 2), dtype=np.int8)
    update_log = np.zeros(1, dtype=np.uint16)
    delta_log = np.zeros(1, dtype=np.int8)
    _dense_batch_kernel(
        neurons,
        state.copy(),
        field.copy(),
        thresholds,
        weights,
        log_states,
        False,
        0,
        update_log,
        delta_log,
        False,
        0,
    )
    _sparse_batch_kernel(
        neurons,
        state.copy(),
        field.copy(),
        thresholds,
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        log_states,
        False,
        0,
        update_log,
        delta_log,
        False,
        0,
    )
    _reconstruct_states_kernel(
        np.zeros(2, dtype=np.int8),
        np.zeros((1, 1), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int8),
        1,
    )
    _population_rates_kernel(
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=np.float32),
        np.zeros(2, dtype=np.int32),
        np.zeros((1, 1), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int8),
        1,
    )
    _NUMBA_WARMED = True


def _compress_flat_samples(flat: np.ndarray, total_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=np.int64).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if total_pairs <= 2_000_000:
        counts = np.bincount(flat, minlength=total_pairs)
        unique = np.flatnonzero(counts)
        return unique.astype(np.int64, copy=False), counts[unique].astype(np.int64, copy=False)
    unique, counts = np.unique(flat, return_counts=True)
    return unique.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


class NetworkElement:
    """Base class for network elements that occupy a slice in the global state."""
    def __init__(self, reference, name="Some Network Element"):
        self.name = name
        self.reference = reference
        self.view = None

    def set_view(self, view):
        self.view = np.asarray(view, dtype=np.int64)

    def initialze(self):
        pass


class Neuron(NetworkElement):
    """Base class for neuron populations stored in contiguous slices."""
    def __init__(self, reference, N=1, name="Some Neuron", tau=1.0):
        super().__init__(reference, name)
        self.N = N
        self.state = None
        self.tau = tau

    def update(self):
        pass

    def set_view(self, view):
        super().set_view(view)
        self.state = self.reference.state[self.view[0]:self.view[1]]

    def initialze(self):
        self.reference.state[self.view[0]:self.view[1]] = self._initial_state()


class BinaryNeuronPopulation(Neuron):
    """Binary neuron population with threshold activation and configurable initializer.

    Examples
    --------
    >>> net = BinaryNetwork("demo")
    >>> pop = BinaryNeuronPopulation(net, N=3, threshold=0.5, initializer=0)
    >>> net.add_population(pop)
    <...BinaryNeuronPopulation object...>
    """
    def __init__(
        self,
        reference,
        N=1,
        threshold=1.0,
        name="Binary Neuron Population",
        tau=1.0,
        initializer=None,
        **kwargs,
    ):
        super().__init__(reference, N, name, tau=tau)
        self.threshold = float(threshold)
        self.initializer = initializer

    def update(self, weights=None, state=None, index=None, input_value=None, **kwargs):
        if input_value is None:
            if weights is None or state is None:
                raise ValueError("weights and state have to be provided when input_value is not set")
            input_value = np.sum(weights * state)
        return np.heaviside(input_value - self.threshold, 0)

    def _initial_state(self):
        if callable(self.initializer):
            values = np.asarray(self.initializer(self.N), dtype=np.int16)
            if values.size != self.N:
                raise ValueError("Initializer must return {} entries".format(self.N))
            return values
        if self.initializer is None:
            return np.random.choice([0, 1], size=self.N, p=[0.8, 0.2])
        arr = np.asarray(self.initializer)
        if arr.size == 1:
            return np.full(self.N, int(arr.item()), dtype=np.int16)
        if arr.size != self.N:
            raise ValueError("Initializer must define exactly {} elements".format(self.N))
        return arr.astype(np.int16)


class BackgroundActivity(Neuron):
    """Background population that provides a constant or stochastic drive."""
    def __init__(self, reference, N=1, Activity=0.5, Stochastic=False, name="Background Activity", tau=1.0):
        super().__init__(reference, N, name, tau=tau)
        self.Activity = Activity
        if Stochastic:
            self.update = self.update_stochastic
        else:
            self.update = self.update_deterministic

    def update_stochastic(self, weights=None, state=None, Index=None, **kwargs):
        return np.random.choice([0, 1], 1) * self.update_deterministic(weights, state, **kwargs)

    def update_deterministic(self, weights=None, state=None, Index=None, **kwargs):
        # if activity is a float, set all neurons to this activity
        if isinstance(self.Activity, float):
            return self.Activity
        # if activity is a function, set neurons by this function
        elif callable(self.Activity):
            return self.Activity()
        else:
            return 1.0

    def initialze(self):
        self.state = np.array([self.update() for _ in range(self.N)])


class Synapse(NetworkElement):
    """Base class for synapse objects that write weight blocks into the network."""
    def __init__(self, reference, pre, post, name="Some Synapse"):
        super().__init__(reference, name=post.name + " <- " + pre.name)
        self.pre = pre
        self.post = post

    def set_view(self, view):
        super().set_view(view)

    def _write_block(self, block: np.ndarray):
        """Add a dense block to the global weight matrix (dense or sparse backend)."""
        post_start, post_end = int(self.view[1, 0]), int(self.view[1, 1])
        pre_start, pre_end = int(self.view[0, 0]), int(self.view[0, 1])
        self.reference._write_weight_block(post_start, post_end, pre_start, pre_end, block)

    def _append_sparse_entries(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        if self.reference.weight_mode != "sparse":
            raise RuntimeError("Sparse entries can only be appended in sparse weight mode.")
        if rows.size == 0:
            return
        post_start = int(self.view[1, 0])
        pre_start = int(self.view[0, 0])
        self.reference._append_sparse_entries(
            np.asarray(rows, dtype=np.int64) + post_start,
            np.asarray(cols, dtype=np.int64) + pre_start,
            np.asarray(values, dtype=self.reference.weight_dtype),
        )

    def initialze(self):
        raise NotImplementedError


class PairwiseBernoulliSynapse(Synapse):
    """Sample independent Bernoulli connections for every pre/post pair.

    Expected output
    ---------------
    After `network.initialize(...)`, the addressed weight block contains zeros
    and `j`-valued entries sampled independently with probability `p`.
    """
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.p = float(p)
        self.j = float(j)

    def initialze(self):
        p = self.p
        iterations = 1
        while p > 1:
            p /= 2.0
            iterations += 1
        if self.reference.weight_mode == "sparse":
            total_pairs = self.post.N * self.pre.N
            if total_pairs == 0 or p <= 0.0:
                return
            data_chunks: List[np.ndarray] = []
            row_chunks: List[np.ndarray] = []
            col_chunks: List[np.ndarray] = []
            for _ in range(iterations):
                sample_count = int(np.random.binomial(total_pairs, p))
                if sample_count <= 0:
                    continue
                if sample_count >= total_pairs:
                    flat = np.arange(total_pairs, dtype=np.int64)
                else:
                    flat = np.random.choice(total_pairs, size=sample_count, replace=False).astype(np.int64, copy=False)
                rows = flat // self.pre.N
                cols = flat % self.pre.N
                row_chunks.append(rows)
                col_chunks.append(cols)
                data_chunks.append(np.full(sample_count, self.j, dtype=self.reference.weight_dtype))
            if data_chunks:
                self._append_sparse_entries(
                    np.concatenate(row_chunks),
                    np.concatenate(col_chunks),
                    np.concatenate(data_chunks),
                )
            return
        shape = (self.post.N, self.pre.N)
        block = np.zeros(shape, dtype=self.reference.weight_dtype)
        for _ in range(iterations):
            draws = (np.random.random(size=shape) < p).astype(block.dtype, copy=False)
            block += draws * self.j
        self._write_block(block)


class PoissonSynapse(Synapse):
    """Sample Poisson-distributed multi-edge counts per pre/post pair.

    Expected output
    ---------------
    The addressed weight block contains non-negative multiples of `j` with
    Poisson-distributed counts.
    """
    def __init__(self, reference, pre, post, rate=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.rate = float(rate)
        self.j = float(j)

    def initialze(self):
        if self.reference.weight_mode == "sparse":
            total_pairs = self.post.N * self.pre.N
            if total_pairs == 0 or self.rate <= 0.0:
                return
            sample_count = int(np.random.poisson(lam=self.rate * total_pairs))
            if sample_count <= 0:
                return
            flat = np.random.randint(total_pairs, size=sample_count, dtype=np.int64)
            unique, counts = _compress_flat_samples(flat, total_pairs)
            rows = unique // self.pre.N
            cols = unique % self.pre.N
            values = counts.astype(self.reference.weight_dtype, copy=False) * self.j
            self._append_sparse_entries(rows, cols, values)
            return
        shape = (self.post.N, self.pre.N)
        samples = np.random.poisson(lam=self.rate, size=shape).astype(self.reference.weight_dtype, copy=False)
        self._write_block(samples * self.j)


class FixedIndegreeSynapse(Synapse):
    """Connect each target neuron to a fixed number of randomly drawn inputs.

    Expected output
    ---------------
    Each target row receives approximately `round(p * N_pre)` incoming entries.
    When `multapses` is `True`, presynaptic partners are sampled with replacement.
    When `multapses` is `False`, partners are sampled without replacement.
    """
    def __init__(self, reference, pre, post, p=0.5, j=1.0, multapses=True):
        super().__init__(reference, pre, post)
        self.p = float(p)
        self.j = float(j)
        self.multapses = bool(multapses)

    def initialze(self):
        p = max(self.p, 0.0)
        target_count = int(round(p * self.pre.N))
        target_count = max(target_count, 0)
        if target_count == 0:
            return
        if (not self.multapses) and target_count > self.pre.N:
            raise ValueError(
                f"Fixed-indegree sampling without multapses requested indegree {target_count} "
                f"from only {self.pre.N} presynaptic neurons."
            )
        if self.reference.weight_mode == "sparse":
            rows = np.repeat(np.arange(self.post.N, dtype=np.int64), target_count)
            if self.multapses:
                cols = np.random.randint(self.pre.N, size=self.post.N * target_count, dtype=np.int64)
            else:
                col_chunks = [
                    np.random.choice(self.pre.N, size=target_count, replace=False).astype(np.int64, copy=False)
                    for _ in range(self.post.N)
                ]
                cols = np.concatenate(col_chunks) if col_chunks else np.zeros(0, dtype=np.int64)
            values = np.full(rows.size, self.j, dtype=self.reference.weight_dtype)
            self._append_sparse_entries(rows, cols, values)
            return
        block = np.zeros((self.post.N, self.pre.N), dtype=self.reference.weight_dtype)
        for tgt in range(self.post.N):
            pres = np.random.choice(self.pre.N, size=target_count, replace=self.multapses)
            np.add.at(block[tgt], pres, self.j)
        self._write_block(block)


class AllToAllSynapse(Synapse):
    """Dense all-to-all block with constant weight value.

    Expected output
    ---------------
    The addressed weight block is filled with the constant value `j`.
    """
    def __init__(self, reference, pre, post, j=1.0):
        super().__init__(reference, pre, post)
        self.j = float(j)

    def initialze(self):
        if self.reference.weight_mode == "sparse":
            total_pairs = self.post.N * self.pre.N
            if total_pairs == 0:
                return
            rows = np.repeat(np.arange(self.post.N, dtype=np.int64), self.pre.N)
            cols = np.tile(np.arange(self.pre.N, dtype=np.int64), self.post.N)
            values = np.full(total_pairs, self.j, dtype=self.reference.weight_dtype)
            self._append_sparse_entries(rows, cols, values)
            return
        block = np.full((self.post.N, self.pre.N), self.j, dtype=self.reference.weight_dtype)
        self._write_block(block)


class BinaryNetwork:
    """Binary network simulator with dense/sparse backends and diff-log tracing.

    Examples
    --------
    >>> np.random.seed(0)
    >>> net = BinaryNetwork("demo")
    >>> exc = BinaryNeuronPopulation(net, N=2, threshold=0.5, tau=5.0, initializer=[0, 1])
    >>> inh = BinaryNeuronPopulation(net, N=1, threshold=0.5, tau=5.0, initializer=[0])
    >>> net.add_population(exc)
    <...BinaryNeuronPopulation object...>
    >>> net.add_population(inh)
    <...BinaryNeuronPopulation object...>
    >>> net.add_synapse(AllToAllSynapse(net, exc, exc, j=0.6))
    >>> net.initialize(weight_mode="dense", autapse=False)
    >>> net.state.shape
    (3,)
    """
    def __init__(self, name="Some Binary Network"):
        self.name = name
        self.N = 0
        self.population: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.state: Optional[np.ndarray] = None
        self.weights_dense: Optional[np.ndarray] = None
        self.weights_csr = None
        self.weights_csc = None
        self.weights = None  # compatibility alias
        self.LUT = None  # look up table for the update function
        self.sim_steps = 0
        self.population_lookup = None
        self.neuron_lookup = None
        self.update_prob = None
        self.thresholds = None
        self.field = None
        self.weight_mode = "dense"
        self.weight_dtype = np.float32
        self._population_views = np.zeros((0, 2), dtype=np.int64)
        self._population_cdf = np.zeros(0, dtype=np.float64)
        self._sparse_rows: List[np.ndarray] = []
        self._sparse_cols: List[np.ndarray] = []
        self._sparse_data: List[np.ndarray] = []
        self._step_log_buffer: Optional[np.ndarray] = None
        self._step_log_index = 0
        self._step_log_dummy = np.zeros((0, 0), dtype=np.int8)
        self._diff_log_updates: Optional[np.ndarray] = None
        self._diff_log_deltas: Optional[np.ndarray] = None
        self._diff_log_index = 0
        self._diff_log_dummy_updates = np.zeros(0, dtype=np.uint16)
        self._diff_log_dummy_deltas = np.zeros(0, dtype=np.int8)

    def add_population(self, population: Neuron):
        self.population.append(population)
        self.N += population.N
        return population

    def add_synapse(self, synapse: Synapse):
        self.synapses.append(synapse)

    def initialize(
        self,
        autapse: bool = False,
        weight_mode: str = "auto",
        ram_budget_gb: float = 12.0,
        weight_dtype=np.float32,
    ):
        """Allocate state, sample connectivity, and prepare the cached input field.

        Examples
        --------
        >>> np.random.seed(1)
        >>> net = BinaryNetwork("init-demo")
        >>> pop = BinaryNeuronPopulation(net, N=2, threshold=0.1, initializer=[1, 0])
        >>> net.add_population(pop)
        <...BinaryNeuronPopulation object...>
        >>> net.initialize(weight_mode="dense")
        >>> net.field.shape
        (2,)
        """
        if self.N == 0:
            raise RuntimeError("Cannot initialize network without populations.")
        self.weight_dtype = np.dtype(weight_dtype)
        if self.weight_dtype not in (np.float32, np.float64):
            raise ValueError("weight_dtype must be float32 or float64.")
        self.state = np.zeros(self.N, dtype=np.int8)
        self.field = np.zeros(self.N, dtype=self.weight_dtype)
        self.update_prob = np.zeros(self.N, dtype=self.weight_dtype)
        self.population_lookup = np.zeros(self.N, dtype=np.int32)
        self.neuron_lookup = np.zeros(self.N, dtype=np.int32)
        self.thresholds = np.zeros(self.N, dtype=self.weight_dtype)
        pop_count = len(self.population)
        pop_views = np.zeros((pop_count, 2), dtype=np.int64)
        pop_update_mass = np.zeros(pop_count, dtype=np.float64)
        N_start = 0
        for idx, population in enumerate(self.population):
            population.set_view([N_start, N_start + population.N])
            N_start += population.N
            population.initialze()
            pop_views[idx, :] = population.view
            pop_update_mass[idx] = population.N / max(population.tau, 1e-9)
            self.update_prob[population.view[0]:population.view[1]] = 1.0 / max(population.tau, 1e-9)
            self.population_lookup[population.view[0]:population.view[1]] = idx
            self.neuron_lookup[population.view[0]:population.view[1]] = np.arange(
                population.N, dtype=np.int32
            )
            threshold_value = getattr(population, "threshold", 0.0)
            self.thresholds[population.view[0]:population.view[1]] = float(threshold_value)
        self.LUT = np.array([population.view for population in self.population])
        total = float(self.update_prob.sum())
        if not math.isfinite(total) or total <= 0:
            raise RuntimeError("Invalid update probabilities. Check tau values.")
        self.update_prob /= total
        pop_mass_total = float(pop_update_mass.sum())
        if not math.isfinite(pop_mass_total) or pop_mass_total <= 0:
            raise RuntimeError("Invalid population update masses. Check tau values.")
        self._population_views = pop_views
        self._population_cdf = np.cumsum(pop_update_mass / pop_mass_total)
        self.weight_mode = self._choose_weight_mode(weight_mode, ram_budget_gb)
        self.weights_dense = None
        self.weights_csr = None
        self.weights_csc = None
        self.weights = None
        self._sparse_rows.clear()
        self._sparse_cols.clear()
        self._sparse_data.clear()
        if self.weight_mode == "dense":
            self.weights_dense = np.zeros((self.N, self.N), dtype=self.weight_dtype)
        else:
            if sp is None:
                raise ModuleNotFoundError(
                    "SciPy is required for sparse weight mode. Install it via 'pip install scipy'."
                )
        for synapse in self.synapses:
            synapse.set_view(
                np.array(
                    [[synapse.pre.view[0], synapse.pre.view[1]], [synapse.post.view[0], synapse.post.view[1]]],
                    dtype=np.int64,
                )
            )
            synapse.initialze()
        if self.weight_mode == "dense":
            if not autapse:
                np.fill_diagonal(self.weights_dense, 0.0)
            self.weights = self.weights_dense
        else:
            row = np.concatenate(self._sparse_rows) if self._sparse_rows else np.zeros(0, dtype=np.int64)
            col = np.concatenate(self._sparse_cols) if self._sparse_cols else np.zeros(0, dtype=np.int64)
            data = np.concatenate(self._sparse_data) if self._sparse_data else np.zeros(0, dtype=self.weight_dtype)
            if not autapse and row.size:
                keep = row != col
                row = row[keep]
                col = col[keep]
                data = data[keep]
            matrix = sp.coo_matrix((data, (row, col)), shape=(self.N, self.N), dtype=self.weight_dtype)
            self.weights_csr = matrix.tocsr()
            self.weights_csc = matrix.tocsc()
            self.weights = self.weights_csr
        self._recompute_field()
        self.sim_steps = 0

    def _recompute_field(self):
        state_float = self.state.astype(self.weight_dtype, copy=False)
        if self.weight_mode == "dense":
            self.field = self.weights_dense @ state_float
        else:
            self.field = self.weights_csr.dot(state_float).astype(self.weight_dtype, copy=False)

    def _write_weight_block(self, row_start, row_end, col_start, col_end, block):
        block = np.asarray(block, dtype=self.weight_dtype)
        if block.shape != (row_end - row_start, col_end - col_start):
            raise ValueError("Block shape does not match target slice.")
        if self.weight_mode == "dense":
            self.weights_dense[row_start:row_end, col_start:col_end] += block
        else:
            rows = np.arange(row_start, row_end, dtype=np.int64)
            cols = np.arange(col_start, col_end, dtype=np.int64)
            row_idx = np.repeat(rows, block.shape[1])
            col_idx = np.tile(cols, block.shape[0])
            values = block.reshape(-1)
            mask = values != 0.0
            if mask.any():
                self._sparse_rows.append(row_idx[mask])
                self._sparse_cols.append(col_idx[mask])
                self._sparse_data.append(values[mask])

    def _append_sparse_entries(self, row_idx: np.ndarray, col_idx: np.ndarray, values: np.ndarray) -> None:
        if self.weight_mode != "sparse":
            raise RuntimeError("Sparse entries can only be appended in sparse weight mode.")
        rows = np.asarray(row_idx, dtype=np.int64).ravel()
        cols = np.asarray(col_idx, dtype=np.int64).ravel()
        data = np.asarray(values, dtype=self.weight_dtype).ravel()
        if rows.size != cols.size or rows.size != data.size:
            raise ValueError("Sparse row/col/data arrays must have the same length.")
        if rows.size == 0:
            return
        mask = data != 0.0
        if mask.any():
            self._sparse_rows.append(rows[mask])
            self._sparse_cols.append(cols[mask])
            self._sparse_data.append(data[mask])

    def _choose_weight_mode(self, requested: str, ram_budget_gb: float) -> str:
        requested = str(requested or "auto").lower()
        if requested in {"dense", "sparse"}:
            return requested
        if requested != "auto":
            raise ValueError("weight_mode must be 'dense', 'sparse', or 'auto'.")
        # Estimate dense memory footprint assuming float32.
        bytes_per_entry = np.dtype(self.weight_dtype).itemsize
        dense_bytes = self.N * self.N * bytes_per_entry
        dense_gb = dense_bytes / (1024 ** 3)
        safety = 0.6 * float(ram_budget_gb)
        print("Simulating network as: " + "dense" if dense_gb <= safety else "sparse")
        return "dense" if dense_gb <= safety else "sparse"

    def _select_neurons(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        if self._population_cdf.size == 0:
            return np.random.choice(self.N, size=count, p=self.update_prob)
        pop_indices = np.searchsorted(self._population_cdf, np.random.random(size=count), side="right")
        pop_indices = np.minimum(pop_indices, self._population_views.shape[0] - 1)
        counts = np.bincount(pop_indices, minlength=self._population_views.shape[0])
        neurons = np.empty(count, dtype=np.int64)
        offset = 0
        for pop_idx, pop_count in enumerate(counts):
            if pop_count == 0:
                continue
            start, end = self._population_views[pop_idx]
            neurons[offset:offset + pop_count] = np.random.randint(start, end, size=pop_count)
            offset += pop_count
        np.random.shuffle(neurons)
        return neurons

    def _update_batch_dense(self, neurons: np.ndarray, log_states, log_enabled: bool, log_offset: int):
        if neurons.size == 0:
            return
        diff_enabled = self._diff_log_updates is not None and self._diff_log_deltas is not None
        if diff_enabled:
            diff_updates = self._diff_log_updates
            diff_deltas = self._diff_log_deltas
            diff_offset = self._diff_log_index
        else:
            diff_updates = self._diff_log_dummy_updates
            diff_deltas = self._diff_log_dummy_deltas
            diff_offset = 0
        _dense_batch_kernel(
            neurons,
            self.state,
            self.field,
            self.thresholds,
            self.weights_dense,
            log_states,
            log_enabled,
            log_offset,
            diff_updates,
            diff_deltas,
            diff_enabled,
            diff_offset,
        )

    def _update_batch_sparse(self, neurons: np.ndarray, log_states, log_enabled: bool, log_offset: int):
        if neurons.size == 0:
            return
        diff_enabled = self._diff_log_updates is not None and self._diff_log_deltas is not None
        if diff_enabled:
            diff_updates = self._diff_log_updates
            diff_deltas = self._diff_log_deltas
            diff_offset = self._diff_log_index
        else:
            diff_updates = self._diff_log_dummy_updates
            diff_deltas = self._diff_log_dummy_deltas
            diff_offset = 0
        _sparse_batch_kernel(
            neurons,
            self.state,
            self.field,
            self.thresholds,
            self.weights_csc.data,
            self.weights_csc.indices,
            self.weights_csc.indptr,
            log_states,
            log_enabled,
            log_offset,
            diff_updates,
            diff_deltas,
            diff_enabled,
            diff_offset,
        )

    def update(self):
        neuron = self._select_neurons(1)
        self._process_batch(neuron)

    def _process_batch(self, neurons: Sequence[int]):
        neurons = np.asarray(neurons, dtype=np.int64)
        log_enabled = self._step_log_buffer is not None
        log_offset = self._step_log_index
        if log_enabled and self._step_log_buffer is not None:
            if log_offset + neurons.size > self._step_log_buffer.shape[0]:
                raise RuntimeError(
                    "Step logging buffer exhausted. Increase allocated steps when enabling step logging."
                )
            log_buffer = self._step_log_buffer
        else:
            log_buffer = self._step_log_dummy
        if self._diff_log_updates is not None and self._diff_log_deltas is not None:
            if self._diff_log_index + neurons.size > self._diff_log_updates.shape[0]:
                raise RuntimeError("Diff logging buffer exhausted. Increase allocated steps when enabling diff logging.")
        if self.weight_mode == "dense":
            self._update_batch_dense(neurons, log_buffer, log_enabled, log_offset)
        else:
            self._update_batch_sparse(neurons, log_buffer, log_enabled, log_offset)
        if log_enabled:
            self._step_log_index += neurons.size
        if self._diff_log_updates is not None and self._diff_log_deltas is not None:
            self._diff_log_index += neurons.size
        self.sim_steps += neurons.size

    def run(self, steps=1000, batch_size=1):
        """Advance the network for `steps` asynchronous updates.

        Examples
        --------
        >>> np.random.seed(2)
        >>> net = BinaryNetwork("run-demo")
        >>> pop = BinaryNeuronPopulation(net, N=2, threshold=0.1, initializer=[1, 0])
        >>> net.add_population(pop)
        <...BinaryNeuronPopulation object...>
        >>> net.initialize(weight_mode="dense")
        >>> net.run(steps=4, batch_size=2)
        >>> int(net.sim_steps)
        4
        """
        if self.state is None or self.update_prob is None:
            raise RuntimeError("Call initialize() before run().")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        steps_done = 0
        while steps_done < steps:
            current_batch = min(batch_size, steps - steps_done)
            neurons = self._select_neurons(current_batch)
            self._process_batch(neurons)
            steps_done += current_batch

    def enable_step_logging(self, steps: int):
        steps = int(max(0, steps))
        if steps == 0:
            self._step_log_buffer = None
            self._step_log_index = 0
            return
        self._step_log_buffer = np.zeros((steps, self.N), dtype=np.int8)
        self._step_log_index = 0

    def consume_step_log(self) -> np.ndarray:
        if self._step_log_buffer is None:
            return np.zeros((0, self.N), dtype=np.int8)
        filled = min(self._step_log_index, self._step_log_buffer.shape[0])
        data = self._step_log_buffer[:filled].copy()
        self._step_log_buffer = None
        self._step_log_index = 0
        return data

    def enable_diff_logging(self, steps: int):
        """Allocate a diff-log buffer for `steps` asynchronous updates.

        Examples
        --------
        >>> net = BinaryNetwork("log-demo")
        >>> pop = BinaryNeuronPopulation(net, N=1, threshold=0.0, initializer=[0])
        >>> net.add_population(pop)
        <...BinaryNeuronPopulation object...>
        >>> net.initialize(weight_mode="dense")
        >>> net.enable_diff_logging(steps=3)
        >>> net._diff_log_updates.shape
        (3,)
        """
        steps = int(max(0, steps))
        if steps == 0:
            self._diff_log_updates = None
            self._diff_log_deltas = None
            self._diff_log_index = 0
            return
        self._diff_log_updates = np.zeros(steps, dtype=np.uint16)
        self._diff_log_deltas = np.zeros(steps, dtype=np.int8)
        self._diff_log_index = 0

    def consume_diff_log(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the recorded diff log as `(updates, deltas)` row arrays.

        Expected output
        ---------------
        The returned arrays have shape `(1, recorded_steps)` for the current
        simulation backend.
        """
        if self._diff_log_updates is None or self._diff_log_deltas is None:
            return np.zeros((1, 0), dtype=np.uint16), np.zeros((1, 0), dtype=np.int8)
        filled = min(self._diff_log_index, self._diff_log_updates.shape[0])
        updates = self._diff_log_updates[:filled].copy()[None, :]
        deltas = self._diff_log_deltas[:filled].copy()[None, :]
        self._diff_log_updates = None
        self._diff_log_deltas = None
        self._diff_log_index = 0
        return updates, deltas

    @staticmethod
    def reconstruct_states_from_diff_logs(
        initial_state: np.ndarray,
        updates: np.ndarray,
        deltas: np.ndarray,
        *,
        sample_interval: int = 1,
    ) -> np.ndarray:
        """Reconstruct sampled full network states from diff-log traces.

        Examples
        --------
        >>> BinaryNetwork.reconstruct_states_from_diff_logs(
        ...     initial_state=np.array([0, 0], dtype=np.uint8),
        ...     updates=np.array([[0, 1, 0]], dtype=np.uint16),
        ...     deltas=np.array([[1, 1, -1]], dtype=np.int8),
        ... )
        array([[1, 0],
               [1, 1],
               [0, 1]], dtype=uint8)
        """
        update_arr = np.asarray(updates, dtype=np.int64)
        delta_arr = np.asarray(deltas, dtype=np.int8)
        if update_arr.ndim == 1:
            update_arr = update_arr[None, :]
        if delta_arr.ndim == 1:
            delta_arr = delta_arr[None, :]
        state = np.asarray(initial_state, dtype=np.int8).ravel().copy()
        if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
            return np.zeros((0, state.size), dtype=np.uint8)
        stride = max(1, int(sample_interval))
        return _reconstruct_states_kernel(state, update_arr, delta_arr, stride)

    def population_rates_from_diff_logs(
        self,
        initial_state: np.ndarray,
        updates: np.ndarray,
        deltas: np.ndarray,
        *,
        sample_interval: int = 1,
        populations: Optional[Sequence[Neuron]] = None,
    ) -> np.ndarray:
        """Compute per-population activity traces directly from diff logs.

        Examples
        --------
        >>> net = BinaryNetwork("rates-demo")
        >>> pop_a = BinaryNeuronPopulation(net, N=1, initializer=[0])
        >>> pop_b = BinaryNeuronPopulation(net, N=1, initializer=[0])
        >>> net.add_population(pop_a)
        <...BinaryNeuronPopulation object...>
        >>> net.add_population(pop_b)
        <...BinaryNeuronPopulation object...>
        >>> net.initialize(weight_mode="dense")
        >>> rates = net.population_rates_from_diff_logs(
        ...     np.array([0, 0], dtype=np.uint8),
        ...     np.array([[0, 1, 0]], dtype=np.uint16),
        ...     np.array([[1, 1, -1]], dtype=np.int8),
        ... )
        >>> rates.tolist()
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        """
        pops = list(self.population if populations is None else populations)
        if not pops:
            return np.zeros((0, 0), dtype=np.float32)
        update_arr = np.asarray(updates, dtype=np.int64)
        delta_arr = np.asarray(deltas, dtype=np.int8)
        if update_arr.ndim == 1:
            update_arr = update_arr[None, :]
        if delta_arr.ndim == 1:
            delta_arr = delta_arr[None, :]
        if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
            return np.zeros((0, len(pops)), dtype=np.float32)
        initial = np.asarray(initial_state, dtype=np.int8).ravel()
        pop_count = len(pops)
        sizes = np.zeros(pop_count, dtype=np.float32)
        cluster_of = np.empty(initial.size, dtype=np.int32)
        cluster_sums = np.zeros(pop_count, dtype=np.int64)
        for idx, pop in enumerate(pops):
            start, end = int(pop.view[0]), int(pop.view[1])
            cluster_of[start:end] = idx
            sizes[idx] = max(1, end - start)
            cluster_sums[idx] = int(initial[start:end].sum())
        stride = max(1, int(sample_interval))
        return _population_rates_kernel(cluster_sums, sizes, cluster_of, update_arr, delta_arr, stride)

    @staticmethod
    def extract_spike_events_from_diff_logs(
        updates: np.ndarray,
        deltas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract onset events `(times, neuron_ids)` from diff-log traces.

        Examples
        --------
        >>> times, ids = BinaryNetwork.extract_spike_events_from_diff_logs(
        ...     np.array([[0, 1, 0]], dtype=np.uint16),
        ...     np.array([[1, 1, -1]], dtype=np.int8),
        ... )
        >>> times.tolist()
        [0.0, 1.0]
        >>> ids.tolist()
        [0, 1]
        """
        update_arr = np.asarray(updates, dtype=np.int64)
        delta_arr = np.asarray(deltas, dtype=np.int8)
        if update_arr.ndim == 1:
            update_arr = update_arr[None, :]
        if delta_arr.ndim == 1:
            delta_arr = delta_arr[None, :]
        if update_arr.ndim != 2 or delta_arr.shape != update_arr.shape:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        mask = delta_arr > 0
        if not mask.any():
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        per_step, steps = update_arr.shape
        times = np.repeat(np.arange(steps, dtype=np.int64), per_step)
        flat_updates = update_arr.reshape(-1, order="F")
        flat_mask = mask.reshape(-1, order="F")
        return times[flat_mask].astype(np.float64, copy=False), flat_updates[flat_mask].astype(np.int64, copy=False)


def _build_demo_network(weight_mode: str) -> BinaryNetwork:
    net = BinaryNetwork(f"demo-{weight_mode}")
    pop_e = BinaryNeuronPopulation(net, N=4, threshold=0.5, tau=5.0)
    pop_i = BinaryNeuronPopulation(net, N=4, threshold=0.5, tau=5.0)
    net.add_population(pop_e)
    net.add_population(pop_i)
    net.add_synapse(PairwiseBernoulliSynapse(net, pop_e, pop_e, p=0.2, j=1.0))
    net.add_synapse(PairwiseBernoulliSynapse(net, pop_i, pop_i, p=0.2, j=1.0))
    net.add_synapse(PairwiseBernoulliSynapse(net, pop_e, pop_i, p=0.5, j=1.1))
    net.add_synapse(PairwiseBernoulliSynapse(net, pop_i, pop_e, p=0.5, j=-0.8))
    net.initialize(weight_mode=weight_mode, autapse=False, ram_budget_gb=0.1)
    return net


if __name__ == "__main__":
    np.random.seed(0)
    print("Running dense backend demo...")
    dense_net = _build_demo_network("dense")
    dense_net.run(steps=20, batch_size=4)
    print("Dense state:", dense_net.state)
    if sp is not None:
        print("Running sparse backend demo...")
        sparse_net = _build_demo_network("sparse")
        sparse_net.run(steps=20, batch_size=4)
        print("Sparse state:", sparse_net.state)
    else:
        print("SciPy not available; skipping sparse demo.")
