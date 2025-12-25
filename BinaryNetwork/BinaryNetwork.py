# Created by Felix J. Schmitt on 05/30/2023.
# Class for binary networks (state is 0 or 1)

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from numba import njit

try:  # pragma: no cover - optional dependency
    import scipy.sparse as sp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sp = None


@njit(cache=True)
def _dense_batch_kernel(neurons, state, field, thresholds, weights, log_states, log_enabled, log_offset):
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
        if old_state == 0:
            new_state = 1 if potential >= thresholds[neuron] else 0
        else:
            new_state = 0
        if new_state != old_state:
            delta = new_state - old_state
            state[neuron] = new_state
            for target in range(neuron_count):
                field[target] += delta * weights[target, neuron]
        if log_enabled:
            log_states[log_offset + idx, :] = state


@njit(cache=True)
def _sparse_batch_kernel(neurons, state, field, thresholds, data, indices, indptr, log_states, log_enabled, log_offset):
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
        if old_state == 0:
            new_state = 1 if potential >= thresholds[neuron] else 0
        else:
            new_state = 0
        if new_state != old_state:
            delta = new_state - old_state
            state[neuron] = new_state
            start = indptr[neuron]
            end = indptr[neuron + 1]
            for ptr in range(start, end):
                row = indices[ptr]
                field[row] += delta * data[ptr]
        if log_enabled:
            log_states[log_offset + idx, :] = state


class NetworkElement:
    def __init__(self, reference, name="Some Network Element"):
        self.name = name
        self.reference = reference
        self.view = None

    def set_view(self, view):
        self.view = np.asarray(view, dtype=np.int64)

    def initialze(self):
        pass


class Neuron(NetworkElement):
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
    # Neuron which
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

    def initialze(self):
        raise NotImplementedError


class PairwiseBernoulliSynapse(Synapse):
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.p = float(p)
        self.j = float(j)

    def initialze(self):
        shape = (self.post.N, self.pre.N)
        block = np.zeros(shape, dtype=self.reference.weight_dtype)
        p = self.p
        iterations = 1
        while p > 1:
            p /= 2.0
            iterations += 1
        for _ in range(iterations):
            draws = (np.random.random(size=shape) < p).astype(block.dtype, copy=False)
            block += draws * self.j
        self._write_block(block)


class PoissonSynapse(Synapse):
    def __init__(self, reference, pre, post, rate=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.rate = float(rate)
        self.j = float(j)

    def initialze(self):
        shape = (self.post.N, self.pre.N)
        samples = np.random.poisson(lam=self.rate, size=shape).astype(self.reference.weight_dtype, copy=False)
        self._write_block(samples * self.j)


class FixedIndegreeSynapse(Synapse):
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.p = float(p)
        self.j = float(j)

    def initialze(self):
        block = np.zeros((self.post.N, self.pre.N), dtype=self.reference.weight_dtype)
        p = max(self.p, 0.0)
        target_count = int(round(p * self.pre.N))
        target_count = min(max(target_count, 0), self.pre.N)
        if target_count == 0:
            self._write_block(block)
            return
        for tgt in range(self.post.N):
            pres = np.random.choice(self.pre.N, size=target_count, replace=True)
            np.add.at(block[tgt], pres, self.j)
        self._write_block(block)


class AllToAllSynapse(Synapse):
    def __init__(self, reference, pre, post, j=1.0):
        super().__init__(reference, pre, post)
        self.j = float(j)

    def initialze(self):
        block = np.full((self.post.N, self.pre.N), self.j, dtype=self.reference.weight_dtype)
        self._write_block(block)


class BinaryNetwork:
    def __init__(self, name="Some Binary Network"):
        self.name = name
        self.N = 0
        self.population: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.state: np.ndarray | None = None
        self.weights_dense: np.ndarray | None = None
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
        self._sparse_rows: List[np.ndarray] = []
        self._sparse_cols: List[np.ndarray] = []
        self._sparse_data: List[np.ndarray] = []
        self._step_log_buffer: np.ndarray | None = None
        self._step_log_index = 0
        self._step_log_dummy = np.zeros((0, 0), dtype=np.int8)
        self._update_queue_enabled = False
        self._update_queue_pool = np.zeros(0, dtype=np.int64)
        self._update_queue_buffer = np.zeros(0, dtype=np.int64)
        self._update_queue_chunk = 0
        self._update_queue_position = 0

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
        N_start = 0
        for idx, population in enumerate(self.population):
            population.set_view([N_start, N_start + population.N])
            N_start += population.N
            population.initialze()
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
            matrix = sp.coo_matrix((data, (row, col)), shape=(self.N, self.N), dtype=self.weight_dtype)
            if not autapse:
                matrix.setdiag(0.0)
            matrix.sum_duplicates()
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
        if self._update_queue_enabled:
            return self._draw_from_update_queue(count)
        return np.random.choice(self.N, size=count, p=self.update_prob)

    def _update_batch_dense(self, neurons: np.ndarray, log_states, log_enabled: bool, log_offset: int):
        if neurons.size == 0:
            return
        _dense_batch_kernel(
            neurons,
            self.state,
            self.field,
            self.thresholds,
            self.weights_dense,
            log_states,
            log_enabled,
            log_offset,
        )

    def _update_batch_sparse(self, neurons: np.ndarray, log_states, log_enabled: bool, log_offset: int):
        if neurons.size == 0:
            return
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
        if self.weight_mode == "dense":
            self._update_batch_dense(neurons, log_buffer, log_enabled, log_offset)
        else:
            self._update_batch_sparse(neurons, log_buffer, log_enabled, log_offset)
        if log_enabled:
            self._step_log_index += neurons.size
        self.sim_steps += neurons.size

    def run(self, steps=1000, batch_size=1):
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

    def configure_update_queue(
        self,
        pool_indices: Sequence[int] | np.ndarray | None,
        *,
        chunk_size: int | None = None,
    ):
        if pool_indices is None:
            self._update_queue_enabled = False
            self._update_queue_pool = np.zeros(0, dtype=np.int64)
            self._update_queue_buffer = np.zeros(0, dtype=np.int64)
            self._update_queue_chunk = 0
            self._update_queue_position = 0
            return
        pool = np.asarray(pool_indices, dtype=np.int64)
        if pool.ndim != 1:
            raise ValueError("pool_indices must form a flat sequence of neuron indexes.")
        if pool.size == 0:
            raise ValueError("pool_indices must contain at least one neuron index.")
        if np.any(pool < 0) or np.any(pool >= self.N):
            raise ValueError("pool_indices must contain valid neuron indexes.")
        self._update_queue_pool = pool
        max_chunk = pool.size
        if chunk_size is None or chunk_size <= 0 or chunk_size > max_chunk:
            chunk = max_chunk
        else:
            chunk = int(chunk_size)
        self._update_queue_chunk = chunk
        self._update_queue_buffer = np.empty(chunk, dtype=np.int64)
        self._update_queue_position = chunk
        self._update_queue_enabled = True

    def _refill_update_queue(self):
        if not self._update_queue_enabled or self._update_queue_chunk <= 0:
            raise RuntimeError("Update queue is not configured.")
        selection = np.random.choice(
            self._update_queue_pool,
            size=self._update_queue_chunk,
            replace=False,
        )
        self._update_queue_buffer[:] = selection
        self._update_queue_position = 0

    def _draw_from_update_queue(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        if self._update_queue_chunk <= 0:
            raise RuntimeError("Update queue is not configured.")
        result = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            remaining = self._update_queue_chunk - self._update_queue_position
            if remaining <= 0:
                self._refill_update_queue()
                remaining = self._update_queue_chunk
            take = min(remaining, count - filled)
            start = self._update_queue_position
            end = start + take
            result[filled:filled + take] = self._update_queue_buffer[start:end]
            self._update_queue_position = end
            filled += take
        return result


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
