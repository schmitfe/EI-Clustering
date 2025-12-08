import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, special

import connectivit

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:  # pragma: no cover - optional dependency
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy import special as jspecial
    import optimistix as optx

    HAS_JAX = True
except Exception:  # pragma: no cover - optional dependency
    jnp = None
    jspecial = None
    optx = None
    HAS_JAX = False


@dataclass
class RateSystem:
    parameter: dict
    v1_fixed: float
    kappa: Optional[float] = None
    connection_type: Optional[str] = None
    focus_population: Optional[Union[int, Sequence[int]]] = None
    prefer_jax: bool = True
    max_steps: int = 256

    def __post_init__(self):
        self.Q = int(self.parameter["Q"])
        self.population_count = 2 * self.Q
        param_copy = dict(self.parameter)
        param_kappa = param_copy.pop("kappa", None)
        focus_override = param_copy.pop("focus_population", None)
        if self.kappa is not None:
            mix = float(self.kappa)
        elif param_kappa is not None:
            mix = param_kappa
        else:
            mix = 0.0
        param_conn = param_copy.pop("connection_type", None)
        conn_kind = self.connection_type if self.connection_type is not None else param_conn or "bernoulli"
        focus_config = self.focus_population if self.focus_population is not None else focus_override
        self.focus_indices = self._resolve_focus_indices(focus_config)
        matrices = connectivit.linear_connectivity(mixing_parameter=mix, connection_type=conn_kind, **param_copy)
        self.A = matrices.A
        self.B = matrices.B
        self.bias = matrices.bias
        tau_e = self.parameter["tau_e"]
        tau_i = self.parameter["tau_i"]
        self.tau = np.ones((self.population_count,), dtype=float)
        self.tau[:self.Q] *= tau_e
        self.tau[self.Q:] *= tau_i
        self._initialize_group_constraints()
        self.use_jax = self.prefer_jax and HAS_JAX and optx is not None
        self._jax_args = None

    def _full_rates_numpy(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == self.population_count:
            return arr
        if arr.size == self.group_count:
            group_values = arr.copy()
            group_values[self.focus_group_index] = self.v1_fixed
        else:
            if arr.size != self.dim:
                raise ValueError(
                    f"Expected solution vector of length {self.dim}, {self.group_count}, or {self.population_count}, "
                    f"got {arr.size}."
                )
            group_values = self.selector_matrix @ arr
            group_values += self.focus_vector * self.v1_fixed
        return self.group_membership @ group_values

    def _phi_numpy(self, full_rates: np.ndarray) -> np.ndarray:
        mean = self.A.dot(full_rates) + self.bias
        var = np.maximum(self.B.dot(full_rates), 1e-12)
        return 0.5 * (1 - special.erf(-mean / np.sqrt(2.0 * var)))

    def _phi_jacobian_numpy(self, full_rates: np.ndarray) -> np.ndarray:
        mean = self.A.dot(full_rates) + self.bias
        var = np.maximum(self.B.dot(full_rates), 1e-12)
        inv_sqrt = 1.0 / np.sqrt(2.0 * var)
        exp_term = np.exp(-(mean ** 2) / (2.0 * var))
        coeff = (1.0 / np.sqrt(np.pi)) * exp_term * inv_sqrt
        correction = mean / (2.0 * var)
        return coeff[:, None] * (self.A - correction[:, None] * self.B)

    def residual_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = self._full_rates_numpy(x)
        phi = self._phi_numpy(rates)
        residual = (phi - rates) / self.tau
        if self.dim == 0:
            return np.zeros((0,), dtype=float)
        return self.residual_matrix @ residual

    @staticmethod
    def _jax_residual(x, args):
        A, B, bias, tau, v1, focus_vector, selector, membership, reduction = args
        group_values = selector @ x + focus_vector * v1
        full_rates = membership @ group_values
        mean = A @ full_rates + bias
        var = jnp.maximum(B @ full_rates, 1e-12)
        phi = 0.5 * (1 - jspecial.erf(-mean / jnp.sqrt(2.0 * var)))
        residual = (phi - full_rates) / tau
        return reduction @ residual

    def solve(self, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        initial = self._coerce_initial(initial_guess)
        if self.dim == 0:
            residual = self.residual_numpy(initial)
            return initial, residual, True
        if self.use_jax:
            try:
                return self._solve_with_optimistix(initial)
            except Exception:
                pass
        return self._solve_with_scipy(initial)

    def _solve_with_scipy(self, initial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        result = optimize.root(
            self.residual_numpy,
            initial,
            method="hybr",
            tol=1e-9,
            options={"maxfev": 4000},
        )
        return result.x, result.fun, bool(result.success)

    def _solve_with_optimistix(self, initial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        try:
            args = self._prepare_jax_args()
        except Exception as exc:
            self.use_jax = False
            raise exc
        x0 = jnp.asarray(initial, dtype=jnp.float64)
        solver = optx.Newton(rtol=1e-9, atol=1e-9, max_steps=self.max_steps)
        solution = optx.root_find(self._jax_residual, solver, x0, args=args)
        success = bool(solution.result == optx.RESULTS.successful)
        value = np.asarray(solution.value, dtype=float)
        return value, self.residual_numpy(value), success

    def _prepare_jax_args(self):
        if self._jax_args is None:
            self._jax_args = (
                jnp.asarray(self.A, dtype=jnp.float64),
                jnp.asarray(self.B, dtype=jnp.float64),
                jnp.asarray(self.bias, dtype=jnp.float64),
                jnp.asarray(self.tau, dtype=jnp.float64),
                float(self.v1_fixed),
                jnp.asarray(self.focus_vector, dtype=jnp.float64),
                jnp.asarray(self.selector_matrix, dtype=jnp.float64),
                jnp.asarray(self.group_membership, dtype=jnp.float64),
                jnp.asarray(self.residual_matrix, dtype=jnp.float64),
            )
        return self._jax_args

    def phi_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = self._full_rates_numpy(np.asarray(x, dtype=float))
        return self._phi_numpy(rates)

    def full_rates_numpy(self, x: np.ndarray) -> np.ndarray:
        return self._full_rates_numpy(np.asarray(x, dtype=float))

    def jacobian_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = self._full_rates_numpy(np.asarray(x, dtype=float))
        return (self._phi_jacobian_numpy(rates) - np.eye(self.population_count)) / self.tau[:, np.newaxis]

    def _initialize_group_constraints(self) -> None:
        groups = self._build_population_groups(self.focus_indices)
        if not groups:
            raise ValueError("At least one population group must be defined.")
        covered = np.concatenate(groups)
        if covered.size != self.population_count or not np.all(np.sort(covered) == np.arange(self.population_count)):
            raise ValueError("Population groups must cover each population index exactly once.")
        self.groups: List[np.ndarray] = groups
        self.group_count = len(groups)
        self.focus_group_index = 0
        self.solve_groups = [idx for idx in range(self.group_count) if idx != self.focus_group_index]
        self.dim = len(self.solve_groups)
        self.focus_population_mask = np.zeros(self.population_count, dtype=bool)
        self.focus_population_mask[self.focus_indices] = True
        self.non_focus_indices = np.where(~self.focus_population_mask)[0]
        self.group_membership = np.zeros((self.population_count, self.group_count), dtype=float)
        for idx, members in enumerate(self.groups):
            self.group_membership[members, idx] = 1.0
        self.group_sizes = np.array([len(members) for members in self.groups], dtype=float)
        self.group_inverse_sizes = 1.0 / self.group_sizes
        self.focus_vector = np.zeros(self.group_count, dtype=float)
        self.focus_vector[self.focus_group_index] = 1.0
        self.selector_matrix = np.zeros((self.group_count, self.dim), dtype=float)
        for col, group_idx in enumerate(self.solve_groups):
            self.selector_matrix[group_idx, col] = 1.0
        self.residual_matrix = np.zeros((self.dim, self.population_count), dtype=float)
        for row, group_idx in enumerate(self.solve_groups):
            members = self.groups[group_idx]
            self.residual_matrix[row, members] = 1.0 / len(members)

    def _build_population_groups(self, focus: np.ndarray) -> List[np.ndarray]:
        focus_set = set(int(idx) for idx in focus.tolist())
        if not focus_set:
            focus_set = {0}
        groups: List[np.ndarray] = [np.array(sorted(focus_set), dtype=int)]
        paired_inhib = sorted(
            idx + self.Q
            for idx in focus_set
            if idx < self.Q and (idx + self.Q) not in focus_set
        )
        paired_excit = sorted(
            idx - self.Q
            for idx in focus_set
            if idx >= self.Q and (idx - self.Q) not in focus_set
        )
        other_excit = sorted(
            idx for idx in range(self.Q) if idx not in focus_set and idx not in paired_excit
        )
        other_inhib = sorted(
            idx for idx in range(self.Q, self.population_count)
            if idx not in focus_set and idx not in paired_inhib
        )
        if other_excit:
            groups.append(np.array(other_excit, dtype=int))
        if paired_inhib:
            groups.append(np.array(paired_inhib, dtype=int))
        if other_inhib:
            groups.append(np.array(other_inhib, dtype=int))
        if paired_excit:
            groups.append(np.array(paired_excit, dtype=int))
        return groups

    def _reduce_full_rates(self, full_rates: np.ndarray) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0,), dtype=float)
        full = np.asarray(full_rates, dtype=float).reshape((self.population_count,))
        group_sums = self.group_membership.T @ full
        group_means = group_sums * self.group_inverse_sizes
        return group_means[self.solve_groups]

    def _coerce_initial(self, initial_guess: Optional[np.ndarray]) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0,), dtype=float)
        if initial_guess is None:
            return np.full((self.dim,), 0.01, dtype=float)
        arr = np.asarray(initial_guess, dtype=float).ravel()
        if arr.size == self.dim:
            return arr
        if arr.size == self.group_count:
            copy = arr.copy()
            copy[self.focus_group_index] = self.v1_fixed
            return copy[self.solve_groups]
        if arr.size == self.population_count:
            return self._reduce_full_rates(arr)
        non_focus_count = self.population_count - len(self.focus_indices)
        if arr.size == non_focus_count:
            full_rates = np.empty(self.population_count, dtype=float)
            full_rates[self.focus_population_mask] = self.v1_fixed
            full_rates[~self.focus_population_mask] = arr
            return self._reduce_full_rates(full_rates)
        raise ValueError(
            f"Initial guess must have length {self.dim}, {self.group_count}, "
            f"{self.population_count}, or {non_focus_count}, got {arr.size}."
        )

    def _resolve_focus_indices(self, focus_config) -> np.ndarray:
        if focus_config is None:
            entries = [0]
        elif isinstance(focus_config, (int, np.integer)):
            entries = [int(focus_config)]
        else:
            entries = []
            for value in focus_config:
                if isinstance(value, slice):
                    start = 0 if value.start is None else value.start
                    stop = self.population_count if value.stop is None else value.stop
                    step = 1 if value.step is None else value.step
                    entries.extend(range(start, stop, step))
                else:
                    entries.append(int(value))
        focus = sorted(set(entries))
        if not focus:
            focus = [0]
        for idx in focus:
            if idx < 0 or idx >= self.population_count:
                raise ValueError(
                    f"Focus index {idx} out of bounds for population size {self.population_count}."
                )
        return np.asarray(focus, dtype=int)


__all__ = ["RateSystem"]
