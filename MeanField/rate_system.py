"""Generic mean-field root-finding and ERF utilities."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, special

from sim_config import sim_tag_from_cfg

logger = logging.getLogger(__name__)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:  # pragma: no cover - optional dependency
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy import special as jspecial
    import optimistix as optx

    HAS_JAX = True
except Exception:  # pragma: no cover - optional dependency
    jnp = None
    jspecial = None
    optx = None
    HAS_JAX = False

JAX_SOLVER_CACHE: Dict[Tuple[int, float, int], Dict[str, Any]] = {}

__pdoc__ = {
    "SolverConvergenceError": False,
    "HAS_JAX": False,
    "JAX_SOLVER_CACHE": False,
    "interpolate_curve": False,
}


@dataclass
class ERFResult:
    """Container for an event-rate-function sweep.

    Examples
    --------
    >>> result = ERFResult(x_data=[0.1, 0.2], y_data=[0.12, 0.22], solves=[], completed=True)
    >>> result.completed
    True
    """
    x_data: List[float]
    y_data: List[float]
    solves: List[np.ndarray]
    completed: bool


class SolverConvergenceError(RuntimeError):
    """Raised when the Optimistix solver exhausts its iteration budget."""

    def __init__(self, v_focus: float, max_steps: int):
        message = f"Optimistix solver did not converge within {int(max_steps)} steps at v_focus={float(v_focus):.6f}."
        super().__init__(message)
        self.v_focus = float(v_focus)
        self.max_steps = int(max_steps)


class RateSystem:
    """General mean-field solver with helper utilities for ERF and fixpoints.

    Subclasses implement `_build_dynamics(...)` and then inherit the fixed-point
    solver, ERF sweep, and fixpoint analysis helpers.
    """
    # Minimum allowed variance to prevent numerical instabilities.
    # Chosen small enough to avoid affecting dynamics, only avoids divide-by-zero.
    VAR_EPS = 1e-12
    # Accept near-roots even when the underlying SciPy solver does not set success=True.
    RESIDUAL_ACCEPT_TOL = 5e-4
    # Smallest continuation step introduced when adaptively subdividing the ERF sweep.
    ADAPTIVE_CONTINUATION_MIN_STEP = 1e-3

    def __init__(
        self,
        parameter: Dict,
        v_focus: float,
        *,
        focus_population: Optional[Union[int, Sequence[int]]] = None,
        prefer_jax: bool = True,
        root_tol: float = 1e-9,
        max_function_evals: int = 4000,
        max_newton_steps: Optional[int] = 1000,
        **network_kwargs,
    ) -> None:
        self.parameter = dict(parameter)
        self.v_focus = float(v_focus)
        self.prefer_jax = bool(prefer_jax)
        self.root_tol = root_tol
        self.max_function_evals = max_function_evals
        default_steps = 256
        if max_newton_steps is None:
            self.max_steps = default_steps
        else:
            steps = int(max_newton_steps)
            if steps <= 0:
                raise ValueError("max_newton_steps must be positive.")
            self.max_steps = steps
        self.network_kwargs = dict(network_kwargs)
        dynamics = self._build_dynamics(self.parameter, **self.network_kwargs)
        if len(dynamics) == 4:
            self.A, self.B, self.bias, self.tau = dynamics
            self.C = np.zeros_like(self.B)
        elif len(dynamics) == 5:
            self.A, self.B, self.C, self.bias, self.tau = dynamics
            if self.C is None:
                self.C = np.zeros_like(self.B)
        else:
            raise ValueError("Expected _build_dynamics to return (A, B, bias, tau) or (A, B, C, bias, tau).")
        self.population_count = int(self.A.shape[0])
        if self.A.shape != self.B.shape or self.A.shape != self.C.shape:
            raise ValueError("Connectivity mean and variance matrices must match in shape.")
        if self.bias.shape[0] != self.population_count or self.tau.shape[0] != self.population_count:
            raise ValueError("Bias and tau vectors must match the matrix dimensions.")
        if focus_population is not None:
            focus_config = focus_population
        elif isinstance(self.parameter.get("focus_population"), (list, tuple, range)):
            focus_config = self.parameter.get("focus_population")
        else:
            count = int(self.parameter.get("focus_count", 1) or 1)
            focus_config = list(range(max(count, 1)))
        self.focus_indices = self._resolve_focus_indices(focus_config)
        self._initialize_group_constraints()
        self.use_jax = self.prefer_jax and HAS_JAX and optx is not None
        self._jax_args = None

    # --- abstract hooks -------------------------------------------------
    def _build_dynamics(
        self, parameter: Dict, **network_kwargs
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        raise NotImplementedError("Derived classes must implement '_build_dynamics'.")

    def _build_population_groups(self, focus: np.ndarray) -> List[np.ndarray]:
        focus_set = set(int(idx) for idx in focus.tolist())
        if not focus_set:
            focus_set = {0}
        groups: List[np.ndarray] = [np.array(sorted(focus_set), dtype=int)]
        remaining = [idx for idx in range(self.population_count) if idx not in focus_set]
        for idx in remaining:
            groups.append(np.array([idx], dtype=int))
        return groups

    # --- solver core ----------------------------------------------------
    def _initialize_group_constraints(self) -> None:
        groups = self._build_population_groups(self.focus_indices)
        covered = np.concatenate(groups)
        if covered.size != self.population_count or not np.all(np.sort(covered) == np.arange(self.population_count)):
            raise ValueError("Population groups must cover each population exactly once.")
        self.groups: List[np.ndarray] = groups
        self.group_count = len(groups)
        self.focus_group_index = 0
        self.solve_groups = [idx for idx in range(self.group_count) if idx != self.focus_group_index]
        self.dim = len(self.solve_groups)
        self.focus_population_mask = np.zeros(self.population_count, dtype=bool)
        self.focus_population_mask[self.focus_indices] = True
        self.group_membership = np.zeros((self.population_count, self.group_count), dtype=float)
        for idx, members in enumerate(self.groups):
            self.group_membership[members, idx] = 1.0
        self.group_sizes = np.array([len(members) for members in self.groups], dtype=float)
        self.group_inverse_sizes = np.reciprocal(self.group_sizes)
        self.focus_vector = np.zeros(self.group_count, dtype=float)
        self.focus_vector[self.focus_group_index] = 1.0
        self.selector_matrix = np.zeros((self.group_count, self.dim), dtype=float)
        for col, group_idx in enumerate(self.solve_groups):
            self.selector_matrix[group_idx, col] = 1.0
        self.residual_matrix = np.zeros((self.dim, self.population_count), dtype=float)
        for row, group_idx in enumerate(self.solve_groups):
            members = self.groups[group_idx]
            self.residual_matrix[row, members] = 1.0 / len(members)

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
                raise ValueError(f"Focus index {idx} out of bounds for population size {self.population_count}.")
        return np.asarray(focus, dtype=int)

    def _full_rates_numpy(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == self.population_count:
            return arr
        if arr.size == self.group_count:
            group_values = arr.copy()
            group_values[self.focus_group_index] = self.v_focus
        else:
            if arr.size != self.dim:
                raise ValueError(
                    f"Expected vector of length {self.dim}, {self.group_count}, or {self.population_count}, got {arr.size}."
                )
            group_values = self.selector_matrix @ arr
            group_values += self.focus_vector * self.v_focus
        return self.group_membership @ group_values

    def _phi_numpy(self, full_rates: np.ndarray) -> np.ndarray:
        mean = self.A.dot(full_rates) + self.bias
        rates_sq = full_rates * full_rates
        var = np.maximum(self.B.dot(full_rates) + self.C.dot(rates_sq), self.VAR_EPS)
        return 0.5 * (1 - special.erf(-mean / np.sqrt(2.0 * var)))

    def _phi_jacobian_numpy(self, full_rates: np.ndarray) -> np.ndarray:
        mean = self.A.dot(full_rates) + self.bias
        rates_sq = full_rates * full_rates
        var = np.maximum(self.B.dot(full_rates) + self.C.dot(rates_sq), self.VAR_EPS)
        inv_sqrt = 1.0 / np.sqrt(2.0 * var)
        exp_term = np.exp(-(mean ** 2) / (2.0 * var))
        coeff = (1.0 / np.sqrt(np.pi)) * exp_term * inv_sqrt
        correction = mean / (2.0 * var)
        dvar_dm = self.B + self.C * (2.0 * full_rates[None, :])
        return coeff[:, None] * (self.A - correction[:, None] * dvar_dm)

    def residual_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = self._full_rates_numpy(x)
        phi = self._phi_numpy(rates)
        residual = (phi - rates) / self.tau
        if self.dim == 0:
            return np.zeros((0,), dtype=float)
        return self.residual_matrix @ residual

    @staticmethod
    def _jax_residual(x, v_focus, args):
        A, B, C, bias, tau, focus_vector, selector, membership, reduction, var_eps = args
        group_values = selector @ x + focus_vector * v_focus
        full_rates = membership @ group_values
        mean = A @ full_rates + bias
        var = jnp.maximum(B @ full_rates + C @ (full_rates * full_rates), var_eps)
        phi = 0.5 * (1 - jspecial.erf(-mean / jnp.sqrt(2.0 * var)))
        residual = (phi - full_rates) / tau
        return reduction @ residual

    def solve(self, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Solve for a fixed point of the reduced mean-field system.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, bool]
            `(x, residual, success)` where `x` is the reduced solution vector.

        Expected output
        ---------------
        `success` is `True` when the nonlinear solver converged and `residual`
        is close to zero.
        """
        initial = self._coerce_initial(initial_guess)
        if self.dim == 0:
            residual = self.residual_numpy(initial)
            return initial, residual, True
        if self.use_jax:
            try:
                return self._solve_with_optimistix(initial)
            except SolverConvergenceError:
                raise
            except (ValueError, RuntimeError, TypeError, AttributeError) as e:
                logger.warning(
                    "Optimistix solver failed with %s: %s. Falling back to scipy solver.",
                    type(e).__name__,
                    str(e),
                )
                self.use_jax = False
        return self._solve_with_scipy(initial)

    @staticmethod
    def _residual_norm(residual: np.ndarray) -> float:
        arr = np.asarray(residual, dtype=float).ravel()
        if arr.size == 0:
            return 0.0
        return float(np.linalg.norm(arr))

    def _normalize_solver_result(
        self,
        x: np.ndarray,
        residual: np.ndarray,
        success: bool,
        *,
        method: str,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        x_arr = np.asarray(x, dtype=float).ravel()
        residual_arr = np.asarray(residual, dtype=float).ravel()
        finite = np.isfinite(x_arr).all() and np.isfinite(residual_arr).all()
        accepted = bool(success) and finite
        if (not accepted) and finite:
            residual_norm = self._residual_norm(residual_arr)
            if residual_norm <= self.RESIDUAL_ACCEPT_TOL:
                logger.debug(
                    "Accepting near-root from %s at v_focus %.6f with residual norm %.3e.",
                    method,
                    float(self.v_focus),
                    residual_norm,
                )
                accepted = True
        return x_arr, residual_arr, accepted

    def _solve_with_scipy(self, initial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        best_x = np.asarray(initial, dtype=float).ravel()
        best_residual = np.asarray(self.residual_numpy(best_x), dtype=float).ravel()
        best_norm = self._residual_norm(best_residual)

        def remember(x: np.ndarray, residual: np.ndarray) -> None:
            nonlocal best_x, best_residual, best_norm
            if not (np.isfinite(x).all() and np.isfinite(residual).all()):
                return
            norm = self._residual_norm(residual)
            if norm < best_norm:
                best_x = np.asarray(x, dtype=float).ravel()
                best_residual = np.asarray(residual, dtype=float).ravel()
                best_norm = norm

        hybr = optimize.root(
            self.residual_numpy,
            initial,
            method="hybr",
            tol=self.root_tol,
            options={"maxfev": self.max_function_evals},
        )
        x_hybr, residual_hybr, success_hybr = self._normalize_solver_result(
            hybr.x,
            hybr.fun,
            bool(hybr.success),
            method="scipy.root(hybr)",
        )
        if success_hybr:
            return x_hybr, residual_hybr, True
        remember(x_hybr, residual_hybr)

        lm_initial = x_hybr if np.isfinite(x_hybr).all() else np.asarray(initial, dtype=float).ravel()
        lm = optimize.root(
            self.residual_numpy,
            lm_initial,
            method="lm",
            tol=self.root_tol,
            options={"maxiter": self.max_function_evals},
        )
        x_lm, residual_lm, success_lm = self._normalize_solver_result(
            lm.x,
            lm.fun,
            bool(lm.success),
            method="scipy.root(lm)",
        )
        if success_lm:
            return x_lm, residual_lm, True
        remember(x_lm, residual_lm)

        ls_initial = x_lm if np.isfinite(x_lm).all() else lm_initial
        ls = optimize.least_squares(
            self.residual_numpy,
            ls_initial,
            method="trf",
            xtol=self.root_tol,
            ftol=self.root_tol,
            gtol=self.root_tol,
            max_nfev=self.max_function_evals,
        )
        x_ls, residual_ls, success_ls = self._normalize_solver_result(
            ls.x,
            ls.fun,
            bool(ls.success),
            method="scipy.least_squares(trf)",
        )
        if success_ls:
            return x_ls, residual_ls, True
        remember(x_ls, residual_ls)

        return best_x, best_residual, False

    def _solve_with_optimistix(self, initial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        args = self._prepare_jax_args()
        solver_entry = self._get_jax_solver()
        if solver_entry is None:
            raise RuntimeError("Optimistix solver is unavailable.")
        x0 = jnp.asarray(initial, dtype=jnp.float64)
        v_focus = jnp.asarray(float(self.v_focus), dtype=jnp.float64)
        value, status, _ = solver_entry["single"](x0, v_focus, args)
        success = bool(np.asarray(status, dtype=bool))
        if not success:
            raise SolverConvergenceError(self.v_focus, self.max_steps)
        value_np = np.asarray(value, dtype=float)
        return value_np, self.residual_numpy(value_np), success

    def _prepare_jax_args(self):
        if self._jax_args is None:
            self._jax_args = (
                jnp.asarray(self.A, dtype=jnp.float64),
                jnp.asarray(self.B, dtype=jnp.float64),
                jnp.asarray(self.C, dtype=jnp.float64),
                jnp.asarray(self.bias, dtype=jnp.float64),
                jnp.asarray(self.tau, dtype=jnp.float64),
                jnp.asarray(self.focus_vector, dtype=jnp.float64),
                jnp.asarray(self.selector_matrix, dtype=jnp.float64),
                jnp.asarray(self.group_membership, dtype=jnp.float64),
                jnp.asarray(self.residual_matrix, dtype=jnp.float64),
                float(self.VAR_EPS),
            )
        return self._jax_args

    def _get_jax_solver(self):
        if not (self.use_jax and HAS_JAX and optx is not None):
            return None
        if self.dim == 0:
            return None
        key = (self.dim, float(self.root_tol), int(self.max_steps))
        entry = JAX_SOLVER_CACHE.get(key)
        if entry is None:
            entry = _build_jax_solver_entry(self.dim, float(self.root_tol), int(self.max_steps))
            JAX_SOLVER_CACHE[key] = entry
        return entry

    def phi_numpy(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the transfer function on the full-rate state implied by `x`.

        Expected output
        ---------------
        Returns one activity value per population in the interval `[0, 1]`.
        """
        rates = self._full_rates_numpy(np.asarray(x, dtype=float))
        return self._phi_numpy(rates)

    def full_rates_numpy(self, x: np.ndarray) -> np.ndarray:
        """Expand a reduced solver vector into one rate per population.

        Expected output
        ---------------
        The returned array has length `population_count`.
        """
        return self._full_rates_numpy(np.asarray(x, dtype=float))

    def jacobian_numpy(self, x: np.ndarray) -> np.ndarray:
        """Return the Jacobian of the full mean-field residual at `x`.

        Expected output
        ---------------
        The returned matrix has shape `(population_count, population_count)`.
        """
        rates = self._full_rates_numpy(np.asarray(x, dtype=float))
        return (self._phi_jacobian_numpy(rates) - np.eye(self.population_count)) / self.tau[:, np.newaxis]

    def focus_output(self, rates: np.ndarray) -> float:
        """Average the rates over the configured focus populations."""
        values = np.asarray(rates, dtype=float)[self.focus_indices]
        return float(values.mean()) if values.size else float(rates[0])

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
            return np.full((self.dim,), 0.1, dtype=float)
        arr = np.asarray(initial_guess, dtype=float).ravel()
        if arr.size == self.dim:
            return arr
        if arr.size == self.group_count:
            copy = arr.copy()
            copy[self.focus_group_index] = self.v_focus
            return copy[self.solve_groups]
        if arr.size == self.population_count:
            return self._reduce_full_rates(arr)
        non_focus_count = self.population_count - len(self.focus_indices)
        if arr.size == non_focus_count:
            full_rates = np.empty(self.population_count, dtype=float)
            full_rates[self.focus_population_mask] = self.v_focus
            full_rates[~self.focus_population_mask] = arr
            return self._reduce_full_rates(full_rates)
        raise ValueError(
            f"Initial guess must have length {self.dim}, {self.group_count}, {self.population_count}, or {non_focus_count}, got {arr.size}."
        )

    def solve_sequence(
        self,
        v_focus_values: Sequence[float],
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solve a sequence of focus inputs with the accelerated JAX path.

        Expected output
        ---------------
        Returns `(solutions, success_flags)` when the JAX solver is available,
        otherwise `None`.
        """
        values = np.asarray(list(v_focus_values), dtype=float)
        if values.size == 0:
            return np.zeros((0, self.dim)), np.zeros((0,), dtype=bool)
        if self.dim == 0:
            zeros = np.zeros((values.size, 0), dtype=float)
            return zeros, np.ones((values.size,), dtype=bool)
        if not (self.use_jax and HAS_JAX and optx is not None):
            return None
        solver_entry = self._get_jax_solver()
        if solver_entry is None:
            return None
        args = self._prepare_jax_args()
        initial = self._coerce_initial(initial_guess)
        x0 = jnp.asarray(initial, dtype=jnp.float64)
        v_seq = jnp.asarray(values, dtype=jnp.float64)
        try:
            solutions, statuses, _ = solver_entry["scan"](x0, v_seq, args)
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            logger.warning(
                "Optimistix sweep failed with %s: %s. Falling back to sequential solver.",
                type(exc).__name__,
                str(exc),
            )
            self.use_jax = False
            return None
        solution_np = np.asarray(solutions, dtype=float)
        success = np.asarray(statuses, dtype=bool)
        return solution_np, success

    # --- class helpers --------------------------------------------------
    @classmethod
    def generate_erf_curve(
        cls,
        parameter: Dict,
        *,
        start: float = 0.02,
        end: float = 1.0,
        step_number: int = 20,
        retry_step: Optional[float] = None,
        initial_guess: Optional[np.ndarray] = None,
        fallback_initials: Optional[Sequence[float]] = None,
        **network_kwargs,
    ) -> ERFResult:
        """Compute the ERF for ``parameter``.

        When ``retry_step`` is provided, the solver retries inputs separated by
        that value whenever convergence fails. If the ERF cannot be completed
        the ``completed`` flag is ``False`` and no serialization should happen.
        Parameters
        ----------
        start : float, optional
            Lower bound of the input range for the ERF (default: 0.02).
        end : float, optional
            Upper bound of the input range for the ERF (default: 1.0).
        step_number : int, optional
            Number of steps between ``start`` and ``end`` (default: 20).
        ...

        Expected output
        ---------------
        Returns an :class:`ERFResult` whose `x_data` stores the driven focus
        inputs and whose `y_data` stores the corresponding mean-field outputs.
        """
        ERF_EPS = 1e-12  # Tolerance for floating-point comparison of v_in against end
        x_data: List[float] = []
        y_data: List[float] = []
        solves: List[np.ndarray] = []
        step = (end - start) / max(step_number, 1)
        adaptive_min_step = min(abs(step), cls.ADAPTIVE_CONTINUATION_MIN_STEP) if step != 0 else cls.ADAPTIVE_CONTINUATION_MIN_STEP
        aborted = False
        fallback_values = list(fallback_initials) if fallback_initials is not None else [0.02, 0.2, 0.5, 0.8, 0.98]

        def solve_with_fallback_initials(
            system: "RateSystem",
            initial: Optional[np.ndarray],
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
            best_solution: Optional[np.ndarray] = None
            best_residual: Optional[np.ndarray] = None
            best_norm = float("inf")
            candidates: List[Optional[np.ndarray]] = [initial]
            for seed in fallback_values:
                if system.dim == 0:
                    candidate = np.zeros((0,), dtype=float)
                else:
                    candidate = np.full((system.dim,), float(seed), dtype=float)
                candidates.append(candidate)
            for candidate in candidates:
                try:
                    solution, residual, success = system.solve(candidate)
                except SolverConvergenceError as exc:
                    logger.debug("Skipping v_in %.6f: %s", float(system.v_focus), str(exc))
                    continue
                residual_arr = np.asarray(residual, dtype=float).ravel()
                if success:
                    return np.asarray(solution, dtype=float).ravel(), residual_arr, True
                if np.isfinite(residual_arr).all():
                    residual_norm = cls._residual_norm(residual_arr)
                    if residual_norm < best_norm:
                        best_solution = np.asarray(solution, dtype=float).ravel()
                        best_residual = residual_arr
                        best_norm = residual_norm
            return best_solution, best_residual, False

        vector_values: Optional[List[float]] = None
        if retry_step is None:
            vector_values = []
            if step == 0:
                vector_values.append(float(start))
            else:
                cursor = float(start)
                while cursor <= end + ERF_EPS:
                    vector_values.append(cursor)
                    cursor += step
        initial_focus = float(start)
        if vector_values:
            initial_focus = float(vector_values[0])
        system = cls(parameter, initial_focus, **network_kwargs)
        current_initial = initial_guess
        prefetched_solutions: Optional[np.ndarray] = None
        prefix_limit = 0
        if vector_values and system.use_jax:
            seq_result = system.solve_sequence(vector_values, initial_guess=current_initial)
            if seq_result is not None:
                prefetched_solutions, success_flags = seq_result
                failure = np.flatnonzero(~success_flags)
                prefix_limit = int(failure[0]) if failure.size else len(vector_values)
        if vector_values is not None:
            idx = 0
            while idx < len(vector_values):
                v_in = vector_values[idx]
                system.v_focus = float(v_in)
                use_prefetched = prefetched_solutions is not None and idx < prefix_limit
                if use_prefetched:
                    solution = np.asarray(prefetched_solutions[idx], dtype=float)
                    success = True
                else:
                    solution, residual, success = solve_with_fallback_initials(system, current_initial)
                    if not success:
                        prev_v = x_data[-1] if x_data else None
                        if prev_v is not None:
                            gap = float(v_in) - float(prev_v)
                            if gap > adaptive_min_step + ERF_EPS:
                                midpoint = float(prev_v) + 0.5 * gap
                                vector_values.insert(idx, midpoint)
                                prefetched_solutions = None
                                prefix_limit = 0
                                continue
                        aborted = True
                        break
                phi_values = system.phi_numpy(solution)
                x_data.append(system.v_focus)
                y_data.append(system.focus_output(phi_values))
                solves.append(solution)
                current_initial = solution
                idx += 1
            completed = (not aborted) and (len(x_data) == len(vector_values))
        else:
            v_in = float(start)
            next_value = v_in
            while v_in <= end + ERF_EPS:
                system.v_focus = float(v_in)
                solution, residual, success = solve_with_fallback_initials(system, current_initial)
                if not success:
                    if retry_step is not None:
                        v_in += retry_step
                        next_value = v_in
                        if solution is not None:
                            current_initial = solution
                        continue
                    aborted = True
                    break
                phi_values = system.phi_numpy(solution)
                x_data.append(system.v_focus)
                y_data.append(system.focus_output(phi_values))
                solves.append(solution)
                current_initial = solution
                if step == 0:
                    next_value = float("inf")
                    break
                v_in = system.v_focus + step
                next_value = v_in
            completed = (not aborted) and (step == 0 or next_value > end + ERF_EPS)
        return ERFResult(x_data=x_data, y_data=y_data, solves=solves, completed=completed)

    @classmethod
    def compute_fixpoints(
        cls,
        sweep_entry: Sequence,
        *,
        tol: float = 1e-3,
        interpolation_steps: int = 10_000,
        **network_kwargs,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Compute fixed points from an ERF sweep.
        Fixed points with slope larger than 1 at the intersection with the identity line are considered unstable in the 1D map approximation.

        Parameters
        ----------
        sweep_entry : sequence
            Tuple (x_data, y_data, solves, parameter) as returned by generate_erf_curve.
        tol : float, optional
            Tolerance for detecting crossings of the identity line (x = y) and for
            merging nearby crossings. Crossings where |x - y| <= tol are treated as
            fixed points, and crossings within tol of each other are merged. Default
            is 1e-3.
        interpolation_steps : int, optional
            Number of interpolation points used to refine the ERF before searching
            for crossings. Larger values increase accuracy but also cost. Default
            is 10_000.

        Expected output
        ---------------
        Returns a dictionary keyed by fixed-point location. Each value contains
        at least `stability`, `rates`, `residual_norm`, and `solver_success`.
        """
        SLOPE_STABILITY_THRESHOLD = 1.0 # |d(ERF)/dv| < 1 ⇒ stable in 1D; > 1 ⇒ unstable
        x_data, y_data, solves, parameter = sweep_entry
        x_interp, y_interp = interpolate_curve(x_data, y_data, steps=interpolation_steps)
        if x_interp.size == 0 or y_interp.size == 0:
            print("Skipping fixpoint analysis: empty ERF data.")
            return {}
        diff = x_interp - y_interp
        crossings = []
        prev_diff = diff[0]
        for idx in range(1, len(diff)):
            curr_diff = diff[idx]
            cross_val = None
            if np.abs(curr_diff) <= tol:
                cross_val = y_interp[idx]
            elif np.abs(prev_diff) <= tol:
                cross_val = y_interp[idx - 1]
            elif prev_diff * curr_diff < 0:
                weight = prev_diff / (prev_diff - curr_diff)
                cross_val = y_interp[idx - 1] + weight * (y_interp[idx] - y_interp[idx - 1])
            if cross_val is not None:
                if crossings and np.abs(cross_val - crossings[-1][0]) <= tol:
                    crossings[-1] = (float(cross_val), idx)
                else:
                    crossings.append((float(cross_val), idx))
            prev_diff = curr_diff
        fixpoints: Dict[float, Dict[str, Any]] = {}
        if not crossings:
            return fixpoints
        solves_array = [np.asarray(s, dtype=float) for s in solves]
        v_out_old = np.asarray(y_data, dtype=float)
        for cross_point, idx in crossings:
            print(f"Cross-Point: {cross_point}")
            if idx <= 0:
                slope = np.inf
            else:
                slope = (y_interp[idx] - y_interp[idx - 1]) / (x_interp[idx] - x_interp[idx - 1])
            entry: Dict[str, Any] = {
                "stability": "unstable",
                "rates": None,
                "residual_norm": float("inf"),
                "solver_success": False,
                "slope": float(slope) if np.isfinite(slope) else float("inf"),
                "included": False,
            }
            slope_unstable = not np.isfinite(slope) or slope > SLOPE_STABILITY_THRESHOLD
            if len(solves_array) == 0:
                entry["reason"] = "missing_erf_solution"
                fixpoints[cross_point] = entry
                continue
            closest_idx = int(np.argmin(np.abs(v_out_old - cross_point)))
            closest_idx = min(max(closest_idx, 0), len(solves_array) - 1)
            initial = solves_array[closest_idx]
            system = cls(parameter, cross_point, **network_kwargs)
            try:
                solve, residual, success = system.solve(initial)
            except SolverConvergenceError as exc:
                entry["solver_success"] = False
                entry["residual_norm"] = float("inf")
                entry["rates"] = None
                entry["stability"] = "unstable"
                entry["reason"] = "solver_failed"
                entry["error"] = str(exc)
                fixpoints[cross_point] = entry
                continue
            residual = np.asarray(residual, dtype=float)
            residual_norm = float(np.linalg.norm(residual)) if residual.size else 0.0
            if not np.isfinite(residual).all():
                residual_norm = float("inf")
            entry["solver_success"] = bool(success)
            entry["residual_norm"] = residual_norm
            if success and np.isfinite(residual).all():
                entry["rates"] = system.full_rates_numpy(solve)
            else:
                entry["rates"] = None
            if slope_unstable or not success or not np.isfinite(residual).all():
                if not success or not np.isfinite(residual).all():
                    print("Warning: convergence problems near cross-point")
                stability = "unstable"
            else:
                jacobian = system.jacobian_numpy(solve)
                if not np.isfinite(jacobian).all():
                    stability = "unstable"
                else:
                    try:
                        eigval = np.linalg.eigvals(jacobian)
                    except np.linalg.LinAlgError:
                        stability = "unstable"
                    else:
                        stability = "stable" if (eigval < 0).all() else "unstable"
            entry["stability"] = stability
            if "reason" not in entry:
                entry["reason"] = "slope" if slope_unstable else None
            fixpoints[cross_point] = entry
        return fixpoints


def _build_jax_solver_entry(dim: int, root_tol: float, max_steps: int):
    if not (HAS_JAX and optx is not None):
        raise RuntimeError("JAX/Optimistix are required for the accelerated solver.")
    solver = optx.Newton(rtol=root_tol, atol=root_tol)

    def residual_with_focus(x, packed):
        vf, args = packed
        return RateSystem._jax_residual(x, vf, args)

    def run_one(x0, v_focus, args):
        packed = (v_focus, args)
        solution = optx.root_find(residual_with_focus, solver, x0, args=packed, max_steps=max_steps, throw=True)
        # Keep throw=False to avoid the large Optimistix stack; failure is handled upstream.
        status = jnp.asarray(solution.result == optx.RESULTS.successful, dtype=jnp.bool_)
        err_attr = getattr(solution, "error", None)
        if err_attr is None:
            error = jnp.zeros((), dtype=jnp.float64)
        else:
            error = jnp.asarray(err_attr, dtype=jnp.float64)
        return solution.value, status, error

    single = eqx.filter_jit(run_one)
        #jax.jit(run_one))

    def scan_impl(x0, v_values, args):
        def body(carry, vf):
            value, status, error = run_one(carry, vf, args)
            return value, (value, status, error)

        _, outputs = jax.lax.scan(body, x0, v_values)
        values, statuses, errors = outputs
        return values, statuses, errors

    scan = eqx.filter_jit(scan_impl)#jax.jit(scan_impl)
    return {"single": single, "scan": scan}


def interpolate_curve(x: Sequence[float], y: Sequence[float], *, steps: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as _np
    from scipy.interpolate import interp1d as _interp1d

    x_arr = _np.asarray(x, dtype=float)
    y_arr = _np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return x_arr, y_arr
    mask = _np.isfinite(x_arr) & _np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0:
        return x_arr, x_arr
    order = _np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, unique_idx = _np.unique(x_arr, return_index=True)
    x_arr = unique_x
    y_arr = y_arr[unique_idx]
    if x_arr.size == 1:
        x_new = _np.linspace(0, 1, steps)
        y_new = _np.full_like(x_new, y_arr[0])
        return x_new, y_new
    interpolator = _interp1d(x_arr, y_arr, fill_value="extrapolate")
    x_new = _np.linspace(0, 1, steps)
    y_new = interpolator(x_new)
    return x_new, y_new


def ensure_output_folder(parameter: Dict, *, tag: Optional[str] = None) -> str:
    """Create and return the cache folder for a mean-field parameter set.

    Examples
    --------
    >>> folder = ensure_output_folder({"connection_type": "bernoulli", "R_j": 0.8, "Q": 2})
    >>> folder.startswith("data/Bernoulli/Rj00_80/")
    True
    """
    conn_name = str(parameter.get("connection_type", "bernoulli")).strip()
    conn_label = conn_name.capitalize()
    r_j = float(parameter.get("R_j", 0.0))
    rj_label = f"Rj{r_j:05.2f}".replace(".", "_")
    if tag is None:
        filtered = {k: v for k, v in parameter.items() if k != "R_Eplus"}
        tag = sim_tag_from_cfg(filtered)
    folder = os.path.join("data", conn_label, rj_label, tag)
    os.makedirs(folder, exist_ok=True)
    return folder


def serialize_erf(
    file_path: str,
    parameter: Dict,
    result: ERFResult,
    *,
    focus_count: Optional[int] = None,
) -> Optional[str]:
    """Serialize a completed ERF sweep to a pickle file.

    Expected output
    ---------------
    Returns `file_path` on success and `None` when `result.completed` is
    `False`.
    """
    if not result.completed:
        return None
    R_value = float(parameter["R_Eplus"])
    focus_value = focus_count if focus_count is not None else parameter.get("focus_count", 1)
    focus_value = 1 if focus_value is None else int(focus_value)
    key = f"{R_value:.12g}_focus{focus_value}"
    payload = {key: [result.x_data, result.y_data, result.solves, parameter]}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        import pickle

        pickle.dump(payload, file)
    return file_path


def aggregate_data(folder: str) -> str:
    """Merge individual ERF pickle files into one combined pickle.

    Expected output
    ---------------
    Returns the path to `all_data_P_Eplus.pkl` inside `folder`.
    """
    import glob
    import pickle

    list_dir = sorted(
        f for f in glob.glob(f"{folder}/*.pkl") if os.path.basename(f) != "all_data_P_Eplus.pkl"
    )
    if not list_dir:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    with open(list_dir[0], "rb") as file:
        all_files = pickle.load(file)
    for name in list_dir[1:]:
        with open(name, "rb") as file:
            data = pickle.load(file)
        all_files.update(data)
    name = "all_data_P_Eplus.pkl"
    path_sum = os.path.join(folder, name)
    with open(path_sum, "wb") as file:
        pickle.dump(all_files, file)
    return path_sum


__all__ = [
    "RateSystem",
    "ERFResult",
    "ensure_output_folder",
    "serialize_erf",
    "aggregate_data",
]
