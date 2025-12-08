import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, special

from sim_config import sim_tag_from_cfg

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
class ERFResult:
    x_data: List[float]
    y_data: List[float]
    solves: List[np.ndarray]
    completed: bool


class RateSystem:
    """General mean-field solver with helper utilities for ERF and fixpoints."""

    def __init__(
        self,
        parameter: Dict,
        v_focus: float,
        *,
        focus_population: Optional[Union[int, Sequence[int]]] = None,
        prefer_jax: bool = True,
        max_steps: int = 256,
        **network_kwargs,
    ) -> None:
        self.parameter = dict(parameter)
        self.v_focus = float(v_focus)
        self.prefer_jax = bool(prefer_jax)
        self.max_steps = int(max_steps)
        self.network_kwargs = dict(network_kwargs)
        self.A, self.B, self.bias, self.tau = self._build_dynamics(self.parameter, **self.network_kwargs)
        self.population_count = int(self.A.shape[0])
        if self.A.shape != self.B.shape:
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
    def _build_dynamics(self, parameter: Dict, **network_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        A, B, bias, tau, v_focus, focus_vector, selector, membership, reduction = args
        group_values = selector @ x + focus_vector * v_focus
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
                self.use_jax = False
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
        args = self._prepare_jax_args()
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
                float(self.v_focus),
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

    def focus_output(self, rates: np.ndarray) -> float:
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
            return np.full((self.dim,), 0.01, dtype=float)
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

    # --- class helpers --------------------------------------------------
    @classmethod
    def generate_erf_curve(
        cls,
        parameter: Dict,
        *,
        start: float = 0.0,
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
        """
        x_data: List[float] = []
        y_data: List[float] = []
        solves: List[np.ndarray] = []
        v_in = float(start)
        step = (end - start) / max(step_number, 1)
        aborted = False
        current_initial = initial_guess
        fallback_values = list(fallback_initials) if fallback_initials is not None else [0.02, 0.2, 0.5, 0.8, 0.98]
        while v_in <= end + 1e-12:
            system = cls(parameter, v_in, **network_kwargs)
            solution, residual, success = system.solve(current_initial)
            if not success:
                for seed in fallback_values:
                    if system.dim == 0:
                        candidate = np.zeros((0,), dtype=float)
                    else:
                        candidate = np.full((system.dim,), float(seed), dtype=float)
                    solution, residual, success = system.solve(candidate)
                    if success:
                        break
                if retry_step is not None and not success:
                    v_in += retry_step
                    current_initial = solution
                    continue
                if not success:
                    aborted = True
                    break
            phi_values = system.phi_numpy(solution)
            x_data.append(system.v_focus)
            y_data.append(system.focus_output(phi_values))
            solves.append(solution)
            current_initial = solution
            if step == 0:
                break
            v_in = system.v_focus + step
        completed = (not aborted) and (step == 0 or v_in > end + 1e-12)
        return ERFResult(x_data=x_data, y_data=y_data, solves=solves, completed=completed)

    @classmethod
    def compute_fixpoints(
        cls,
        sweep_entry: Sequence,
        *,
        tol: float = 1e-4,
        interpolation_steps: int = 20000,
        **network_kwargs,
    ) -> Dict[float, str]:
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
        fixpoints: Dict[float, str] = {}
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
            if not np.isfinite(slope) or slope > 1:
                fixpoints[cross_point] = "unstable"
                continue
            if len(solves_array) == 0:
                fixpoints[cross_point] = "unstable"
                continue
            closest_idx = int(np.argmin(np.abs(v_out_old - cross_point)))
            closest_idx = min(max(closest_idx, 0), len(solves_array) - 1)
            initial = solves_array[closest_idx]
            system = cls(parameter, cross_point, **network_kwargs)
            solve, residual, success = system.solve(initial)
            if not success or not np.isfinite(residual).all():
                print("Warning: convergence problems near cross-point")
                fixpoints[cross_point] = "unstable"
                continue
            jacobian = system.jacobian_numpy(solve)
            if not np.isfinite(jacobian).all():
                fixpoints[cross_point] = "unstable"
                continue
            try:
                eigval = np.linalg.eigvals(jacobian)
            except np.linalg.LinAlgError:
                fixpoints[cross_point] = "unstable"
                continue
            fixpoints[cross_point] = "stable" if (eigval < 0).all() else "unstable"
        return fixpoints


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


def serialize_erf(file_path: str, parameter: Dict, result: ERFResult) -> Optional[str]:
    if not result.completed:
        return None
    payload = {str(parameter["R_Eplus"]): [result.x_data, result.y_data, result.solves, parameter]}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        import pickle

        pickle.dump(payload, file)
    return file_path


def aggregate_data(folder: str) -> str:
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
    "interpolate_curve",
    "ensure_output_folder",
    "serialize_erf",
    "aggregate_data",
]
