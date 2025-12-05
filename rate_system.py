import os
from dataclasses import dataclass
from typing import Optional, Tuple

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
    prefer_jax: bool = True
    max_steps: int = 256

    def __post_init__(self):
        param_copy = dict(self.parameter)
        mix = self.kappa if self.kappa is not None else param_copy.pop("kappa", None)
        if mix is None:
            mix = 0.0
        matrices = connectivit.linear_connectivity(mix, **param_copy)
        self.A = matrices.A
        self.B = matrices.B
        self.bias = matrices.bias
        self.Q = self.parameter["Q"]
        tau_e = self.parameter["tau_e"]
        tau_i = self.parameter["tau_i"]
        self.tau = np.ones((2 * self.Q,), dtype=float)
        self.tau[:self.Q] *= tau_e
        self.tau[self.Q:] *= tau_i
        self.dim = 2 * self.Q - 1
        self.use_jax = self.prefer_jax and HAS_JAX and optx is not None
        self._jax_args = None

    def _full_rates_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(([self.v1_fixed], x))

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
        return residual[1:]

    @staticmethod
    def _jax_residual(x, args):
        A, B, bias, tau, v1 = args
        full_rates = jnp.concatenate((jnp.array([v1], dtype=x.dtype), x))
        mean = A @ full_rates + bias
        var = jnp.maximum(B @ full_rates, 1e-12)
        phi = 0.5 * (1 - jspecial.erf(-mean / jnp.sqrt(2.0 * var)))
        residual = (phi - full_rates) / tau
        return residual[1:]

    def solve(self, initial_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        initial = np.asarray(initial_guess, dtype=float).reshape((self.dim,))
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
            )
        return self._jax_args

    def phi_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = self._full_rates_numpy(np.asarray(x, dtype=float))
        return self._phi_numpy(rates)

    def jacobian_numpy(self, x: np.ndarray) -> np.ndarray:
        rates = np.asarray(x, dtype=float)
        if rates.shape[0] == self.dim:
            rates = self._full_rates_numpy(rates)
        return (self._phi_jacobian_numpy(rates) - np.eye(2 * self.Q)) / self.tau[:, np.newaxis]


__all__ = ["RateSystem"]
