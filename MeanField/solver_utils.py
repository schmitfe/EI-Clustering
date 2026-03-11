"""
Helper utilities for compiling symbolic rate equations into efficient numerical callables.

Whenever possible the helpers rely on JAX to differentiate the symbolic expressions, but they
fall back to cached SymPy derivatives when JAX is unavailable.  The resulting callables share
the same signature as the original SymPy expressions (``f(v2, v3, ...)``) and always return
NumPy arrays so the solvers can work with a consistent interface.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import sympy
from sympy import Matrix, Symbol, diff

logger = logging.getLogger(__name__)

__all__: list[str] = []
__pdoc__ = {
    "HAS_JAX": False,
    "SystemFunctions": False,
    "prepare_system_functions": False,
}


def _is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get("EI_CLUSTERING_DEBUG", "").lower() in ("1", "true", "yes")


try:
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    HAS_JAX = True
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    HAS_JAX = False


@dataclass(frozen=True)
class SystemFunctions:
    """Container holding compiled residual, Jacobian, Hessian, and residual vector helpers."""

    F: Callable[..., np.ndarray]
    J: Callable[..., np.ndarray]
    H: Callable[..., np.ndarray]
    value_func: Callable[..., np.ndarray]
    backend: str


_SYSTEM_CACHE: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], SystemFunctions] = {}


def _cache_key(funcs: Sequence[sympy.Expr], var_names: Sequence[str], backend: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    expr_key = tuple(sympy.srepr(f) for f in funcs)
    return backend, expr_key, tuple(var_names)


def _wrap_numeric(func: Callable[..., np.ndarray], flatten: bool = False) -> Callable[..., np.ndarray]:
    def _wrapped(*args):
        value = np.asarray(func(*args), dtype=float)
        return value.reshape(-1) if flatten else value

    return _wrapped


def _build_symbolic_bundle(funcs: Sequence[sympy.Expr], var_names: Sequence[str]) -> SystemFunctions:
    symbols = [Symbol(name) for name in var_names]
    func_matrix = Matrix(funcs)
    F_func = sympy.lambdify(symbols, func_matrix, modules="numpy")

    jac_rows = [[diff(phi, sym) for sym in symbols] for phi in funcs]
    jac_matrix = Matrix(jac_rows)
    J_func = sympy.lambdify(symbols, jac_matrix, modules="numpy")

    hess_rows = [[diff(phi, sym, sym) for sym in symbols] for phi in funcs]
    h_matrix = Matrix(hess_rows)
    H_func = sympy.lambdify(symbols, h_matrix, modules="numpy")

    return SystemFunctions(
        F=_wrap_numeric(F_func),
        J=_wrap_numeric(J_func),
        H=_wrap_numeric(H_func),
        value_func=_wrap_numeric(F_func, flatten=True),
        backend="sympy",
    )


def _build_autodiff_bundle(funcs: Sequence[sympy.Expr], var_names: Sequence[str]) -> SystemFunctions:
    if not HAS_JAX:
        raise RuntimeError("JAX is not available in the current environment.")

    symbols = [Symbol(name) for name in var_names]
    func_matrix = Matrix(funcs)
    raw_func = sympy.lambdify(symbols, func_matrix, modules="jax")
    var_len = len(symbols)

    def _vectorized(x):
        values = [x[i] for i in range(var_len)]
        return jnp.asarray(raw_func(*values)).reshape(-1)

    jacobian_raw = jax.jacobian(_vectorized)
    hessian_raw = jax.jacobian(jacobian_raw)

    def F_func(*args):
        return np.asarray(raw_func(*args), dtype=float)

    def J_func(*args):
        x = jnp.asarray(args, dtype=jnp.float64)
        return np.asarray(jacobian_raw(x), dtype=float)

    def H_func(*args):
        x = jnp.asarray(args, dtype=jnp.float64)
        hess = hessian_raw(x)
        diag = jnp.diagonal(hess, axis1=-2, axis2=-1)
        return np.asarray(diag, dtype=float)

    return SystemFunctions(
        F=_wrap_numeric(F_func),
        J=_wrap_numeric(J_func),
        H=_wrap_numeric(H_func),
        value_func=_wrap_numeric(F_func, flatten=True),
        backend="jax",
    )


def prepare_system_functions(
    funcs: Sequence[sympy.Expr],
    var: Sequence[str],
    prefer_autodiff: bool = True,
) -> SystemFunctions:
    """
    Compile symbolic functions into numerical residual/Jacobian/Hessian callables.

    Parameters
    ----------
    funcs:
        The symbolic expressions describing the residual vector.
    var:
        The ordered sequence of variable names matching the symbolic expressions.
    prefer_autodiff:
        Try to compile the expressions via JAX autodiff when available.  Falls back to
        cached SymPy derivatives when JAX cannot be used.

    Returns
    -------
    SystemFunctions
        Bundle of numerical residual, Jacobian, Hessian, and helper callables.
    """
    var_names = tuple(str(v) for v in var)
    use_autodiff = prefer_autodiff and HAS_JAX
    backend = "jax" if use_autodiff else "sympy"
    key = _cache_key(funcs, var_names, backend)

    if key in _SYSTEM_CACHE:
        return _SYSTEM_CACHE[key]

    builder = _build_autodiff_bundle if use_autodiff else _build_symbolic_bundle

    try:
        bundle = builder(funcs, var_names)
    except Exception as e:
        if use_autodiff:
            # Log the JAX compilation failure for debugging purposes
            logger.warning(
                "JAX compilation failed, falling back to SymPy: %s: %s",
                type(e).__name__,
                str(e)
            )
            # In debug mode, re-raise the exception to help developers diagnose issues
            if _is_debug_mode():
                raise

            backend = "sympy"
            key = _cache_key(funcs, var_names, backend)
            if key in _SYSTEM_CACHE:
                return _SYSTEM_CACHE[key]
            bundle = _build_symbolic_bundle(funcs, var_names)
        else:
            raise

    _SYSTEM_CACHE[key] = bundle
    return bundle

