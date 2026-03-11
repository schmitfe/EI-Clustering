#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from plotting import FontCfg, add_panel_label, style_axes  # noqa: E402
from sim_config import add_override_arguments, deep_update, load_from_args, parse_overrides  # noqa: E402


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
DEFAULT_OUTPUT = FIGURES_DIR / "FigureEigenvalues"

BINET_SRC = REPO_ROOT / "legacy" / "Rost" / "code" / "BiNet" / "src"
if str(BINET_SRC) not in sys.path:
    sys.path.insert(0, str(BINET_SRC))

from BiNet import weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eigenvalue distributions for clustered weight matrices at multiple kappa values."
    )
    add_override_arguments(parser)
    parser.add_argument(
        "--kappa-values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Kappa values to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base RNG seed for weight sampling (default: random).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Prefix for saved figure files (default: %(default)s.{png,pdf}).",
    )
    parser.add_argument(
        "--column-override",
        action="append",
        default=[],
        metavar="index:path=value",
        help="Apply overrides to a specific column (0-based index).",
    )
    return parser.parse_args()


def _normalize_connection_type(value: object) -> str:
    label = "bernoulli" if value is None else str(value).replace(" ", "_").replace("-", "_").lower()
    valid = {"bernoulli", "poisson", "fixed_indegree"}
    if label not in valid:
        raise ValueError(f"Unknown connection_type '{value}'. Expected one of {sorted(valid)}.")
    return label


def _resolve_r_eplus(parameter: Dict[str, object]) -> float:
    r_value = parameter.get("R_Eplus")
    if r_value is None:
        q_value = parameter.get("Q")
        if q_value is None:
            raise ValueError("R_Eplus is missing and Q is undefined; provide -O R_Eplus=... .")
        return float(q_value)
    return float(r_value)


def _parse_column_overrides(
    overrides: Sequence[str],
    column_count: int,
) -> Dict[int, Dict[str, object]]:
    parsed: Dict[int, Dict[str, object]] = {}
    for entry in overrides:
        if ":" not in entry:
            raise ValueError(f"Column override '{entry}' is missing ':' (expected index:path=value).")
        index_str, override = entry.split(":", 1)
        try:
            index = int(index_str)
        except ValueError as exc:
            raise ValueError(f"Invalid column index '{index_str}' in override '{entry}'.") from exc
        if not 0 <= index < column_count:
            raise ValueError(f"Column index {index} out of range for {column_count} columns.")
        override_dict = parse_overrides([override])
        parsed[index] = deep_update(parsed.get(index, {}), override_dict)
    return parsed


def _cluster_specs(parameter: Dict[str, object], kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_exc = int(parameter.get("N_E", 0))
    n_inh = int(parameter.get("N_I", 0))
    if n_exc <= 0 or n_inh <= 0:
        raise ValueError("N_E and N_I must be positive to build the weight matrix.")
    q_value = int(parameter.get("Q", 0))
    if q_value <= 0:
        raise ValueError("Q must be a positive integer.")
    if n_exc % q_value != 0 or n_inh % q_value != 0:
        raise ValueError("N_E and N_I must be divisible by Q for EI_jplus_cluster_specs.")

    v_th = float(parameter.get("V_th", 0.0))
    g_value = float(parameter.get("g", 0.0))
    tau_e = float(parameter.get("tau_e", 1.0))
    tau_i = float(parameter.get("tau_i", 1.0))
    p0_ee = float(parameter.get("p0_ee", 0.0))
    p0_ei = float(parameter.get("p0_ei", 0.0))
    p0_ie = float(parameter.get("p0_ie", 0.0))
    p0_ii = float(parameter.get("p0_ii", 0.0))
    r_eplus = _resolve_r_eplus(parameter)
    r_j = float(parameter.get("R_j", 0.0))
    r_iplus = 1.0 + r_j * (r_eplus - 1.0)

    Ns = np.array([n_exc, n_inh], dtype=int)
    ps = np.array([[p0_ee, p0_ei], [p0_ie, p0_ii]], dtype=float)
    Ts = np.array([v_th, v_th], dtype=float)
    taus = np.array([tau_e, tau_i], dtype=float)
    jplus = np.array([[r_eplus, r_iplus], [r_iplus, r_iplus]], dtype=float)

    cNs, cps, cjs, _, _ = weights.EI_jplus_cluster_specs(
        Ns=Ns,
        ps=ps,
        Ts=Ts,
        taus=taus,
        g=g_value,
        Q=q_value,
        jplus=jplus,
        kappa=kappa,
    )
    return np.asarray(cNs, dtype=int), np.asarray(cps, dtype=float), np.asarray(cjs, dtype=float)


def _compute_eigenvalues(
    parameter: Dict[str, object],
    kappa_values: Sequence[float],
    *,
    base_seed: int | None,
) -> List[np.ndarray]:
    conn_type = _normalize_connection_type(parameter.get("connection_type"))
    eigvals: List[np.ndarray] = []
    for idx, kappa in enumerate(kappa_values):
        if base_seed is not None:
            np.random.seed(int(base_seed) + idx)
        cNs, cps, cjs = _cluster_specs(parameter, float(kappa))
        weight_matrix = weights.generate_weight_matrix(
            cNs,
            cps,
            cjs,
            delta_j=None,
            connection_type=conn_type,
        )
        eigvals.append(np.linalg.eigvals(weight_matrix))
    return eigvals


def _plot_eigenvalues(
    eigvals: Sequence[np.ndarray],
    kappa_values: Sequence[float],
    *,
    output_prefix: str,
) -> None:
    n_cols = len(eigvals)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.0), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]
    font_cfg = FontCfg(base=12.0, scale=1.3).resolve()

    all_real = np.concatenate([vals.real for vals in eigvals]) if eigvals else np.zeros(0)
    all_imag = np.concatenate([vals.imag for vals in eigvals]) if eigvals else np.zeros(0)
    if all_real.size == 0 or all_imag.size == 0:
        raise ValueError("No eigenvalues were computed; cannot plot.")
    extent = max(np.max(np.abs(all_real)), np.max(np.abs(all_imag)))
    extent = 1.05 * extent if extent > 0 else 1.0

    for idx, (ax, vals, kappa) in enumerate(zip(axes, eigvals, kappa_values)):
        abs_vals = np.abs(vals)
        r_95 = float(np.percentile(abs_vals, 95))
        rho_w = float(abs_vals.max())
        max_re = float(vals.real.max())
        ax.scatter(vals.real, vals.imag, s=4, color="black", alpha=0.35, linewidths=0)
        ax.add_patch(Circle((0.0, 0.0), r_95, edgecolor="red", facecolor="none", linewidth=1.2))
        ax.set_title(rf"$\kappa={kappa:g}$", fontsize=font_cfg.title)
        if idx == 0:
            ax.set_ylabel(r"$\Im(\lambda)$")
        ax.set_xlabel(r"$\Re(\lambda)$")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal", adjustable="box")
        s = "\n".join([
            fr"$\rho_{{95}} = {r_95:.3g}$",
            fr"$\max(|\lambda|) = {rho_w:.3g}$",
            fr"$\max(\Re(\lambda)) = {max_re:.3g}$",
        ])

        ax.text(0.98, 0.98, s, transform=ax.transAxes,
                ha="right", va="top", fontsize=font_cfg.tick, linespacing=1.2)

        # ax.text(
        #     0.98,
        #     0.98,
        #     fr"$r_{{95}} = {r_95:.3g}$\n"
        #     fr"$\rho_w = {rho_w:.3g}$\n"
        #     fr"$\max\Re(\lambda) = {max_re:.3g}$",
        #     transform=ax.transAxes,
        #     ha="right",
        #     va="top",
        #     fontsize=font_cfg.tick,
        # )
        style_axes(ax, font_cfg)
        add_panel_label(ax, chr(ord("a") + idx), font_cfg, x=-0.1, y=1.05)

    #fig.tight_layout()
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"), dpi=600)
    plt.close(fig)
    print(f"Stored eigenvalue plot at {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    base_parameter = load_from_args(args)
    kappa_values = [float(value) for value in args.kappa_values]
    column_overrides = _parse_column_overrides(args.column_override, len(kappa_values))
    eigvals: List[np.ndarray] = []
    for idx, kappa in enumerate(kappa_values):
        param = deepcopy(base_parameter)
        if idx in column_overrides:
            param = deep_update(param, column_overrides[idx])
        param["kappa"] = float(kappa)
        seed = None if args.seed is None else int(args.seed) + idx
        eigvals.extend(
            _compute_eigenvalues(
                param,
                [kappa],
                base_seed=seed,
            )
        )
    _plot_eigenvalues(eigvals, kappa_values, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
