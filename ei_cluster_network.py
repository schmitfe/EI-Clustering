from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from rate_system import RateSystem


class EIClusterNetwork(RateSystem):
    """Specialized rate system for the EI-cluster network."""

    def __init__(
        self,
        parameter: Dict,
        v_focus: float,
        *,
        kappa: float | None = None,
        connection_type: str | None = None,
        focus_population=None,
        prefer_jax: bool = True,
        max_steps: int = 256,
    ) -> None:
        self.Q = int(parameter["Q"])
        self._explicit_kappa = kappa
        self._explicit_connection = connection_type
        self.collapse_types = bool(parameter.get("collapse_types", True))
        super().__init__(
            parameter,
            v_focus,
            focus_population=focus_population,
            prefer_jax=prefer_jax,
            max_steps=max_steps,
            kappa=kappa,
            connection_type=connection_type,
        )

    def _build_dynamics(self, parameter: Dict, **network_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kappa_value = network_kwargs.get("kappa")
        if kappa_value is None:
            kappa_value = parameter.get("kappa", 0.0)
        conn_kind = network_kwargs.get("connection_type") or parameter.get("connection_type", "bernoulli")
        A, B, bias = self._build_connectivity(parameter, float(kappa_value), str(conn_kind))
        tau = np.ones(2 * self.Q, dtype=float)
        tau[: self.Q] *= parameter["tau_e"]
        tau[self.Q :] *= parameter["tau_i"]
        return A, B, bias, tau

    def _build_population_groups(self, focus: np.ndarray) -> List[np.ndarray]:
        focus_set = set(int(idx) for idx in focus.tolist())
        if not focus_set:
            focus_set = {0}
        if not self.collapse_types:
            return self._build_full_focus_groups(focus_set)
        return self._build_collapsed_groups(focus_set)

    def _build_full_focus_groups(self, focus_set: set[int]) -> List[np.ndarray]:
        groups: List[np.ndarray] = [np.array(sorted(focus_set), dtype=int)]
        excit_focus = sorted(idx for idx in focus_set if 0 <= idx < self.Q)
        paired_inhib = sorted(
            {
                idx + self.Q
                for idx in excit_focus
                if idx + self.Q < 2 * self.Q and (idx + self.Q) not in focus_set
            }
        )
        if paired_inhib:
            groups.append(np.array(paired_inhib, dtype=int))
        remaining_excit = [idx for idx in range(self.Q) if idx not in focus_set]
        for idx in remaining_excit:
            groups.append(np.array([idx], dtype=int))
        remaining_inhib = [
            idx for idx in range(self.Q, 2 * self.Q) if idx not in focus_set and idx not in paired_inhib
        ]
        for idx in remaining_inhib:
            groups.append(np.array([idx], dtype=int))
        return groups

    def _build_collapsed_groups(self, focus_set: set[int]) -> List[np.ndarray]:
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
        other_excit = sorted(idx for idx in range(self.Q) if idx not in focus_set and idx not in paired_excit)
        other_inhib = sorted(
            idx for idx in range(self.Q, 2 * self.Q) if idx not in focus_set and idx not in paired_inhib
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

    def _build_connectivity(self, parameter: Dict, kappa: float, connection_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = parameter["N"]
        N_E = parameter["N_E"]
        N_I = parameter["N_I"]
        V_th = parameter["V_th"]
        g = parameter["g"]
        p0_ee = parameter["p0_ee"]
        p0_ie = parameter["p0_ie"]
        p0_ei = parameter["p0_ei"]
        p0_ii = parameter["p0_ii"]
        m_X = parameter["m_X"]
        R_Eplus = parameter["R_Eplus"]
        R_j = parameter["R_j"]

        n_er = N_E / N
        n_ir = N_I / N
        n_e = N_E / self.Q
        n_i = N_I / self.Q

        theta_E = V_th
        theta_I = V_th
        V_th_vec = np.array([theta_E] * self.Q + [theta_I] * self.Q, dtype=float)

        R_Iplus = 1 + R_j * (R_Eplus - 1)

        j_EE = theta_E / math.sqrt(p0_ee * n_er)
        j_IE = theta_I / math.sqrt(p0_ie * n_er)
        j_EI = -g * j_EE * p0_ee * n_er / (p0_ei * n_ir)
        j_II = -j_IE * p0_ie * n_er / (p0_ii * n_ir)

        scale = 1.0 / math.sqrt(N)
        j_EE *= scale
        j_IE *= scale
        j_EI *= scale
        j_II *= scale

        def mix_scales(R_plus: float) -> tuple[float, float, float, float]:
            """
            Compute scaling factors for in-cluster and out-of-cluster connections.
            
            When Q > 1: Distributes connection strengths between in-cluster (prob_in, weight_in) 
            and out-of-cluster (prob_out, weight_out) based on the clustering strength R_plus.
            
            When Q = 1: Only one cluster exists, so there are no out-of-cluster connections.
            In this case, prob_out and weight_out are set equal to prob_in and weight_in,
            which ensures that all connection parameters use the same scaling without 
            division by zero. The structured matrix builder will handle the single-cluster 
            case appropriately.
            """
            prob_in = R_plus ** (1.0 - kappa)
            weight_in = R_plus ** kappa
            if self.Q == 1:
                # With a single cluster, all connections are in-cluster
                prob_out = prob_in
                weight_out = weight_in
            else:
                prob_out = (self.Q - prob_in) / (self.Q - 1)
                weight_out = (self.Q - weight_in) / (self.Q - 1)
            return prob_in, prob_out, weight_in, weight_out

        P_scale_in_E, P_scale_out_E, J_scale_in_E, J_scale_out_E = mix_scales(R_Eplus)
        P_scale_in_I, P_scale_out_I, J_scale_in_I, J_scale_out_I = mix_scales(R_Iplus)

        P_EE = p0_ee * P_scale_in_E
        p_ee = p0_ee * P_scale_out_E
        P_IE = p0_ie * P_scale_in_I
        p_ie = p0_ie * P_scale_out_I
        P_EI = p0_ei * P_scale_in_I
        p_ei = p0_ei * P_scale_out_I
        P_II = p0_ii * P_scale_in_I
        p_ii = p0_ii * P_scale_out_I

        J_EE = j_EE * J_scale_in_E
        j_ee = j_EE * J_scale_out_E
        J_IE = j_IE * J_scale_in_I
        j_ie = j_IE * J_scale_out_I
        J_EI = j_EI * J_scale_in_I
        j_ei = j_EI * J_scale_out_I
        J_II = j_II * J_scale_in_I
        j_ii = j_II * J_scale_out_I

        EE_IN = J_EE * P_EE * n_e
        EE_OUT = j_ee * p_ee * n_e
        IE_IN = J_IE * P_IE * n_e
        IE_OUT = j_ie * p_ie * n_e
        EI_IN = J_EI * P_EI * n_i
        EI_OUT = j_ei * p_ei * n_i
        II_IN = J_II * P_II * n_i
        II_OUT = j_ii * p_ii * n_i

        mean_values = dict(EE_IN=EE_IN, EE_OUT=EE_OUT, IE_IN=IE_IN, IE_OUT=IE_OUT, EI_IN=EI_IN, EI_OUT=EI_OUT,
                           II_IN=II_IN, II_OUT=II_OUT)
        A = self._structured_matrix(mean_values)

        var_values = dict(
            EE_IN=self._variance(P_EE, J_EE, n_e, connection_type),
            EE_OUT=self._variance(p_ee, j_ee, n_e, connection_type),
            IE_IN=self._variance(P_IE, J_IE, n_e, connection_type),
            IE_OUT=self._variance(p_ie, j_ie, n_e, connection_type),
            EI_IN=self._variance(P_EI, J_EI, n_i, connection_type),
            EI_OUT=self._variance(p_ei, j_ei, n_i, connection_type),
            II_IN=self._variance(P_II, J_II, n_i, connection_type),
            II_OUT=self._variance(p_ii, j_ii, n_i, connection_type),
        )
        B = self._structured_matrix(var_values)

        J_EX = math.sqrt(p0_ee * N_E)
        J_IX = 0.8 * J_EX
        u_extE = J_EX * m_X
        u_extI = J_IX * m_X
        u_ext = np.array([u_extE] * self.Q + [u_extI] * self.Q, dtype=float)
        bias = u_ext - V_th_vec
        return A, B, bias

    def _structured_matrix(self, values: Dict[str, float]) -> np.ndarray:
        size = 2 * self.Q
        matrix = np.zeros((size, size), dtype=float)
        for target in range(size):
            tgt_type = "E" if target < self.Q else "I"
            tgt_cluster = target % self.Q
            for source in range(size):
                src_type = "E" if source < self.Q else "I"
                src_cluster = source % self.Q
                suffix = "_IN" if tgt_cluster == src_cluster else "_OUT"
                key = f"{tgt_type}{src_type}{suffix}"
                matrix[target, source] = values[key]
        return matrix

    @staticmethod
    def _variance(prob: float, weight: float, population: float, connection_type: str) -> float:
        conn_kind = connection_type.lower()
        if conn_kind == "poisson":
            return prob * weight ** 2 * population
        if conn_kind == "fixed-indegree":
            return prob * (1 - (1 / population)) * weight ** 2 * population
        # else: Bernoulli
        return prob * (1 - prob) * weight ** 2 * population


__all__ = ["EIClusterNetwork"]
