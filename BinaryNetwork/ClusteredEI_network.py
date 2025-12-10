from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .BinaryNetwork import (
    AllToAllSynapse,
    BackgroundActivity,
    BinaryNetwork as BaseBinaryNetwork,
    BinaryNeuronPopulation,
    FixedIndegreeSynapse,
    Neuron,
    PairwiseBernoulliSynapse,
    PoissonSynapse,
)


def _normalize_conn_type(kind: str | None) -> str:
    label = "bernoulli" if kind is None else str(kind).replace("_", "-").lower()
    if label not in {"bernoulli", "poisson", "fixed-indegree"}:
        raise ValueError(f"Unknown connection_type '{kind}'. Expected 'bernoulli', 'poisson', or 'fixed-indegree'.")
    return label


def _split_counts(total: int, groups: int) -> List[int]:
    if groups <= 0:
        raise ValueError("Q must be positive.")
    base = total // groups
    remainder = total % groups
    counts = []
    for idx in range(groups):
        counts.append(base + (1 if idx < remainder else 0))
    return counts


def _mix_scales(R_plus: float, Q: int, kappa: float) -> Tuple[float, float, float, float]:
    if Q <= 1:
        value = max(R_plus, 0.0)
        weight = value ** kappa
        return value ** (1.0 - kappa), value ** (1.0 - kappa), weight, weight
    prob_in = R_plus ** (1.0 - kappa)
    prob_out = (Q - prob_in) / (Q - 1)
    weight_in = R_plus ** kappa
    weight_out = (Q - weight_in) / (Q - 1)
    return prob_in, prob_out, weight_in, weight_out


def _compute_cluster_parameters(parameter: Dict, kappa: float) -> Dict[str, np.ndarray]:
    N = float(parameter["N"])
    Q = int(parameter["Q"])
    N_E = float(parameter["N_E"])
    N_I = float(parameter["N_I"])
    V_th = float(parameter["V_th"])
    g = float(parameter["g"])
    p0_ee = float(parameter["p0_ee"])
    p0_ie = float(parameter["p0_ie"])
    p0_ei = float(parameter["p0_ei"])
    p0_ii = float(parameter["p0_ii"])
    R_Eplus = parameter.get("R_Eplus")
    if R_Eplus is None:
        raise ValueError("R_Eplus must be set for binary simulations.")
    R_Eplus = float(R_Eplus)
    R_j = float(parameter.get("R_j", 0.0))

    n_er = N_E / N
    n_ir = N_I / N
    theta_E = V_th
    theta_I = V_th
    R_Iplus = 1.0 + R_j * (R_Eplus - 1.0)

    j_EE = theta_E / math.sqrt(p0_ee * n_er)
    j_IE = theta_I / math.sqrt(p0_ie * n_er)
    j_EI = -g * j_EE * p0_ee * n_er / (p0_ei * n_ir)
    j_II = -j_IE * p0_ie * n_er / (p0_ii * n_ir)

    scale = 1.0 / math.sqrt(N)
    j_EE *= scale
    j_IE *= scale
    j_EI *= scale
    j_II *= scale

    prob_in_E, prob_out_E, weight_in_E, weight_out_E = _mix_scales(R_Eplus, Q, kappa)
    prob_in_I, prob_out_I, weight_in_I, weight_out_I = _mix_scales(R_Iplus, Q, kappa)

    P_EE = p0_ee * prob_in_E
    p_ee = p0_ee * prob_out_E
    P_EI = p0_ei * prob_in_I
    p_ei = p0_ei * prob_out_I
    P_IE = p0_ie * prob_in_I
    p_ie = p0_ie * prob_out_I
    P_II = p0_ii * prob_in_I
    p_ii = p0_ii * prob_out_I

    J_EE = j_EE * weight_in_E
    j_ee = j_EE * weight_out_E
    J_EI = j_EI * weight_in_I
    j_ei = j_EI * weight_out_I
    J_IE = j_IE * weight_in_I
    j_ie = j_IE * weight_out_I
    J_II = j_II * weight_in_I
    j_ii = j_II * weight_out_I

    J_EX = math.sqrt(p0_ee * N_E)
    J_IX = 0.8 * J_EX

    return {
        "p_plus": np.array([[P_EE, P_EI], [P_IE, P_II]], dtype=float),
        "p_minus": np.array([[p_ee, p_ei], [p_ie, p_ii]], dtype=float),
        "j_plus": np.array([[J_EE, J_EI], [J_IE, J_II]], dtype=float),
        "j_minus": np.array([[j_ee, j_ei], [j_ie, j_ii]], dtype=float),
        "theta_E": theta_E,
        "theta_I": theta_I,
        "external_exc": J_EX,
        "external_inh": J_IX,
    }


def _flatten_values(values) -> List:
    if values is None:
        return [None]
    if isinstance(values, np.ndarray):
        return values.flatten().tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _normalize_activity_entry(entry, count: int, default_mode: str) -> List[Dict[str, float] | None]:
    if entry is None:
        return [None] * count
    mode = default_mode
    values = entry
    if isinstance(entry, dict):
        mode = entry.get("mode", default_mode)
        if "values" in entry:
            values = entry["values"]
        elif "value" in entry:
            values = entry["value"]
        else:
            values = None
    items = _flatten_values(values)
    if len(items) == 1 and count > 1:
        items = items * count
    if len(items) != count:
        raise ValueError(f"Initializer definition must provide {count} entries, got {len(items)}.")
    normalized: List[Dict[str, float] | None] = []
    for value in items:
        if value is None:
            normalized.append(None)
        else:
            normalized.append({"mode": mode, "value": float(value)})
    return normalized


def _make_initializer(spec: Dict[str, float] | None, size: int) -> Callable[[int], np.ndarray] | None:
    if spec is None:
        return None
    mode = spec.get("mode", "bernoulli").lower()
    value = float(spec.get("value", 0.0))
    value = min(max(value, 0.0), 1.0)
    if mode == "deterministic":
        def initializer(_count=size, frac=value):
            ones = int(round(frac * _count))
            ones = min(max(ones, 0), _count)
            state = np.zeros(_count, dtype=np.int16)
            if ones > 0:
                state[:ones] = 1
                np.random.shuffle(state)
            return state
        return initializer
    if mode == "bernoulli":
        def initializer(_count=size, prob=value):
            return (np.random.random(_count) < prob).astype(np.int16)
        return initializer
    raise ValueError(f"Unsupported initializer mode '{mode}'.")


def _resolve_initializers(config, excit_sizes: Sequence[int], inhib_sizes: Sequence[int]) -> Tuple[List, List]:
    if config is None:
        return [None] * len(excit_sizes), [None] * len(inhib_sizes)
    if not isinstance(config, dict):
        config = {"default": config}
    default_mode = str(config.get("mode", "bernoulli")).lower()
    excit_specs: List[Dict[str, float] | None] = [None] * len(excit_sizes)
    inhib_specs: List[Dict[str, float] | None] = [None] * len(inhib_sizes)

    def apply(entry, targets):
        if entry is None:
            return
        normalized = _normalize_activity_entry(entry, len(targets), default_mode)
        for idx, spec in enumerate(normalized):
            if spec is not None:
                targets[idx] = spec

    apply(config.get("default"), excit_specs)
    apply(config.get("default"), inhib_specs)
    apply(config.get("excitatory"), excit_specs)
    apply(config.get("inhibitory"), inhib_specs)
    apply(config.get("excitatory_by_cluster"), excit_specs)
    apply(config.get("inhibitory_by_cluster"), inhib_specs)

    excit_initializers = [_make_initializer(spec, size) for spec, size in zip(excit_specs, excit_sizes)]
    inhib_initializers = [_make_initializer(spec, size) for spec, size in zip(inhib_specs, inhib_sizes)]
    return excit_initializers, inhib_initializers


class ClusteredEI_network(BaseBinaryNetwork):
    def __init__(self, parameter: Dict, *, kappa: float | None = None, connection_type: str | None = None, name="Binary EI Network"):
        super().__init__(name)
        self.parameter = dict(parameter)
        self.Q = int(self.parameter["Q"])
        self.connection_type = _normalize_conn_type(connection_type or self.parameter.get("connection_type"))
        self.kappa = float(kappa if kappa is not None else self.parameter.get("kappa", 0.0))
        self.connection_parameters = _compute_cluster_parameters(self.parameter, self.kappa)
        self.E_sizes = _split_counts(int(self.parameter["N_E"]), self.Q)
        self.I_sizes = _split_counts(int(self.parameter["N_I"]), self.Q)
        if sum(self.E_sizes) + sum(self.I_sizes) != int(self.parameter["N"]):
            print("Warning: N does not match N_E + N_I. Proceeding with per-type totals.")
        self.initializers_E, self.initializers_I = _resolve_initializers(
            self.parameter.get("initial_activity"),
            self.E_sizes,
            self.I_sizes,
        )
        self.E_pops: List[BinaryNeuronPopulation] = []
        self.I_pops: List[BinaryNeuronPopulation] = []
        self.other_pops: List[Neuron] = []
        self._structure_created = False

    def _build_populations(self):
        tau_e = float(self.parameter["tau_e"])
        tau_i = float(self.parameter["tau_i"])
        theta_E = self.connection_parameters["theta_E"]
        theta_I = self.connection_parameters["theta_I"]
        for idx, size in enumerate(self.E_sizes):
            pop = BinaryNeuronPopulation(
                self,
                N=size,
                threshold=theta_E,
                tau=tau_e,
                name=f"E{idx}",
                initializer=self.initializers_E[idx],
            )
            pop.cluster_index = idx
            pop.cell_type = "E"
            self.E_pops.append(self.add_population(pop))
        for idx, size in enumerate(self.I_sizes):
            pop = BinaryNeuronPopulation(
                self,
                N=size,
                threshold=theta_I,
                tau=tau_i,
                name=f"I{idx}",
                initializer=self.initializers_I[idx],
            )
            pop.cluster_index = idx
            pop.cell_type = "I"
            self.I_pops.append(self.add_population(pop))

    def _synapse_factory(self):
        if self.connection_type == "poisson":
            return lambda pre, post, p, j: PoissonSynapse(self, pre, post, rate=p, j=j)
        if self.connection_type == "fixed-indegree":
            return lambda pre, post, p, j: FixedIndegreeSynapse(self, pre, post, p=p, j=j)
        return lambda pre, post, p, j: PairwiseBernoulliSynapse(self, pre, post, p=p, j=j)

    def _build_synapses(self):
        builder = self._synapse_factory()
        p_plus = self.connection_parameters["p_plus"]
        p_minus = self.connection_parameters["p_minus"]
        j_plus = self.connection_parameters["j_plus"]
        j_minus = self.connection_parameters["j_minus"]
        pre_groups = [self.E_pops, self.I_pops]
        for target_group_idx, target_pops in enumerate([self.E_pops, self.I_pops]):
            for source_group_idx, source_pops in enumerate(pre_groups):
                for post_pop in target_pops:
                    for pre_pop in source_pops:
                        same_cluster = pre_pop.cluster_index == post_pop.cluster_index
                        p_matrix = p_plus if same_cluster else p_minus
                        j_matrix = j_plus if same_cluster else j_minus
                        p_value = p_matrix[target_group_idx, source_group_idx]
                        j_value = j_matrix[target_group_idx, source_group_idx]
                        self.add_synapse(builder(pre_pop, post_pop, p_value, j_value))

    def _build_background(self):
        m_X = float(self.parameter.get("m_X", 0.0) or 0.0)
        if m_X == 0.0:
            return
        bg = BackgroundActivity(self, N=1, Activity=m_X, name="Background")
        #self.other_pops.append(self.add_population(bg))
        J_EX = float(self.connection_parameters["external_exc"])
        J_IX = float(self.connection_parameters["external_inh"])
        for pop in self.E_pops:
            pop.threshold-=m_X*J_EX
            #self.add_synapse(AllToAllSynapse(self, bg, pop, j=J_EX))
        for pop in self.I_pops:
            pop.threshold -= m_X * J_IX
            #self.add_synapse(AllToAllSynapse(self, bg, pop, j=J_IX))

    def _ensure_structure(self):
        if self._structure_created:
            return
        self._build_populations()
        self._build_synapses()
        self._build_background()
        self._structure_created = True

    def initialize(self, autapse: bool = True):
        self._ensure_structure()
        super().initialize(autapse=autapse)

    def reinitalize(self):
        self.initialize()
