"""BrainScaleS-2 clustered network with two parent populations.

The original :mod:`network` implementation creates one PyNN Population per
cluster.  This variant creates one excitatory and one inhibitory Population and
uses randomized PopulationViews as clusters.  Consequently, every trial can
assign different physical neurons to clusters without relying on a private
BrainScaleS-2 placement API.
"""

from __future__ import annotations

import numpy as np

import network
import pynn_brainscales.brainscales2 as sim


class ViewClusteredNetwork(network.ClusteredNetwork):
    """ClusteredNetwork whose clusters are views into one E and one I pool.

    ``cluster_assignment_seed`` controls which parent-population neurons belong
    to each cluster.  If omitted, ``randseed`` is used.  Connectivity is still
    independently sampled by the seeded connector machinery in the base class.
    """

    def create_populations(self):
        params = self._params
        q = int(params["n_clusters"])
        n_e = int(params["N_E"])
        n_i = int(params["N_I"])

        if n_e % q != 0:
            raise ValueError("N_E must be a multiple of n_clusters")
        if n_i % q != 0:
            raise ValueError("N_I must be a multiple of n_clusters")
        if params["neuron_type"] != "iaf_psc_exp":
            raise ValueError("Model only implemented for iaf_psc_exp")
        if params.get("I_th_E") is None or params.get("I_th_I") is None:
            raise ValueError(
                "ViewClusteredNetwork currently requires I_th_E and I_th_I"
            )

        offset = 100
        e_neuron_params = {
            "v_rest": params["E_L"]
            + offset
            + params["I_th_E"] * (params["V_th_E"] - params["E_L"]),
            "cm": 63,
            "tau_m": params["tau_E"],
            "tau_refrac": params["t_ref"],
            "v_thresh": params["V_th_E"] + offset,
            "v_reset": params["V_r"] + offset,
            "tau_syn_E": params["tau_syn_ex"],
            "tau_syn_I": params["tau_syn_in"],
        }
        i_neuron_params = {
            "v_rest": params["E_L"]
            + offset
            + params["I_th_I"] * (params["V_th_E"] - params["E_L"]),
            "cm": 63,
            "tau_m": params["tau_I"],
            "tau_refrac": params["t_ref"],
            "v_thresh": params["V_th_I"] + offset,
            "v_reset": params["V_r"] + offset,
            "tau_syn_E": params["tau_syn_ex"],
            "tau_syn_I": params["tau_syn_in"],
        }
        if params.get("delta") is not None and params.get("rho") is not None:
            for neuron_params in (e_neuron_params, i_neuron_params):
                neuron_params["delta"] = params["delta"]
                neuron_params["rho"] = params["rho"]

        self._parent_populations = [
            sim.Population(n_e, sim.cells.CalibHXNeuronCuba(**e_neuron_params), label="E_all"),
            sim.Population(n_i, sim.cells.CalibHXNeuronCuba(**i_neuron_params), label="I_all"),
        ]

        assignment_seed = int(
            params.get("cluster_assignment_seed", params.get("randseed", 1))
        )
        assignment_rng = np.random.default_rng(assignment_seed)
        memberships = [assignment_rng.permutation(n_e), assignment_rng.permutation(n_i)]
        sizes = [n_e // q, n_i // q]
        labels = ["E", "I"]
        self._cluster_membership = []
        self._populations = []
        for parent, permutation, size, label in zip(
            self._parent_populations, memberships, sizes, labels
        ):
            type_views = []
            type_membership = []
            for cluster in range(q):
                indices = np.sort(permutation[cluster * size : (cluster + 1) * size])
                type_membership.append(indices.copy())
                type_views.append(
                    sim.PopulationView(parent, indices.tolist(), label=f"{label}{cluster}")
                )
            self._cluster_membership.append(type_membership)
            self._populations.append(type_views)

        self._validate_cluster_membership()

        # PREPARE performs calibration/placement.  Hardware parameters live on
        # the parent populations; the views only select neurons from them.
        bias = int(params.get("synin_bias", 400))
        sim.run(None, sim.RunCommand.PREPARE)
        for parent in self._parent_populations:
            for neuron in parent.actual_hwparams:
                neuron.excitatory_input.i_bias_gm = bias
                neuron.inhibitory_input.i_bias_gm = bias

        synapse_dac_bias = int(params.get("synapse_dac_bias", 1022))
        for synapse_block in sim.simulator.state.grenade_chip_config.synapse_blocks:
            for dac_idx, _ in enumerate(synapse_block.i_bias_dac):
                synapse_block.i_bias_dac[dac_idx] = synapse_dac_bias

    def create_recording_devices(self):
        """Record each parent once; recording overlapping views is unnecessary."""
        for parent in self._parent_populations:
            parent.record("spikes")

    def get_cluster_membership(self):
        """Return parent-population indices for every E and I cluster."""
        return [
            [indices.astype(int).tolist() for indices in memberships]
            for memberships in self._cluster_membership
        ]

    def _validate_cluster_membership(self):
        """Require each parent neuron to occur in exactly one cluster view."""
        expected_sizes = [int(self._params["N_E"]), int(self._params["N_I"])]
        for population_type, (memberships, expected_size) in enumerate(
            zip(self._cluster_membership, expected_sizes)
        ):
            flat = np.concatenate(memberships)
            if flat.size != expected_size or np.unique(flat).size != expected_size:
                raise RuntimeError(
                    f"Cluster views for population type {population_type} overlap or omit neurons"
                )
            if flat.min() < 0 or flat.max() >= expected_size:
                raise RuntimeError(
                    f"Cluster view index outside population type {population_type}"
                )

    def get_recordings(self):
        """Return spikes with IDs remapped to contiguous cluster-major order."""
        raw_parts = []
        parent_offset = 0
        self._recording_id_conventions = {}
        for parent in self._parent_populations:
            trains = parent.get_data().segments[-1].spiketrains
            multiplexed = np.asarray(trains.multiplexed)
            if multiplexed.size:
                multiplexed = multiplexed.copy()
                parent_indices, convention = self._parent_indices_from_source_ids(
                    multiplexed[0, :],
                    parent_offset=parent_offset,
                    parent_size=int(parent.size),
                )
                multiplexed[0, :] = parent_indices + parent_offset
                raw_parts.append(multiplexed)
                self._recording_id_conventions[parent.label] = convention
            else:
                self._recording_id_conventions[parent.label] = "no_spikes"
            parent_offset += int(parent.size)

        if not raw_parts:
            return np.empty((2, 0), dtype=float)

        raw = np.concatenate(raw_parts, axis=1)
        # Source IDs have been normalized to zero-based indices in the combined
        # E/I parent-population space above.
        raw_ids = raw[0].astype(int)
        total_neurons = int(self._params["N_E"] + self._params["N_I"])
        if raw_ids.size and (raw_ids.min() < 0 or raw_ids.max() >= total_neurons):
            raise RuntimeError(
                "Recorded source IDs do not match the expected 1-based parent-population IDs"
            )
        logical_from_parent = np.empty(total_neurons, dtype=int)
        logical_offset = 0
        parent_offset = 0
        for type_membership in self._cluster_membership:
            for indices in type_membership:
                for within_cluster, parent_index in enumerate(indices):
                    logical_from_parent[parent_offset + int(parent_index)] = (
                        logical_offset + within_cluster
                    )
                logical_offset += len(indices)
            parent_offset += sum(len(indices) for indices in type_membership)

        times = raw[1].astype(float)
        logical_ids = logical_from_parent[raw_ids].astype(float)
        keep = times >= float(self._params["warmup"])
        result = np.vstack(
            (times[keep] - float(self._params["warmup"]), logical_ids[keep])
        )
        return result[:, np.argsort(result[0], kind="stable")]

    @staticmethod
    def _parent_indices_from_source_ids(source_ids, *, parent_offset, parent_size):
        """Normalize backend source IDs to zero-based indices within a parent.

        Different PyNN/Neo combinations expose either local or global cell IDs,
        with zero- or one-based numbering.  The separate-population notebook
        used local one-based IDs, whereas parent populations can expose global
        IDs.  Candidate ranges make the distinction explicit and verifiable.
        """
        ids_float = np.asarray(source_ids)
        ids = ids_float.astype(np.int64)
        if not np.all(ids_float == ids):
            raise RuntimeError("Recorded source IDs are not integral")

        candidates = []
        if parent_offset:
            candidates.extend(
                [
                    ("global_one_based", ids - parent_offset - 1),
                    ("global_zero_based", ids - parent_offset),
                ]
            )
        candidates.extend(
            [
                ("local_one_based", ids - 1),
                ("local_zero_based", ids),
            ]
        )
        for convention, parent_indices in candidates:
            if np.all((parent_indices >= 0) & (parent_indices < parent_size)):
                return parent_indices, convention

        observed_min = int(ids.min()) if ids.size else None
        observed_max = int(ids.max()) if ids.size else None
        raise RuntimeError(
            "Recorded source IDs do not match a supported local/global, "
            "zero/one-based convention: "
            f"observed=[{observed_min}, {observed_max}], "
            f"parent_offset={parent_offset}, parent_size={parent_size}"
        )
