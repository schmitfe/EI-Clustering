import pickle
import warnings

import helper

import pynn_brainscales.brainscales2 as sim
from pyNN.random import NumpyRNG
import numpy as np


class ClusteredNetwork:
    """EI-clustered network object to build and simulate the network.

    Provides functions to create neuron populations, stimulation devices and
    recording devices for an EI-clustered network and sets up the simulation on
    BrainScaleS-2 via the PyNN interface.

    Compared to the original BrainScaleS-2 version, this implementation also
    supports:
      * legacy clustering='weight'  -> kappa = 1
      * legacy clustering='probabilities' -> kappa = 0
      * direct numeric clustering values as shorthand for kappa
      * explicit kappa in [0, 1] to interpolate between probability and
        weight clustering.

    BrainScaleS-2-specific constraints are handled conservatively:
      * connection probabilities larger than 1 are approximated by several
        independent Projection objects with probability <= 1 each.
      * pairwise_poisson is approximated by the same repeated-Bernoulli scheme,
        because the BrainScaleS-2 PyNN API exposes FixedProbabilityConnector but
        not a dedicated pairwise_poisson connector.
      * synaptic weights are globally rescaled to the documented 6-bit hardware
        range [-63, 63].
    """

    def __init__(self, sim_dict, net_dict, stim_dict):
        """Initialize the ClusteredNetwork object.

        Parameters are given and explained in the files network_params.py,
        sim_params.py and stimulus_params.py.

        Parameters
        ----------
        sim_dict: dict
            Dictionary with simulation parameters.
        net_dict: dict
            Dictionary with network parameters.
        stim_dict: dict
            Dictionary with stimulus parameters.
        """

        # merge dictionaries of simulation, network and stimulus parameters
        self._params = {**sim_dict, **net_dict, **stim_dict}
        # list of neuron population groups [E_pops, I_pops]
        self._populations = []
        self._recording_devices = []
        self._currentsources = []
        self._projections = []

        self._model_build_pipeline = [
            self.setup_nest,
            self.create_populations,
            self.create_stimulation,
            self.create_recording_devices,
            self.connect,
        ]

        self._params["kappa"] = self._determine_kappa(
            self._params.get("kappa"),
            self._params.get("clustering"),
        )
        self._params["connection_rule"] = self._determine_connection_rule(
            self._params.get("connection_rule"),
            self._params.get("fixed_indegree"),
        )

        self._connector_seed_rng = np.random.default_rng(self._params.get("randseed"))
        self._warned_pairwise_poisson = False

    def setup_nest(self):
        """Initializes the BrainScaleS-2 PyNN backend."""
        setup_kwargs = {}
        if self._params.get("initial_config") is not None:
            setup_kwargs["initial_config"] = self._params["initial_config"]
        if self._params.get("connection") is not None:
            setup_kwargs["connection"] = self._params["connection"]
        sim.setup(**setup_kwargs)

    def create_populations(self):
        """Create all neuron populations.

        n_clusters excitatory and inhibitory neuron populations
        with the parameters of the network are created.
        """

        # make sure number of clusters and units are compatible
        if self._params["N_E"] % self._params["n_clusters"] != 0:
            raise ValueError("N_E must be a multiple of Q")
        if self._params["N_I"] % self._params["n_clusters"] != 0:
            raise ValueError("N_E must be a multiple of Q")
        if self._params["neuron_type"] != "iaf_psc_exp":
            raise ValueError("Model only implemented for iaf_psc_exp neuron model")

        if self._params["I_th_E"] is None:
            I_xE = self._params["I_xE"]  # I_xE is the feed forward excitatory input in pA
        else:
            I_xE = self._params["I_th_E"] * helper.rheobase_current(
                self._params["tau_E"], self._params["E_L"], self._params["V_th_E"], self._params["C_m"]
            )

        if self._params["I_th_I"] is None:
            I_xI = self._params["I_xI"]
        else:
            I_xI = self._params["I_th_I"] * helper.rheobase_current(
                self._params["tau_I"], self._params["E_L"], self._params["V_th_I"], self._params["C_m"]
            )

        offset = 100

        E_neuron_params = {
            "v_rest": self._params["E_L"] + offset + self._params["I_th_E"] * (self._params["V_th_E"] - self._params["E_L"]),
            "cm": 63,  # self._params["C_m"]/1000.,
            "tau_m": self._params["tau_E"],
            "tau_refrac": self._params["t_ref"],
            "v_thresh": self._params["V_th_E"] + offset,
            "v_reset": self._params["V_r"] + offset,
            "tau_syn_E": self._params["tau_syn_ex"],
            "tau_syn_I": self._params["tau_syn_in"],
        }
        I_neuron_params = {
            "v_rest": self._params["E_L"] + offset + self._params["I_th_I"] * (self._params["V_th_E"] - self._params["E_L"]),
            "cm": 63,  # self._params["C_m"]/1000.,
            "tau_m": self._params["tau_I"],
            "tau_refrac": self._params["t_ref"],
            "v_thresh": self._params["V_th_I"] + offset,
            "v_reset": self._params["V_r"] + offset,
            "tau_syn_E": self._params["tau_syn_ex"],
            "tau_syn_I": self._params["tau_syn_in"],
        }

        # iaf_psc_exp allows stochasticity, if not used - don't supply the
        # parameters and use iaf_psc_exp as deterministic model
        if (self._params.get("delta") is not None) and (self._params.get("rho") is not None):
            E_neuron_params["delta"] = self._params["delta"]
            I_neuron_params["delta"] = self._params["delta"]
            E_neuron_params["rho"] = self._params["rho"]
            I_neuron_params["rho"] = self._params["rho"]

        # create the neuron populations
        pop_size_E = self._params["N_E"] // self._params["n_clusters"]
        pop_size_I = self._params["N_I"] // self._params["n_clusters"]
        E_pops = [
            sim.Population(pop_size_E, sim.cells.CalibHXNeuronCuba(**E_neuron_params), label="E" + str(i))
            for i in range(self._params["n_clusters"])
        ]
        I_pops = [
            sim.Population(pop_size_I, sim.cells.CalibHXNeuronCuba(**I_neuron_params), label="I" + str(i))
            for i in range(self._params["n_clusters"])
        ]

        self._populations = [E_pops, I_pops]
        bias = int(self._params.get("synin_bias", 400))
        sim.run(None, sim.RunCommand.PREPARE)
        for pop in sim.simulator.state.populations:
            for neuron in pop.actual_hwparams:
                neuron.excitatory_input.i_bias_gm = bias
                neuron.inhibitory_input.i_bias_gm = bias

        synapse_dac_bias = int(self._params.get("synapse_dac_bias", 1022))
        for synapse_block in sim.simulator.state.grenade_chip_config.synapse_blocks:
            for dac_idx, _ in enumerate(synapse_block.i_bias_dac):
                synapse_block.i_bias_dac[dac_idx] = synapse_dac_bias

    def connect(self):
        """Connect the EI populations using the kappa-controlled clustering."""
        self._connect_clustered(self._params["kappa"])

    def connect_weight(self):
        """Backward-compatible helper for pure weight clustering."""
        self._connect_clustered(1.0)

    def connect_probabilities(self):
        """Backward-compatible helper for pure probability clustering."""
        self._connect_clustered(0.0)

    def _connect_clustered(self, kappa):
        """Connect the clusters with kappa-controlled clustering.

        kappa = 1 reproduces pure weight clustering.
        kappa = 0 reproduces pure probability clustering.
        Intermediate values interpolate multiplicatively between both schemes.
        """

        self._projections = []

        N_E = self._params["N_E"]
        N_I = self._params["N_I"]
        N_total = N_E + N_I
        n_clusters = self._params["n_clusters"]

        js = self._params.get("js")
        if js is None:
            js = helper.calculate_RBN_weights(self._params)
        else:
            js = np.asarray(js, dtype=float)
            if np.isnan(js).any():
                js = helper.calculate_RBN_weights(self._params)
        js = np.asarray(js, dtype=float) * self._params["s"] / np.sqrt(N_total)

        baseline = np.asarray(self._params["baseline_conn_prob"], dtype=float)

        r_eplus = float(self._params["rep"])
        r_iplus = 1.0 + (r_eplus - 1.0) * float(self._params["rj"])

        p_plus = np.ones((2, 2), dtype=float)
        p_minus = np.ones((2, 2), dtype=float)
        weight_plus = np.ones((2, 2), dtype=float)
        weight_minus = np.ones((2, 2), dtype=float)

        for tgt_type in range(2):
            for src_type in range(2):
                if tgt_type == 0 and src_type == 0:
                    pin, pout, win, wout = self._mix_scales(r_eplus, kappa)
                else:
                    pin, pout, win, wout = self._mix_scales(r_iplus, kappa)
                p_plus[tgt_type, src_type] = baseline[tgt_type, src_type] * pin
                p_minus[tgt_type, src_type] = baseline[tgt_type, src_type] * pout
                weight_plus[tgt_type, src_type] = win
                weight_minus[tgt_type, src_type] = wout

        self._params["effective_pplus"] = p_plus.copy()
        self._params["effective_pminus"] = p_minus.copy()
        self._params["effective_weight_plus"] = weight_plus.copy()
        self._params["effective_weight_minus"] = weight_minus.copy()

        self._warn_brainscales_constraints(p_plus, p_minus)

        raw_signed_weights = {
            ("EE", "plus"): js[0, 0] * weight_plus[0, 0],
            ("EE", "minus"): js[0, 0] * weight_minus[0, 0],
            ("EI", "plus"): js[0, 1] * weight_plus[0, 1],
            ("EI", "minus"): js[0, 1] * weight_minus[0, 1],
            ("IE", "plus"): js[1, 0] * weight_plus[1, 0],
            ("IE", "minus"): js[1, 0] * weight_minus[1, 0],
            ("II", "plus"): js[1, 1] * weight_plus[1, 1],
            ("II", "minus"): js[1, 1] * weight_minus[1, 1],
        }
        hw_signed_weights = self._scale_weights_for_hardware(
            raw_signed_weights,
            max_hw_weight=self._params.get("max_hw_weight", 63),
        )
        self._params["hw_scaled_weights"] = dict(hw_signed_weights)

        connection_defs = [
            ("EE", 0, 0, "excitatory"),
            ("EI", 1, 0, "inhibitory"),
            ("IE", 0, 1, "excitatory"),
            ("II", 1, 1, "inhibitory"),
        ]

        for label, src_type, tgt_type, receptor_type in connection_defs:
            same_synapse = sim.standardmodels.synapses.StaticSynapse(weight=hw_signed_weights[(label, "plus")])
            diff_synapse = sim.standardmodels.synapses.StaticSynapse(weight=hw_signed_weights[(label, "minus")])

            for src_cluster, pre in enumerate(self._populations[src_type]):
                for tgt_cluster, post in enumerate(self._populations[tgt_type]):
                    same_cluster = src_cluster == tgt_cluster
                    connection_density = p_plus[tgt_type, src_type] if same_cluster else p_minus[tgt_type, src_type]
                    synapse = same_synapse if same_cluster else diff_synapse
                    if connection_density <= 0:
                        continue

                    autapses_forbidden = same_cluster and src_type == tgt_type
                    connectors = self._build_connectors(
                        connection_density=connection_density,
                        source_population_size=pre.size,
                        allow_self_connections=not autapses_forbidden,
                    )

                    for projection_idx, connector in enumerate(connectors):
                        label_suffix = "plus" if same_cluster else "minus"
                        projection = sim.Projection(
                            pre,
                            post,
                            connector,
                            synapse,
                            receptor_type=receptor_type,
                            label=f"{label}_{label_suffix}_{src_cluster}_{tgt_cluster}_{projection_idx}",
                        )
                        self._projections.append(projection)

    def _build_connectors(self, connection_density, source_population_size, allow_self_connections):
        """Build BrainScaleS-2 connectors for the configured rule.

        BrainScaleS-2 currently exposes probability and fixed-number connectors
        through the PyNN API. pairwise_poisson is approximated by repeated
        FixedProbabilityConnector passes.
        """

        if connection_density <= 0:
            return []

        rule = self._params.get("connection_rule", "pairwise_bernoulli")
        rule = str(rule).lower().replace("-", "_")
        connectors = []

        if rule == "fixed_indegree":
            indegree = int(connection_density * source_population_size)
            if indegree <= 0:
                return []

            # FixedNumberPreConnector is the PyNN connector with fixed indegree.
            # When indegree exceeds the number of distinct available sources, we
            # allow sampling with replacement to approximate multapses.
            max_unique_sources = source_population_size if allow_self_connections else max(source_population_size - 1, 0)
            with_replacement = indegree > max_unique_sources
            connectors.append(
                sim.FixedNumberPreConnector(
                    indegree,
                    allow_self_connections=allow_self_connections,
                    with_replacement=with_replacement,
                    rng=self._new_connector_rng(),
                )
            )
            return connectors

        if rule == "pairwise_poisson" and not self._warned_pairwise_poisson:
            warnings.warn(
                "connection_rule='pairwise_poisson' is approximated on BrainScaleS-2 "
                "by repeated FixedProbabilityConnector passes. This preserves the "
                "mean connection multiplicity but not the exact Poisson statistics.",
                RuntimeWarning,
            )
            self._warned_pairwise_poisson = True

        # Default and fallback: repeated Bernoulli passes.
        iterations = max(1, int(np.ceil(connection_density)))
        effective_probability = float(connection_density) / iterations
        effective_probability = min(max(effective_probability, 0.0), 1.0)

        for _ in range(iterations):
            connectors.append(
                sim.FixedProbabilityConnector(
                    effective_probability,
                    allow_self_connections=allow_self_connections,
                    rng=self._new_connector_rng(),
                )
            )
        return connectors

    def _new_connector_rng(self):
        """Create a PyNN RNG for connector sampling."""
        seed = int(self._connector_seed_rng.integers(1, np.iinfo(np.int32).max))
        return NumpyRNG(seed=seed)

    def _mix_scales(self, r_plus, kappa):
        """Return probability and weight scaling factors for the given ratio."""

        n_clusters = self._params["n_clusters"]
        if n_clusters <= 1 or r_plus == 1.0:
            return 1.0, 1.0, 1.0, 1.0

        kappa = float(kappa)
        prob_in = float(r_plus) ** (1.0 - kappa)
        prob_out = ((n_clusters - r_plus) / float(n_clusters - 1))**(1.0 - kappa)
        weight_in = float(r_plus) ** kappa
        weight_out = ((n_clusters - r_plus) / float(n_clusters - 1))**(kappa)
        return prob_in, prob_out, weight_in, weight_out

    def _scale_weights_for_hardware(self, raw_signed_weights, max_hw_weight=63):
        """Scale signed weights to the BrainScaleS-2 hardware range.

        The BrainScaleS-2 examples and documentation use a 6-bit synaptic weight
        range with magnitude up to 63. To keep the existing BrainScaleS-2 port
        behaviour stable, we preserve relative weight ratios and apply a single
        global scaling factor across all pathways.
        """

        max_hw_weight = float(max_hw_weight)
        if max_hw_weight <= 0:
            raise ValueError("max_hw_weight must be positive")

        values = np.asarray(list(raw_signed_weights.values()), dtype=float)
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs == 0.0:
            return {key: 0 for key in raw_signed_weights}

        scale = max_hw_weight / max_abs
        scaled = {}
        for key, value in raw_signed_weights.items():
            hw_value = int(np.round(float(value) * scale))
            hw_value = int(np.clip(hw_value, -max_hw_weight, max_hw_weight))
            scaled[key] = hw_value
        return scaled

    def _warn_brainscales_constraints(self, p_plus, p_minus):
        """Emit warnings for obvious single-chip BrainScaleS-2 constraint violations."""

        total_neurons = int(self._params["N_E"]) + int(self._params["N_I"])
        if total_neurons > 512:
            warnings.warn(
                "The configured network has more than 512 neurons. A single BrainScaleS-2 "
                "chip exposes 512 neuron compartments, so this setup is unlikely to fit on "
                "hardware without additional partitioning.",
                RuntimeWarning,
            )

        cluster_sizes = [
            int(self._params["N_E"]) // int(self._params["n_clusters"]),
            int(self._params["N_I"]) // int(self._params["n_clusters"]),
        ]
        q = int(self._params["n_clusters"])
        expected_synapses = 0.0

        for tgt_type in range(2):
            for src_type in range(2):
                n_post = cluster_sizes[tgt_type]
                n_pre = cluster_sizes[src_type]

                same_cluster_pairs = q * n_post * n_pre
                if src_type == tgt_type:
                    same_cluster_pairs -= q * min(n_post, n_pre)  # crude autapse correction
                diff_cluster_pairs = q * max(q - 1, 0) * n_post * n_pre

                expected_synapses += same_cluster_pairs * p_plus[tgt_type, src_type]
                expected_synapses += diff_cluster_pairs * p_minus[tgt_type, src_type]

        if expected_synapses > 131072:
            warnings.warn(
                "The expected number of synapses exceeds 131072, i.e. the synapse count of "
                "one BrainScaleS-2 HICANN-X chip. This setup is likely to exceed routing or "
                "placement limits on hardware.",
                RuntimeWarning,
            )

        max_density = max(float(np.max(p_plus)), float(np.max(p_minus)))
        if max_density > 1.0:
            warnings.warn(
                "At least one effective connection probability exceeds 1. The BrainScaleS-2 "
                "backend approximates this using multiple Projection objects. This increases "
                "routing pressure and may fail on hardware for dense networks.",
                RuntimeWarning,
            )

    @staticmethod
    def _determine_kappa(kappa_value, clustering_value):
        """Resolve the effective kappa parameter from the configured inputs."""

        if kappa_value is not None:
            return float(kappa_value)
        if clustering_value is None:
            return 1.0

        if isinstance(clustering_value, str):
            option = clustering_value.lower()
            if option == "weight":
                return 1.0
            if option == "probabilities":
                return 0.0
            try:
                return float(clustering_value)
            except ValueError as exc:
                raise ValueError("Clustering must be 'weight', 'probabilities' or a numeric value.") from exc

        try:
            return float(clustering_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Clustering must be 'weight', 'probabilities' or a numeric value.") from exc

    @staticmethod
    def _determine_connection_rule(connection_rule, fixed_indegree):
        """Normalize connection rule handling across old and new parameter sets."""

        if connection_rule is not None:
            rule = str(connection_rule).lower().replace("-", "_")
            if rule in {"pairwise_bernoulli", "pairwise_poisson", "fixed_indegree"}:
                return rule
            raise ValueError("connection_rule must be 'pairwise_bernoulli', 'pairwise_poisson' or 'fixed_indegree'.")

        return "fixed_indegree" if fixed_indegree else "pairwise_bernoulli"

    def create_stimulation(self):
        """Create a current source and connect it to clusters."""
        if self._params["stim_clusters"] is not None:
            stim_amp = self._params["stim_amp"]  # amplitude of the stimulation current in pA
            stim_starts = self._params["stim_starts"]  # list of stimulation start times
            stim_ends = self._params["stim_ends"]  # list of stimulation end times
            amplitude_values = []
            amplitude_times = []
            for start, end in zip(stim_starts, stim_ends):
                amplitude_times.append(start + self._params["warmup"])
                amplitude_values.append(stim_amp)
                amplitude_times.append(end + self._params["warmup"])
                amplitude_values.append(0.0)
            self._currentsources = [sim.StepCurrentSource(times=amplitude_times, amplitudes=amplitude_values)]
            for stim_cluster in self._params["stim_clusters"]:
                self._populations[0][stim_cluster].inject(self._currentsources[0])

    def create_recording_devices(self):
        """Create spike recording on all populations."""
        for pops in self._populations:
            for pop in pops:
                pop.record("spikes")

    def set_model_build_pipeline(self, pipeline):
        """Set _model_build_pipeline.

        Parameters
        ----------
        pipeline: list
            ordered list of functions executed to build the network model
        """
        self._model_build_pipeline = pipeline

    def setup_network(self):
        """Setup network in BrainScaleS-2 PyNN."""
        for func in self._model_build_pipeline:
            func()

    def simulate(self):
        """Simulate network for warmup + simtime."""
        sim.run(self._params["warmup"] + self._params["simtime"])

    def get_recordings(self):
        """Extract spikes from the recorded populations.

        Returns
        -------
        spiketimes: ndarray
            2D array [2xN_Spikes]
            with spiketimes in row 0 and neuron IDs in row 1.
        """
        previousMaxN = 0
        spiketimes = []
        for pops in self._populations:
            for pop in pops:
                data_pop = pop.get_data()
                data_pop = data_pop.segments[-1].spiketrains
                data_pop = np.array(data_pop.multiplexed)
                data_pop[0, :] += previousMaxN
                previousMaxN += pop.size
                spiketimes.append(data_pop.copy())
                del data_pop
        spiketimes = np.concatenate(spiketimes, axis=1)
        idx_sort = spiketimes[1, :].argsort()
        spiketimes = spiketimes[:, idx_sort]
        spiketimes[[0, 1]] = spiketimes[[1, 0]]
        spiketimes[1] -= 1
        # remove the pre warmup spikes
        spiketimes = spiketimes[:, spiketimes[0] >= self._params["warmup"]]
        spiketimes[0] -= self._params["warmup"]
        return spiketimes

    def get_parameter(self):
        """Get all parameters used to create the network.

        Returns
        -------
        dict
            Dictionary with all parameters of the network and the simulation.
        """
        return self._params

    def create_and_simulate(self):
        """Create and simulate the EI-clustered network.

        Returns
        -------
        spiketimes: ndarray
            2D array [2xN_Spikes]
            with spiketimes in row 0 and neuron IDs in row 1.
        """
        self.setup_network()
        self.simulate()
        return self.get_recordings()

    def get_firing_rates(self, spiketimes=None):
        """Calculate the average firing rates of all excitatory and inhibitory neurons."""

        if spiketimes is None:
            spiketimes = self.get_recordings()
        e_count = spiketimes[:, spiketimes[1] < self._params["N_E"]].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= self._params["N_E"]].shape[1]
        e_rate = e_count / float(self._params["N_E"]) / float(self._params["simtime"]) * 1000.0
        i_rate = i_count / float(self._params["N_I"]) / float(self._params["simtime"]) * 1000.0
        return e_rate, i_rate

    def set_I_x(self, I_XE, I_XI):
        """Set DC currents for excitatory and inhibitory neurons.

        Adds DC currents for the excitatory and inhibitory neurons.
        The DC currents are added to the currents already present in the populations.
        """
        for E_pop in self._populations[0]:
            I_e_loc = E_pop.get("I_e")
            E_pop.set({"I_e": I_e_loc + I_XE})
        for I_pop in self._populations[1]:
            I_e_loc = I_pop.get("I_e")
            I_pop.set({"I_e": I_e_loc + I_XI})

    def get_simulation(self, PathSpikes=None):
        """Create network, simulate and return results."""

        self.setup_network()
        self.simulate()
        spiketimes = self.get_recordings()
        e_rate, i_rate = self.get_firing_rates(spiketimes)

        if PathSpikes is not None:
            with open(PathSpikes, "wb") as outfile:
                pickle.dump(spiketimes, outfile)
        return {
            "e_rate": e_rate,
            "i_rate": i_rate,
            "_params": self.get_parameter(),
            "spiketimes": spiketimes,
        }