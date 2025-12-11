# -*- coding: utf-8 -*-
#
# network.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""PyNEST EI-clustered network: Network Class
---------------------------------------------

``ClusteredNetwork`` class with functions to build and simulate
the EI-clustered network.
"""

import pickle

import helper
import nest
import numpy as np


class ClusteredNetwork:
    """EI-clustered network objeect to build and simulate the network.

    Provides functions to create neuron populations,
    stimulation devices and recording devices for an
    EI-clustered network and setups the simulation in
    NEST (v3.x).

    Attributes
    ----------
    _params: dict
        Dictionary with parameters used to construct network.
    _populations: list
        List of neuron population groups.
    _recording_devices: list
        List of recording devices.
    _currentsources: list
        List of current sources.
    _model_build_pipeline: list
        List of functions to build the network.
    """

    def __init__(self, sim_dict, net_dict, stim_dict):
        """Initialize the ClusteredNetwork object.

        Parameters are given and explained in the files network_params_EI.py,
        sim_params_EI.py and stimulus_params_EI.py.

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

    def setup_nest(self):
        """Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual
        used one if none is supplied.
        """

        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        nest.local_num_threads = self._params.get("n_vp", 4)
        nest.resolution = self._params.get("dt")
        self._params["randseed"] = self._params.get("randseed")
        nest.rng_seed = self._params.get("randseed")

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

        E_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_E"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_E"],
            "V_reset": self._params["V_r"],
            "I_e": (
                I_xE
                if self._params["delta_I_xE"] == 0
                else I_xE * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2)
            ),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
            "V_m": (
                self._params["V_m"]
                if not self._params["V_m"] == "rand"
                else self._params["V_th_E"] - 20 * nest.random.lognormal(0, 1)
            ),
        }
        I_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_I"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_I"],
            "V_reset": self._params["V_r"],
            "I_e": (
                I_xI
                if self._params["delta_I_xE"] == 0
                else I_xI * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2)
            ),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
            "V_m": (
                self._params["V_m"]
                if not self._params["V_m"] == "rand"
                else self._params["V_th_I"] - 20 * nest.random.lognormal(0, 1)
            ),
        }

        # iaf_psc_exp allows stochasticity, if not used - don't supply the parameters and use
        # iaf_psc_exp as deterministic model
        if (self._params.get("delta") is not None) and (self._params.get("rho") is not None):
            E_neuron_params["delta"] = self._params["delta"]
            I_neuron_params["delta"] = self._params["delta"]
            E_neuron_params["rho"] = self._params["rho"]
            I_neuron_params["rho"] = self._params["rho"]

        # create the neuron populations
        pop_size_E = self._params["N_E"] // self._params["n_clusters"]
        pop_size_I = self._params["N_I"] // self._params["n_clusters"]
        E_pops = [
            nest.Create(self._params["neuron_type"], n=pop_size_E, params=E_neuron_params)
            for _ in range(self._params["n_clusters"])
        ]
        I_pops = [
            nest.Create(self._params["neuron_type"], n=pop_size_I, params=I_neuron_params)
            for _ in range(self._params["n_clusters"])
        ]

        self._populations = [E_pops, I_pops]

    def connect(self):
        """Connect the EI populations using the kappa-controlled clustering."""

        N_E = self._params["N_E"]
        N_I = self._params["N_I"]
        N_total = N_E + N_I

        js = self._params.get("js")
        if js is None or np.isnan(js).any():
            js = helper.calculate_RBN_weights(self._params)
        js = js * self._params["s"] / np.sqrt(N_total)

        baseline = self._params["baseline_conn_prob"]
        r_eplus = self._params["rep"]
        r_iplus = 1.0 + (r_eplus - 1.0) * self._params["rj"]

        prob_in_e, prob_out_e, weight_in_e, weight_out_e = self._mix_scales(r_eplus)
        prob_in_i, prob_out_i, weight_in_i, weight_out_i = self._mix_scales(r_iplus)

        p_plus = np.ones((2, 2))
        p_minus = np.ones((2, 2))
        weight_plus = np.ones((2, 2))
        weight_minus = np.ones((2, 2))

        # Build lookup tables for in-cluster and out-of-cluster probabilities/weights for each
        # combination of source/target types so that the double loop below can simply index them.
        for tgt in range(2):
            for src in range(2):
                if tgt == 0 and src == 0:
                    pin, pout, win, wout = (prob_in_e, prob_out_e, weight_in_e, weight_out_e)
                else:
                    pin, pout, win, wout = (prob_in_i, prob_out_i, weight_in_i, weight_out_i)
                p_plus[tgt, src] = baseline[tgt, src] * pin
                p_minus[tgt, src] = baseline[tgt, src] * pout
                weight_plus[tgt, src] = win
                weight_minus[tgt, src] = wout

        # Each tuple describes (synapse label, source population index, target population index).
        connection_defs = [
            ("EE", 0, 0),
            ("EI", 1, 0),
            ("IE", 0, 1),
            ("II", 1, 1),
        ]

        for label, src_type, tgt_type in connection_defs:
            base_weight = js[tgt_type, src_type]
            syn_plus = f"{label}_plus"
            syn_minus = f"{label}_minus"
            nest.CopyModel(
                "static_synapse",
                syn_plus,
                {"weight": base_weight * weight_plus[tgt_type, src_type], "delay": self._params["delay"]},
            )
            nest.CopyModel(
                "static_synapse",
                syn_minus,
                {"weight": base_weight * weight_minus[tgt_type, src_type], "delay": self._params["delay"]},
            )

            plus_params, plus_iterations = self._build_connection_params(p_plus[tgt_type, src_type], src_type)
            minus_params, minus_iterations = self._build_connection_params(p_minus[tgt_type, src_type], src_type)

            for tgt_cluster, post in enumerate(self._populations[tgt_type]):
                for src_cluster, pre in enumerate(self._populations[src_type]):
                    if tgt_cluster == src_cluster:
                        if plus_iterations == 0:
                            continue
                        for _ in range(plus_iterations):
                            nest.Connect(pre, post, plus_params, syn_plus)
                    else:
                        if minus_iterations == 0:
                            continue
                        for _ in range(minus_iterations):
                            nest.Connect(pre, post, minus_params, syn_minus)

    def _build_connection_params(self, probability, source_type):
        """Build connection parameters for the requested rule."""

        if probability <= 0:
            return None, 0

        rule = self._params.get("connection_rule", "pairwise_bernoulli")
        rule = str(rule).lower().replace("-", "_")

        if rule == "pairwise_poisson":
            conn_params = {
                "rule": "pairwise_poisson",
                "pairwise_avg_num_conns": probability,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            return conn_params, 1

        if rule == "fixed_indegree":
            cluster_size = (
                self._params["N_E"] // self._params["n_clusters"]
                if source_type == 0
                else self._params["N_I"] // self._params["n_clusters"]
            )
            indegree = int(probability * cluster_size)
            if indegree <= 0:
                return None, 0
            conn_params = {
                "rule": "fixed_indegree",
                "indegree": indegree,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            return conn_params, 1

        if rule == "pairwise_bernoulli":
            iterations = 1 if probability <= 1.0 else int(np.ceil(probability))
            effective_prob = probability / iterations
            conn_params = {
                "rule": "pairwise_bernoulli",
                "p": effective_prob,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            return conn_params, iterations

        raise ValueError(f"Unknown connection rule '{self._params.get('connection_rule')}'")

    def _mix_scales(self, r_plus):
        """Return probability and weight scaling factors for the given cluster ratio."""

        n_clusters = self._params["n_clusters"]
        if n_clusters <= 1 or r_plus == 1.0:
            return 1.0, 1.0, 1.0, 1.0

        kappa = float(self._params["kappa"])
        prob_in = r_plus ** (1.0 - kappa)
        prob_out = (n_clusters - prob_in) / float(n_clusters - 1)
        weight_in = r_plus ** kappa
        weight_out = (n_clusters - weight_in) / float(n_clusters - 1)
        return prob_in, prob_out, weight_in, weight_out

    @staticmethod
    def _determine_kappa(kappa_value, clustering_value):
        """Resolve the effective kappa parameter from legacy inputs."""

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
            self._currentsources = [nest.Create("step_current_generator")]
            for stim_cluster in self._params["stim_clusters"]:
                nest.Connect(self._currentsources[0], self._populations[0][stim_cluster])
            nest.SetStatus(
                self._currentsources[0],
                {
                    "amplitude_times": amplitude_times,
                    "amplitude_values": amplitude_values,
                },
            )

    def create_recording_devices(self):
        """Creates a spike recorder

        Create and connect a spike recorder to all neuron populations
        in self._populations.
        """
        self._recording_devices = [nest.Create("spike_recorder")]
        self._recording_devices[0].record_to = "memory"

        all_units = self._populations[0][0]
        for E_pop in self._populations[0][1:]:
            all_units += E_pop
        for I_pop in self._populations[1]:
            all_units += I_pop
        nest.Connect(all_units, self._recording_devices[0], "all_to_all")  # Spikerecorder

    def set_model_build_pipeline(self, pipeline):
        """Set _model_build_pipeline

        Parameters
        ----------
        pipeline: list
            ordered list of functions executed to build the network model
        """
        self._model_build_pipeline = pipeline

    def setup_network(self):
        """Setup network in NEST

        Initializes NEST and creates
        the network in NEST, ready to be simulated.
        Functions saved in _model_build_pipeline are executed.
        """
        for func in self._model_build_pipeline:
            func()

    def simulate(self):
        """Simulates network for a period of warmup+simtime"""
        nest.Simulate(self._params["warmup"] + self._params["simtime"])

    def get_recordings(self):
        """Extract spikes from Spikerecorder

        Extract spikes form the Spikerecorder connected
        to all populations created in create_populations.
        Cuts the warmup period away and sets time relative to end of warmup.
        Ids 1:N_E correspond to excitatory neurons,
        N_E+1:N_E+N_I correspond to inhibitory neurons.

        Returns
        -------
        spiketimes: ndarray
            2D array [2xN_Spikes]
            of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
        """
        events = nest.GetStatus(self._recording_devices[0], "events")[0]
        # convert them to the format accepted by spiketools
        spiketimes = np.append(events["times"][None, :], events["senders"][None, :], axis=0)
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
            of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
        """
        self.setup_network()
        self.simulate()
        return self.get_recordings()

    def get_firing_rates(self, spiketimes=None):
        """Calculates the average firing rates of
        all excitatory and inhibitory neurons.

        Calculates the firing rates of all excitatory neurons
        and the firing rates of all inhibitory neurons
        created by self.create_populations.
        If spiketimes are not supplied, they get extracted.

        Parameters
        ----------
        spiketimes: ndarray
            2D array [2xN_Spikes] of spiketimes
            with spiketimes in row 0 and neuron IDs in row 1.

        Returns
        -------
        tuple[float, float]
            average firing rates of excitatory (0)
            and inhibitory (1) neurons (spikes/s)
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        e_count = spiketimes[:, spiketimes[1] < self._params["N_E"]].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= self._params["N_E"]].shape[1]
        e_rate = e_count / float(self._params["N_E"]) / float(self._params["simtime"]) * 1000.0
        i_rate = i_count / float(self._params["N_I"]) / float(self._params["simtime"]) * 1000.0
        return e_rate, i_rate

    def set_I_x(self, I_XE, I_XI):
        """Set DC currents for excitatory and inhibitory neurons
        Adds DC currents for the excitatory and inhibitory neurons.
        The DC currents are added to the currents already
        present in the populations.

        Parameters
        ----------
        I_XE: float
            extra DC current for excitatory neurons [pA]
        I_XI: float
            extra DC current for inhibitory neurons [pA]
        """
        for E_pop in self._populations[0]:
            I_e_loc = E_pop.get("I_e")
            E_pop.set({"I_e": I_e_loc + I_XE})
        for I_pop in self._populations[1]:
            I_e_loc = I_pop.get("I_e")
            I_pop.set({"I_e": I_e_loc + I_XI})

    def get_simulation(self, PathSpikes=None):
        """Create network, simulate and return results

        Creates the EI-clustered network and simulates it with
        the parameters supplied in the object creation.
        Returns a dictionary with firing rates,
        timing information (dict) and parameters (dict).
        If PathSpikes is supplied the spikes get saved to a pickle file.

        Parameters
        ----------
        PathSpikes: str (optional)
            Path of file for spiketimes, if None, no file is saved

        Returns
        -------
        dict
         Dictionary with firing rates,
         spiketimes (ndarray) and parameters (dict)
        """

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
