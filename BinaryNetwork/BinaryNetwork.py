# Created by Felix J. Schmitt on 05/30/2023.
# Class for binary networks (state is 0 or 1)

import numpy as np
import os

class NetworkElement:
    def __init__(self, reference, name="Some Network Element"):
        self.name = name
        self.reference = reference
        self.view = None
    def set_view(self, view):
        self.view = view
    def initialze(self):
        pass

class Neuron(NetworkElement):
    def __init__(self, reference, N=1, name="Some Neuron", tau=1.0):
        super().__init__(reference, name)
        self.N = N
        self.state = None
        self.tau = tau
    def update(self):
        pass
    def set_view(self, view):
        self.state = self.reference.state[view[0]:view[1]]
        self.view = view
    def initialze(self):
        self.reference.state[self.view[0]:self.view[1]] = self._initial_state()

class BinaryNeuronPopulation(Neuron):
    def __init__(self, reference, N=1, threshold=1.0, name="Binary Neuron Population", tau=1.0, initializer=None, **kwargs):
        super().__init__(reference, N, name, tau=tau)
        self.threshold = threshold
        self.initializer = initializer
    def update(self, weights=None, state=None, index=None, input_value=None, **kwargs):
        if input_value is None:
            if weights is None or state is None:
                raise ValueError("weights and state have to be provided when input_value is not set")
            input_value = np.sum(weights * state)
        return np.heaviside(input_value - self.threshold, 0)
    def _initial_state(self):
        if callable(self.initializer):
            values = np.asarray(self.initializer(self.N), dtype=np.int16)
            if values.size != self.N:
                raise ValueError("Initializer must return {} entries".format(self.N))
            return values
        if self.initializer is None:
            return np.random.choice([0, 1], size=self.N, p=[0.8, 0.2])
        arr = np.asarray(self.initializer)
        if arr.size == 1:
            return np.full(self.N, int(arr.item()), dtype=np.int16)
        if arr.size != self.N:
            raise ValueError("Initializer must define exactly {} elements".format(self.N))
        return arr.astype(np.int16)


class BackgroundActivity(Neuron):
    # Neuron which
    def __init__(self, reference, N=1, Activity=0.5, Stochastic=False, name="Background Activity", tau=1.0):
        super().__init__(reference, N, name, tau=tau)
        self.Activity = Activity
        if Stochastic:
            self.update = self.update_stochastic
        else:
            self.update = self.update_deterministic
    def update_stochastic(self, weights=None, state=None, Index=None, **kwargs):
        return np.random.choice([0, 1], 1) * self.update_deterministic(weights, state, **kwargs)
    def update_deterministic(self, weights=None, state=None, Index=None, **kwargs):
        # if activity is a float, set all neurons to this activity
        if isinstance(self.Activity, float):
            return self.Activity
        # if activity is a function, set neurons by this function
        elif callable(self.Activity):
            return self.Activity()
        else:
            return 1.0

    def initialze(self):
        self.state = np.array([self.update() for i in range(self.N)])

class Synapse(NetworkElement):
    def __init__(self, reference, pre, post, name="Some Synapse"):
        super().__init__(reference, name= post.name + " <- " + pre.name)
        self.pre = pre
        self.post = post
        # weights is a matrix of shape (pre.N, post.N)
        self.weights = None
    def set_view(self, view):

        self.weights = self.reference.weights[view[1,0]:view[1,1], view[0,0]:view[0,1]]
        self.view = view
    def initialze(self):
        self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] = np.random.rand(self.post.N, self.pre.N)

class PairwiseBernoulliSynapse(Synapse):
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post )
        self.p = p
        self.j = j
    def initialze(self):
        # if p is greater 1, split into two synapses
        p = self.p
        n_iterations = 1
        while p > 1:
            p /= 2
            n_iterations += 1
        if n_iterations > 1:
            print("Warning: p > 1, splitting synapse into " + str(n_iterations) + " synapses")

        for i in range(n_iterations):
            self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] += \
                np.random.choice([0, self.j], size=(self.post.N, self.pre.N), p=[1-p, p])

class PoissonSynapse(Synapse):
    def __init__(self, reference, pre, post, rate=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.rate = rate
        self.j = j
    def initialze(self):
        samples = np.random.poisson(lam=self.rate, size=(self.post.N, self.pre.N))
        self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] += samples * self.j

class FixedIndegreeSynapse(Synapse):
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post)
        self.p = p
        self.j = j
    def initialze(self):
        p = max(self.p, 0.0)
        target_count = int(round(p * self.pre.N))
        target_count = min(max(target_count, 0), self.pre.N)
        if target_count == 0:
            return
        for target in range(self.post.N):
            pres = np.random.choice(self.pre.N, size=target_count, replace=True)
            idx_rows = self.view[1, 0] + target
            idx_cols = self.view[0, 0] + pres
            self.reference.weights[idx_rows, idx_cols] += self.j

class AllToAllSynapse(Synapse):
    def __init__(self, reference, pre, post, j=1.0):
        super().__init__(reference, pre, post)
        self.j = j
    def initialze(self):
        self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] = \
            np.ones((self.post.N, self.pre.N)) * self.j


class BinaryNetwork:
    def __init__(self, name="Some Binary Network"):
        self.name = name
        self.N = 0
        self.population = []
        self.synapses = []
        self.state = None
        self.weights = None
        self.LUT = None # look up table for the update function
        self.sim_steps = 0
        self.population_lookup = None
        self.neuron_lookup = None

    def add_population(self, population):
        self.population.append(population)
        self.N += population.N
        return population

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def initialize(self, autapse=False):
        self.state = np.zeros(self.N, dtype=np.int16)
        self.weights = np.zeros((self.N, self.N))
        self.update_prob = np.zeros(self.N)
        self.population_lookup = np.zeros(self.N, dtype=np.int32)
        self.neuron_lookup = np.zeros(self.N, dtype=np.int32)
        N_start = 0
        for idx, population in enumerate(self.population):
            population.set_view([N_start,N_start + population.N])
            N_start += population.N
            population.initialze()
            # fill update_prob with the inverse tau
            self.update_prob[population.view[0]:population.view[1]] = 1.0/population.tau
            self.population_lookup[population.view[0]:population.view[1]] = idx
            self.neuron_lookup[population.view[0]:population.view[1]] = np.arange(population.N, dtype=np.int32)
        self.LUT= np.array([population.view for population in self.population])
        # normalize update_prob
        self.update_prob /= self.update_prob.sum()
        for synapse in self.synapses:
            synapse.set_view(np.array([[synapse.pre.view[0], synapse.pre.view[1]],[synapse.post.view[0], synapse.post.view[1]]]))
            synapse.initialze()
        self.sim_steps = 0
        # set diagonal to zero
        if not autapse:
            np.fill_diagonal(self.weights, 0)

    def update(self):
        # choose a random neuron according to update_prob and update it
        #neuron = np.random.randint(self.N)
        neuron = np.random.choice(self.N, p=self.update_prob)
        # find the population to which the neuron belongs
        population_idx = self.population_lookup[neuron]

        # find the index of the neuron in the population
        neuronIDX = self.neuron_lookup[neuron]
        # update the neuron
        if self.state[neuron] == 0:
            self.state[neuron] = self.population[population_idx].update(self.weights[neuron, :],
                                                                    self.state, neuronIDX)
        else:
            self.state[neuron] = 0
        self.sim_steps += 1

    def _update_batch(self, neurons):
        neurons = np.asarray(neurons, dtype=np.int64)
        batch_size = neurons.size
        if batch_size == 0:
            return
        potentials = self.weights[neurons, :].dot(self.state)
        for idx in range(batch_size):
            neuron = neurons[idx]
            population_idx = self.population_lookup[neuron]
            neuronIDX = self.neuron_lookup[neuron]
            old_state = self.state[neuron]
            if old_state == 0:
                new_state = self.population[population_idx].update(self.weights[neuron, :],
                                                                   self.state, neuronIDX,
                                                                   input_value=potentials[idx])
            else:
                new_state = 0
            if new_state != old_state:
                delta = new_state - old_state
                self.state[neuron] = new_state
                if idx + 1 < batch_size:
                    remaining = neurons[idx + 1:]
                    potentials[idx + 1:] += delta * self.weights[remaining, neuron]
            self.sim_steps += 1

    def run(self, steps=1000, batch_size=1):
        if batch_size <= 1:
            for i in range(steps):
                self.update()
            return
        steps_done = 0
        while steps_done < steps:
            current_batch = min(batch_size, steps - steps_done)
            neurons = np.random.choice(self.N, size=current_batch, p=self.update_prob)
            self._update_batch(neurons)
            steps_done += current_batch
