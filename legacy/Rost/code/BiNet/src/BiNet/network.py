
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True,
                  language_level=3)

try:  # tqdm is optional; skip progress bars if not available
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable

from . import cstuff
from .weights import generate_balanced_weights


STATE_TYPE = np.int8

class UpdateQueue(object):
    def __init__(self,Ns,n_updates,update_ratios=None,mode = 'sequential'):
        if update_ratios is None or np.isscalar(update_ratios):
            update_ratios = np.ones(len(Ns))
        for u in update_ratios:
            assert isinstance(u,int) or (u).is_integer(), 'update_ratios need to be whole numbers'
        self.Ns = Ns
        self.n_updates = n_updates
        self.update_ratios = update_ratios
        self.mode = mode
        # generate and update_pool in which indices are represented update_ratios time
        pool = []
        nbins = np.cumsum(np.concatenate([[0],Ns]))
        
        for i in range(len(Ns)):
            repeats = int(update_ratios[i])
            indices = list(range(nbins[i], nbins[i+1]))
            pool += indices * repeats
        self.pool = np.array(pool)
        np.random.shuffle(self.pool)
        self.queue = []


        #self.inds = range(len(self.pool))
        #np.random.shuffle(self.inds)

    def _refill_queue(self):
        if self.mode == 'random':
            self.queue = cstuff.sample_without_replacement(self.n_updates, len(self.pool), sum(self.Ns)).tolist()
        else :
            
            
            inds = np.arange(len(self.pool))
            if self.mode == 'sequential':
                np.random.shuffle(inds)
            while len(inds)>0:
                self.queue.append(inds[:self.n_updates])
                inds = inds[self.n_updates:]
        
    def next(self):
        
        try:
            inds = self.queue.pop()
        except:
            self._refill_queue()
                
            inds = self.queue.pop()
        return self.pool[inds]
        










class _BaseNetwork(object):
    def __init__(self,weights,T,input_weights=None):
        """ Base class for binary networks implementing only the integration loop. """
        self.weights =weights
        
        self.set_weights(weights)
        self.T = T
        self.N = len(T)
        self.state = np.zeros(self.N).astype(STATE_TYPE)
        self.old_state = np.zeros_like(self.state)
        self.new_states = None
        self.input_weights = input_weights
        self.current_updates = None
        self.reference_state = self.state.copy()
        self.last_update_log = None
        self.last_delta_log = None
        self.subthreshold_record = []
        self.update_record = []
        self.nonzero_states = None

    def _get_next_updates(self):
        """ dummy. needs to be overridden..."""
        return np.arange(self.N)
    def _plasticity(self,input):
        """ dummy. needs to be overridden..."""
        pass
    def set_weights(self,weights):
        self.weights =weights.copy()
       
    def get_weights(self):
        return self.weights.copy()

    def set_input_weights(self,input_weights):
        self.input_weights =input_weights
    def get_input_weights(self):
        return self.input_weights.copy()
    def initialise_state(self,prob):
        self.state = (np.random.rand(*self.state.shape)<prob).astype(STATE_TYPE)
        self.nonzero_states = cstuff.NonZeroCounter(self.state)
        self.reference_state = self.state.copy()
    def set_state(self,state):
        self.state = state.astype(STATE_TYPE)
        self.nonzero_states = cstuff.NonZeroCounter(self.state)
        self.reference_state = self.state.copy()
     
    def _update_old_state(self):
        # isolated in function for profiling
        
        self.old_state[self.current_updates] = self.state[self.current_updates]
    def _integration_step(self,input,strict_spiking,record_subthreshold,record_updates):
        # remember the old state. this is mainly useful for plasticity...
        self._update_old_state()
        
        self.current_updates = self._get_next_updates()
        
        
        

        if record_subthreshold:
            Nbins= [0] + np.cumsum(self.Ns).tolist()
            inputs =[]
            for i in range(len(self.Ns)):
                inputs.append(self.weights[:,Nbins[i]:Nbins[i+1]].dot(self.state[Nbins[i]:Nbins[i+1]]))
            inputs.append(self.input_weights.dot(input))
            self.subthreshold_record.append(inputs)
        if record_updates:
            self.update_record.append(self.current_updates)
        if self.new_states is None:
            self.new_states = np.zeros(len(self.current_updates),dtype = STATE_TYPE)
        cstuff.calc_new_states(self.new_states,self.current_updates,self.weights,self.state,self.input_weights,input,self.T,self.nonzero_states)
        
        
        

        self.state[self.current_updates] = self.new_states
        #print self.state.dtype
        #print self.state.dtype
        # call plasticity method, which may be empty
        self._plasticity(input)
        
        delta = self.state[self.current_updates].astype(np.int8, copy=True) - self.old_state[self.current_updates].astype(np.int8, copy=False)
        if strict_spiking:
            spikes = delta > 0
        else:
            spikes = self.state[self.current_updates]
        return spikes, delta
    
    def forward(
        self,
        input,
        return_state=False,
        strict_spiking=False,
        record_subthreshold=False,
        record_updates=False,
        return_spiketimes=False,
    ):
        """ pushes the input array through the network.
            if return state is False, only the newly updated
            spikes in each time step are returned.
            """
        if self.nonzero_states is None:
            self.set_state(self.state)
        assert self.input_weights is not None, 'input weights need to be set'
        assert self.input_weights.shape == (self.N,input.shape[0]), 'input weights must have shape (N,n_inputs)'

        self.update_record = []
        delta_log: list[np.ndarray] = []
        initial_state = self.state.astype(STATE_TYPE, copy=True)
        steps = input.shape[1]
        for i in tqdm(range(steps), desc="Legacy network updates", disable=steps < 1000):
            _, delta = self._integration_step(
                input[:, i],
                strict_spiking,
                record_subthreshold,
                True,
            )
            delta_log.append(delta.astype(np.int8, copy=True))
        if self.update_record:
            self.last_update_log = np.stack(
                [np.asarray(entry, dtype=np.uint16) for entry in self.update_record],
                axis=1,
            )
            self.last_delta_log = np.stack(delta_log, axis=1)
        else:
            self.last_update_log = np.zeros((0, 0), dtype=np.uint16)
            self.last_delta_log = np.zeros((0, 0), dtype=np.int8)
        print(f"Different events [-,0,+]: {np.unique(self.last_delta_log, return_counts=True)}")
        self.reference_state = initial_state
        state_output = None
        spike_array = None
        if return_state:
            state_output = (
                initial_state.astype(STATE_TYPE, copy=True),
                np.asarray(self.last_update_log, dtype=np.uint16, order="F"),
                np.asarray(self.last_delta_log, dtype=np.int8, order="F"),
            )
        if return_spiketimes and self.last_delta_log.size:
            spike_array = self._extract_spike_events(self.last_update_log, self.last_delta_log)
        if return_state and return_spiketimes:
            if spike_array is None:
                spike_array = np.zeros((2, 0), dtype=np.int64)
            return state_output, spike_array
        if return_state:
            return state_output
        if return_spiketimes:
            if spike_array is None:
                return np.zeros((2, 0), dtype=np.int64)
            return spike_array
        return None

    def _reconstruct_states(self, base_state, updates_log, delta_log, *, final_only=False):
        updates = np.asarray(updates_log, dtype=np.int64)
        deltas = np.asarray(delta_log, dtype=np.int8)
        if updates.ndim != 2 or deltas.shape != updates.shape:
            return np.zeros((self.N, 0), dtype=STATE_TYPE) if not final_only else np.zeros(self.N, dtype=STATE_TYPE)
        steps = updates.shape[1]
        if steps == 0:
            return base_state.astype(STATE_TYPE, copy=True) if not final_only else base_state.astype(STATE_TYPE, copy=True)
        base = np.asarray(base_state, dtype=np.int8).reshape(self.N)
        if final_only:
            accumulator = np.zeros(self.N, dtype=np.int32)
            for idx in range(steps):
                units = updates[:, idx]
                if units.size == 0:
                    continue
                delta = deltas[:, idx].astype(np.int32, copy=False)
                np.add.at(accumulator, units, delta)
            result = np.clip(base + accumulator, 0, 1)
            return result.astype(STATE_TYPE, copy=False)
        result = np.zeros((self.N, steps), dtype=STATE_TYPE)
        current = base.copy()
        for idx in range(steps):
            units = updates[:, idx]
            delta = deltas[:, idx]
            if units.size:
                current[units] = np.clip(current[units] + delta, 0, 1)
            result[:, idx] = current.astype(STATE_TYPE, copy=False)
        return result

    def _extract_spike_events(self, updates_log, delta_log):
        updates = np.asarray(updates_log)
        deltas = np.asarray(delta_log)
        if updates.ndim != 2 or deltas.shape != updates.shape:
            return np.zeros((2, 0), dtype=np.int64)
        per_step, steps = updates.shape
        if per_step == 0 or steps == 0:
            return np.zeros((2, 0), dtype=np.int64)
        mask = deltas > 0
        if not mask.any():
            return np.zeros((2, 0), dtype=np.int64)
        times = np.repeat(np.arange(steps, dtype=np.int64), per_step)
        flat_units = updates.reshape(-1, order="F").astype(np.int64, copy=False)
        flat_mask = mask.reshape(-1, order="F")
        return np.vstack((times[flat_mask], flat_units[flat_mask]))

    def compute_fields_from_states(self, states, external_input=None):
        """Compute subthreshold fields for recorded states."""
        state_block = np.asarray(states, dtype=float)
        if state_block.ndim == 1:
            state_block = state_block[:, None]
        if state_block.shape[0] != self.N:
            raise ValueError("states must have length equal to network size")
        if external_input is None:
            external_block = np.zeros((self.input_weights.shape[1], state_block.shape[1]))
        else:
            external_block = np.asarray(external_input, dtype=float)
            if external_block.ndim == 1:
                external_block = external_block[:, None]
        if external_block.shape[0] != self.input_weights.shape[1]:
            raise ValueError("external_input dimension mismatch.")
        syn_input = self.weights.dot(state_block)
        ext_input = self.input_weights.dot(external_block)
        return syn_input + ext_input - self.T[:, None]

class BalancedNetwork(_BaseNetwork):
    def __init__(self,Ns,ps,Ts,n_updates=1,update_ratios=[1,2],update_mode = 'random',delta_j=None,delta_T = None,g=1):
        """ implements a balanced network of excitatory and inhibitory units. 

            Ns:      List of population sizes [N_E,N_I]
            
            ps:      Matrix of connection probabilities must have shape (2,2)
            
            Ts:      List of thresholds [T_E,T_I]
            
            taus:    List of time constants [tau_E,tau_I]. Separate update queues
                     are generated for each population so that each unit from 
                     population A is updated on average every tau_A time steps.
            
            delta_j: Fractional range for distributed synaptic weights. Weights
                     are drawn from [j*(1-delta_j),j*(1+delta_j)] must be scalar
                     or have shape (2,2)

            delta_T: Fractional range for distributed thresholds, simular to delta_j.
                     Must be scalar or have length 2.  

            g:       Factor multiplying the EI weights

        """
        
        # generate the thresholds
        T = []
        if delta_T is not None:
            if np.isscalar(delta_T):
                delta_T = [delta_T]*len(Ns)
        for i,N in enumerate(Ns):
            t = np.ones(N) * Ts[i]
            # apply variance if required
            if delta_T is not None:
                t += delta_T[i]* (np.random.rand(N)- 0.5)
            T.append(t)
        T = np.concatenate(T)

        # generate weights
        w = generate_balanced_weights(Ns,ps,Ts,delta_j,g= g)
        _BaseNetwork.__init__(self,w,T)

        offsets = [0] + [N for N in Ns[:-1]]
        self.update_queue = UpdateQueue(Ns,n_updates,update_ratios,mode=update_mode)
        self.Ns = Ns
        self.ps = ps

        total_updates = np.array([n*u for n,u in zip(Ns,update_ratios)]).sum()
        self.taus = [total_updates/float(n_updates*ur) for ur in update_ratios]

    def _get_next_updates(self):
        return self.update_queue.next()
        
