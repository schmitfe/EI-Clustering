import unittest
import numpy as np
import pylab
from .. import network as network


class TestUpdateQueue(unittest.TestCase):
    def test_sequential(self):

        uq = network.UpdateQueue([100,100],n_updates = 1,update_ratios = [2,3],mode='sequential')

        updates = pylab.zeros(200).astype(int)

        for i in range(200+300):
            updates[uq.next()] += 1

        expected_updates = pylab.ones(200).astype(int)*2
        expected_updates[100:] = 3
        self.assertTrue((updates==expected_updates).all())


    def test_random(self):
        Ns = [1000,1000]
        update_ratios = [1,2]
        uq = network.UpdateQueue(Ns,n_updates = 1,update_ratios = update_ratios,mode='random')

        updates = pylab.zeros(sum(Ns)).astype(int)

        for i in range(Ns[0]*update_ratios[0] + Ns[1]*update_ratios[1]):
            updates[uq.next()] += 1

        expected_updates = pylab.ones(sum(Ns)).astype(int)*update_ratios[0]
        expected_updates[Ns[0]:] = update_ratios[1]
        self.assertTrue((updates!=expected_updates).any())

        self.assertAlmostEqual(updates[:Ns[0]].mean(),update_ratios[0],delta = 0.05)
        self.assertAlmostEqual(updates[Ns[0]:].mean(),update_ratios[1],delta = 0.05)




class TestBaseNetwork(unittest.TestCase):
    def test_forward(self):
        N  =20
        T = 10
        weights = (pylab.rand(N,N)<0.5) * pylab.randn(N,N)
        Ts = pylab.rand(N)
        input_weights = pylab.rand(N,1)
        input = pylab.rand(1,T)

        net = network._BaseNetwork(weights,Ts,input_weights)

        packet = net.forward(input, return_state=True)
        output = net._reconstruct_states(*packet)
        self.assertTrue(output.shape == (N,T))

        expected_output = pylab.zeros((N,T)) 

        state = pylab.zeros_like(Ts)

        for t in range(T):
            for u in range(N):
                expected_output[u,t] = ((weights[u]*state).sum()+input_weights[u]*input[:,t])>Ts[u]
            state = expected_output[:,t]

        self.assertTrue((expected_output==output).all())
    
    def test_output_spiketimes(self):
        
        N  =20
        T = 10
        weights = (pylab.rand(N,N)<0.5) * pylab.randn(N,N)
        Ts = pylab.rand(N)
        input_weights = pylab.rand(N,1)
        input = pylab.rand(1,T)

        net = network._BaseNetwork(weights,Ts,input_weights)
        state = pylab.randint(0,2,N).astype(bool)
        net.set_state(state)
        packet = net.forward(input, return_state=True, strict_spiking=True)
        init_state, updates, deltas = packet
        state_log = net._reconstruct_states(init_state, updates, deltas)
        onset_count = int((net.last_delta_log > 0).sum())
        net.set_state(state)
        spiketimes= net.forward(input,return_spiketimes = True)
        self.assertEqual(onset_count,spiketimes.shape[1])

        if spiketimes.shape[1]:
            spikes = np.zeros_like(state_log, dtype=bool)
            spikes[spiketimes[1],spiketimes[0]] = True
            self.assertTrue(spikes.sum() == spiketimes.shape[1])

    def test_reconstruct_final_state(self):
        N  =20
        T = 10
        weights = (pylab.rand(N,N)<0.5) * pylab.randn(N,N)
        Ts = pylab.rand(N)
        input_weights = pylab.rand(N,1)
        input = pylab.rand(1,T)

        net = network._BaseNetwork(weights,Ts,input_weights)
        packet = net.forward(input, return_state=True)
        states = net._reconstruct_states(*packet)
        final_state = net._reconstruct_states(*packet, final_only=True)
        self.assertTrue(states.shape[1] == T)
        self.assertTrue((states[:,-1] == final_state).all())
    


        


class TestBalancedNetwork(unittest.TestCase):
    
    def test_updates(self):
        Ns = pylab.array([100,50])
        ps = pylab.array([[0.2,0.5],[0.5,0.5]])
        Ts = pylab.ones(2)
        for n_updates in [1,10]:
           
            net = network.BalancedNetwork(Ns,ps,Ts,n_updates,update_mode = 'random')
            maxtau = int(max(net.taus))
            us = [net._get_next_updates() for t in range(100*Ns.sum())]
            updates = pylab.zeros((Ns.sum(),len(us)))
            for t in range(len(us)):
                updates[us[t],t] = 1

            e_ints = []
            i_ints = []
            for u in range(Ns[0]):
                times = pylab.find(updates[u])
                e_ints += pylab.diff(times).tolist()
            for u in range(Ns[0],Ns.sum()):
                times = pylab.find(updates[u])
                i_ints += pylab.diff(times).tolist()
            e_ints = pylab.array(e_ints)
            i_ints = pylab.array(i_ints)
           
            self.assertAlmostEqual(e_ints.mean(),net.taus[0],delta =0.1*net.taus[0])
            self.assertAlmostEqual(i_ints.mean(),net.taus[1],delta =0.1*net.taus[1])

    def test_outputs_and_spikes(self):
        T  =5
        Ns = pylab.array([5,5])
        ps = pylab.ones((2,2))*0.5
        Ts = pylab.ones(2)
        n_updates = 1
        w_in = pylab.ones((Ns.sum(),1))
        w_in[:Ns[0]] = 10
        w_in[Ns[0]:]  = 5.
        net = network.BalancedNetwork(Ns,ps,Ts,n_updates)
        w = net.get_weights()
        net.set_input_weights(w_in)
        net.state = pylab.rand(net.N)<0.5
        initial_state = net.state.copy()
        input = pylab.ones((1,T))*0.1
        output = []
        updates = []
        for t in range(T):
           
            packet = net.forward(input[:,[t]],return_state = True)
            reconstructed = net._reconstruct_states(*packet)
            output.append(reconstructed[:,0])
            updates.append(net.current_updates)
        output = pylab.array(output).T
       
        expected_output = pylab.zeros_like(output)
        last_state = initial_state
        for t in range(T):
            expected_output[:,t] = last_state
            for u in updates[t]:
                rec = (w[u] * last_state).sum()
                ext =w_in[u]*input[0,t]
                expected_output[u,t] = (rec+ext)>net.T[u]
            last_state = output[:,t].copy()
        
        self.assertTrue((expected_output == output).all())

        



