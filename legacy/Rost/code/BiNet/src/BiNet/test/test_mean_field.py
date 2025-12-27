import unittest
import pylab
from .. import mean_field
from .. import weights
class TestWeighStats(unittest.TestCase):
    def test_means(self):
        # constant js
        for i in range(3):
            Ns=pylab.randint(500,1000,2)
            ps = 0.1+0.5*pylab.rand(2,2)
            js = 1. +3*pylab.rand(2,2)

            w = weights.generate_weight_matrix(Ns,ps,js,delta_j=0)
            mean_weights = mean_field._mean_weights(Ns,ps,js)

            mw = pylab.zeros((2,2))
            N_list = [0]+Ns.tolist()
            for i in range(2):
                for j in range(2):
                    w_block = w[sum(N_list[:i+1]):sum(N_list[:i+2]),sum(N_list[:j+1]):sum(N_list[:j+2])]
                    
                    mw[i,j] = w_block.sum(axis=1).mean()
            self.assertTrue(pylab.allclose(mw,mean_weights,rtol = 0.05))

    def test_vars(self):
        
        # constant js
        for i in range(1):
            Ns=pylab.randint(1000,2000,2)
            ps = 0.2+0.5*pylab.rand(2,2)
            js = 1. +3*pylab.rand(2,2)

            w = weights.generate_weight_matrix(Ns,ps,js,delta_j=0)
            weight_vars = mean_field._weight_vars(Ns,ps,js,delta_j = 0)

            wv = pylab.zeros((2,2))
            N_list = [0]+Ns.tolist()
            for i in range(2):
                for j in range(2):
                    w_block = w[sum(N_list[:i+1]):sum(N_list[:i+2]),sum(N_list[:j+1]):sum(N_list[:j+2])]
                    wv[i,j] = w_block.sum(axis=1).var()
             
            self.assertTrue(pylab.allclose(wv,weight_vars,rtol = 0.15))
            
        # variable js,ps=1.
        for i in range(1):
            Ns=pylab.randint(1000,2000,2)
            ps = pylab.ones((2,2))
            js = 1. +3*pylab.rand(2,2)
            delta_j = pylab.rand(2,2)*2
            w = weights.generate_weight_matrix(Ns,ps,js,delta_j=delta_j)
            weight_vars = mean_field._weight_vars(Ns,ps,js,delta_j = delta_j)
           
            wv = pylab.zeros((2,2))
            N_list = [0]+Ns.tolist()
            for i in range(2):
                for j in range(2):
                    w_block = w[sum(N_list[:i+1]):sum(N_list[:i+2]),sum(N_list[:j+1]):sum(N_list[:j+2])]
                    wv[i,j] = w_block.sum(axis=1).var()

            
            self.assertTrue(pylab.allclose(wv,weight_vars,rtol = 0.15))

        # variable js,variable ps
        for i in range(1):
            Ns=pylab.randint(1000,2000,2)
            ps = 0.2 + 0.5*pylab.rand(2,2)
            js = 1. +3*pylab.rand(2,2)
            delta_j = pylab.rand(2,2)*2
            w = weights.generate_weight_matrix(Ns,ps,js,delta_j=delta_j)
            weight_vars = mean_field._weight_vars(Ns,ps,js,delta_j = delta_j)
           
            wv = pylab.zeros((2,2))
            N_list = [0]+Ns.tolist()
            for i in range(2):
                for j in range(2):
                    w_block = w[sum(N_list[:i+1]):sum(N_list[:i+2]),sum(N_list[:j+1]):sum(N_list[:j+2])]
                    wv[i,j] = w_block.sum(axis=1).var()

            
            self.assertTrue(pylab.allclose(wv,weight_vars,rtol = 0.15))


class TestSteadyStateRates(unittest.TestCase):
    def test_carl_rates(self):
        # for large n and k, solution should converge to the results of van vresswijk
        Ns = pylab.array([1e10,1e10])
        K = 1e5
        g=1.2
        ps = pylab.ones((2,2))*K/Ns[pylab.newaxis,:]
        Ts = pylab.ones(2)
        js = weights.calc_js(Ns,ps,Ts,g)
        taus = pylab.array([10,5.])
        jxs = pylab.ones(2)
        jxs[1] = 0.5*jxs[0]
        
        mx = 0.2
        
        
        def func(m):
            return mean_field.dm_dt(m,taus,js,ps,Ns,Ts,jxs*pylab.sqrt(K),mx)

        ss_m = mean_field.integrate_to_convergence(func,pylab.rand(2),delta = 1e-5,verbose=False)[0]
        
        ss_m = mean_field.integrate_to_convergence(func,pylab.array(ss_m),dt = 0.0001,delta = 1e-10,verbose=False,solve=False)[0]
        # rates according to van vreeswijk 1998

        me = (1*jxs[0]-g*jxs[1])/(g-1)*mx
        mi = (jxs[0]-jxs[1])/(g-1)*mx
        
        

        self.assertAlmostEqual(me,ss_m[0],delta = 0.01)
        self.assertAlmostEqual(mi,ss_m[1],delta = 0.01)
        # rates calculated by mean_field equations should also match
        ms_mf = mean_field.m_steady_state_theoretical(Ns,ps,js,jxs*pylab.sqrt(K),mx)
        self.assertAlmostEqual(me,ms_mf[0],delta = 0.01)
        self.assertAlmostEqual(mi,ms_mf[1],delta = 0.01)

