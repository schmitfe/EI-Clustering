import unittest
import pylab
from BiNet import weights


class TestGenerateWeightMatrix(unittest.TestCase):

    def test_calc_j(self):
        for i in range(3):
            Ns = pylab.randint(1000,2000,2)
            ps = 0.1 + 0.5*pylab.rand(2,2)
            Ts = 0.1+pylab.rand(2)
            js = weights.calc_js(Ns,ps,Ts,1)

            for row in [0,1]:
                # root{k} = root{p*N} should equal T for excitation
                self.assertAlmostEqual(pylab.sqrt(ps[row,0]*Ns[0])*js[row,0]/pylab.sqrt(Ns.sum()),Ts[row])
                # rows should be balanced
                self.assertAlmostEqual(ps[row,0]*Ns[0]*js[row,0]/pylab.sqrt(Ns.sum()),-ps[row,1]*Ns[1]*js[row,1]/pylab.sqrt(Ns.sum()))

    def test_generate_weight_block(self):
        for N_out in pylab.randint(10,1000,3):
            for N_in in pylab.randint(10,1000,3):
                for p in [0.1,0.5,1.]:
                    wb = weights.generate_weight_block(N_out,N_in,p)
                    self.assertEqual(wb.shape,(N_out,N_in))
                    self.assertAlmostEqual(p,wb.mean(),delta = 0.05)

    def test_generate_weight_matrix(self):

        # test shape
        ps = pylab.ones((3,3))
        js = pylab.ones((3,3))
        Ns = pylab.array([100,200,300])
        wm = weights.generate_weight_matrix(Ns,ps,js,None)
        self.assertEqual(wm.shape,(Ns.sum(),Ns.sum()))
        # diagonal should always be zero
        self.assertTrue((wm[range(Ns.sum()),range(Ns.sum())]==0).all())

        # test ps and j values
        ps = pylab.rand(2,2)*0.5 + 0.1
        js = pylab.rand(2,2)*5 +1
        js[:,1] *= -1
        N=600
        Ns = pylab.array([500,100])
        wm = weights.generate_weight_matrix(Ns,ps,js,None)

        Ns = [0]+Ns.tolist()
        for i in range(2):
            for j in range(2):
                w = wm[Ns[i]:Ns[i]+Ns[i+1],Ns[j]:Ns[j]+Ns[j+1]]
                self.assertAlmostEqual((w!=0).mean(),ps[i,j],delta = 0.05)
                j_block = w[w!=0].mean()
                self.assertAlmostEqual(j_block*pylab.sqrt(N),js[i,j])


        # test distributed js
        ps = pylab.rand(2,2)*0.5 + 0.1
        js = pylab.rand(2,2)*5 +1
        js[:,1] *= -1
        delta_j = pylab.rand(2,2)
        N=600
        Ns = pylab.array([500,100])
        wm = weights.generate_weight_matrix(Ns,ps,js,delta_j)

        Ns = [0]+Ns.tolist()
        for i in range(2):
            for j in range(2):
                w = wm[Ns[i]:Ns[i]+Ns[i+1],Ns[j]:Ns[j]+Ns[j+1]]
                j_block = w[w!=0]*pylab.sqrt(N)
                self.assertAlmostEqual(j_block.mean(),js[i,j],delta =0.1)
                # lower and upper depend on wheter j is positive or negative
                lower = min(js[i,j]-0.5*js[i,j]*delta_j[i,j],js[i,j]+0.5*js[i,j]*delta_j[i,j])
                upper = max(js[i,j]-0.5*js[i,j]*delta_j[i,j],js[i,j]+0.5*js[i,j]*delta_j[i,j])
                self.assertTrue(j_block.min()>lower)
                self.assertTrue(j_block.max()<upper)
                # variance of a unifor distribution
                self.assertAlmostEqual(j_block.var()/((upper-lower)**2 /12.),1,delta = 0.05)

                





    def test_balanced_weight_matrix(self):
        
        Ns = pylab.array([2000,1000])
        N = Ns.sum()
        ps = pylab.ones((2,2)) * 0.5
        ps[0,0] = 0.2
        Ts = pylab.array([1,1])
        g = 1.2
        wm = weights.generate_balanced_weights(Ns,ps,Ts,delta_j=None,g=g)

        Ns = [0] +Ns.tolist()

        w_sum =pylab.zeros((2,2))

        for i in range(2):
            for j in range(2):
                w = wm[Ns[i]:Ns[i]+Ns[i+1],Ns[j]:Ns[j]+Ns[j+1]]
                self.assertAlmostEqual((w!=0).mean(),ps[i,j],delta = 0.05)
                w_sum[i,j] = w.sum(axis=1).mean()
        w_sum[0,1]/=g
        self.assertAlmostEqual((w_sum[:,0]/w_sum[:,1]).mean(),-1,delta  =0.05)



class TestClusterSpecs(unittest.TestCase):
    
    def test_mazzucato_specs(self):

        Ns = pylab.array([1000,250])
        ps = pylab.ones((2,2)) * 0.5
        ps[0,0] = 0.2
        Ts = pylab.ones((2))
        taus = pylab.array([10,5])
        g = 1.5
        f = 0.9
        Q = 30
        j_plus = 1.5
        gamma = 0.5

        newNs,newps,newjs,newTs,newtaus = weights.mazzucato_cluster_specs(Ns,ps,Ts,taus,g,f,Q,j_plus,gamma,original_j_minus = True)
        self.assertEqual(Ns[1],newNs[-1])
        self.assertEqual(Ns[0],sum(newNs[:-1]))
        self.assertEqual(Ns[0]*f,sum(newNs[:-2]))
        self.assertTrue((newps[:-1,:-1]==ps[0,0]).all())
        self.assertTrue((newps[-1,:-1]==ps[1,0]).all())    
        self.assertTrue((newps[:-1,-1]==ps[0,1]).all())    
        self.assertTrue((newps[-1,-1]==ps[1,1]).all())      

        js = weights.calc_js(Ns,ps,Ts,g)
        j_minus = 1 -gamma*f*(j_plus-1)
        self.assertTrue((newjs[range(Q),range(Q)]==js[0,0]*j_plus).all())
        newjs[range(Q),range(Q)] = js[0,0]*j_minus
        self.assertTrue(pylab.allclose(newjs[:Q,:Q],js[0,0]*j_minus))
        self.assertTrue((newjs[Q,:-1]==js[0,0]).all())
        self.assertTrue((newjs[:-1,Q]==js[0,0]).all())
        self.assertTrue((newjs[-1,:-1]==js[1,0]).all())    
        self.assertTrue((newjs[:-1,-1]==js[0,1]).all())    
        self.assertTrue((newjs[-1,-1]==js[1,1]).all())

    def test_doiron_specs(self):

        Ns = pylab.array([1000,250])
        ps = pylab.ones((2,2)) * 0.5
        ps[0,0] = 0.2
        Ts = pylab.ones((2))
        taus = pylab.array([10,5])
        g = 1.5
        Q = 20
        j_plus = 1.5
        R_EE = 2.5

        newNs,newps,newjs,newTs,newtaus = weights.doiron_cluster_specs(Ns,ps,Ts,taus,g,Q,R_EE,j_plus)

        self.assertEqual(Ns[1],newNs[-1])
        self.assertEqual(Ns[0],sum(newNs[:-1]))
        
        p_out = ps[0,0]*Q/(Q-1+R_EE)
        p_in = R_EE * p_out
        self.assertTrue(pylab.allclose(newps[range(Q),range(Q)],p_in))
        newps[range(Q),range(Q)] = p_out
        self.assertTrue((newps[:-1,:-1]==p_out).all())
        self.assertTrue((newps[-1,:-1]==ps[1,0]).all())    
        self.assertTrue((newps[:-1,-1]==ps[0,1]).all())    
        self.assertTrue((newps[-1,-1]==ps[1,1]).all())

        js = weights.calc_js(Ns,ps,Ts,g)
        self.assertTrue((newjs[range(Q),range(Q)]==js[0,0]*j_plus).all())
        newjs[range(Q),range(Q)] = js[0,0]
        self.assertTrue(pylab.allclose(newjs[:Q,:Q],js[0,0]))
        self.assertTrue((newjs[-1,:-1]==js[1,0]).all())    
        self.assertTrue((newjs[:-1,-1]==js[0,1]).all())    
        self.assertTrue((newjs[-1,-1]==js[1,1]).all())
    

    def test_EI_spec_1(self):
        Ns = pylab.array([1000,250])
        ps = pylab.ones((2,2)) * 0.5
        ps[0,0] = 0.2
        Ts = pylab.ones((2))
        taus = pylab.array([10,5])
        g = 1.5
        js = weights.calc_js(Ns, ps, Ts, g)
        Q = 10
        j_plus_EE = 1.5
        R_EE = 2.
        R_EI = 1.5 
        j_plus_EI = 2. 
        R_IE = 3.
        j_plus_IE = 3.
        R_II = 0.5
        j_plus_II = 0.1

        newNs,newps,newjs,newTs,newtaus = weights.EI_cluster_specs_1(Ns,ps,Ts,taus,g,Q,R_EE,R_EI,R_IE,R_II,j_plus_EE,j_plus_EI,j_plus_IE,j_plus_II)

        self.assertEqual(newNs.shape[0], 2*Q)
        self.assertEqual(newps.shape, (2*Q,2*Q))
        self.assertEqual(newjs.shape, (2*Q,2*Q))
        self.assertEqual(newTs.shape[0], 2*Q)
        self.assertEqual(newtaus.shape[0], 2*Q)
        
        K_EE = ps[0,0]*Ns[0]

        for i in range(Q):
            k_row = 0
            for j in range(Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_EE)

        K_EI = ps[0,1]*Ns[1]

        for i in range(Q):
            k_row = 0
            for j in range(Q,2*Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_EI)

        K_IE = ps[1,0]*Ns[0]

        for i in range(Q,2*Q):
            k_row = 0
            for j in range(Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_IE)

        K_II = ps[1,1]*Ns[1]
        for i in range(Q,2*Q):
            k_row = 0
            for j in range(Q,2*Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_II)

        
        for i in range(2*Q):
            for j in range(2*Q):
                
                if i<Q and j<Q:
                    if i==j:
                        self.assertAlmostEqual(newjs[i,j],js[0,0]*j_plus_EE)
                    else:
                        self.assertAlmostEqual(newjs[i,j],js[0,0])
                elif i<Q and j>=Q:
                    if i== (j-Q):
                        self.assertAlmostEqual(newjs[i,j],js[0,1]*j_plus_EI)
                    else:
                        self.assertAlmostEqual(newjs[i,j],js[0,1])
                elif i>=Q and j<Q:
                    if (i-Q)==j:
                        self.assertAlmostEqual(newjs[i,j],js[1,0]*j_plus_IE)
                    else:
                        self.assertAlmostEqual(newjs[i,j],js[1,0])
                else:
                    if i==j:
                        self.assertAlmostEqual(newjs[i,j],js[1,1]*j_plus_II)
                    else:
                        self.assertAlmostEqual(newjs[i,j],js[1,1])

        self.assertTrue((newtaus[:Q]==taus[0]).all())
        self.assertTrue((newtaus[Q:]==taus[1]).all())

        self.assertTrue((newTs[:Q]==Ts[0]).all())
        self.assertTrue((newTs[Q:]==Ts[1]).all())
    def test_EI_spec_2(self):
        Ns = pylab.array([1000,250])
        ps = pylab.ones((2,2)) * 0.5
        ps[0,0] = 0.2
        Ts = pylab.ones((2))
        taus = pylab.array([10,5])
        g = 1.5
        js = weights.calc_js(Ns, ps, Ts, g)
        Q = 10
        R_EE = 2.
        R_EI = 1.5 
        R_IE = 3.
        R_II = 0.5
        

        newNs,newps,newjs,newTs,newtaus = weights.EI_cluster_specs_2(Ns,ps,Ts,taus,g,Q,R_EE,R_EI,R_IE,R_II)

        self.assertEqual(newNs.shape[0], 2*Q)
        self.assertEqual(newps.shape, (2*Q,2*Q))
        self.assertEqual(newjs.shape, (2*Q,2*Q))
        self.assertEqual(newTs.shape[0], 2*Q)
        self.assertEqual(newtaus.shape[0], 2*Q)
        
        K_EE = ps[0,0]*Ns[0]/float(Q)*(1+(Q-1)/R_EE)

        for i in range(Q):
            k_row = 0
            for j in range(Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_EE)

        K_EI =ps[0,1]*Ns[1]/float(Q)*(1+(Q-1)/R_EI)

        for i in range(Q):
            k_row = 0
            for j in range(Q,2*Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_EI)

        K_IE =ps[1,0]*Ns[0]/float(Q)*(1+(Q-1)/R_IE)

        for i in range(Q,2*Q):
            k_row = 0
            for j in range(Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_IE)

        K_II = ps[1,1]*Ns[1]/float(Q)*(1+(Q-1)/R_II)
        for i in range(Q,2*Q):
            k_row = 0
            for j in range(Q,2*Q):
                k_row += newNs[j]*newps[i,j]
            self.assertAlmostEqual(k_row,K_II)

        
        for i in range(2*Q):
            for j in range(2*Q):
                
                if i<Q and j<Q:
                    row_sum_old = Ns[0]*ps[0,0]*js[0,0]
                    row_sum_new = (newjs[i,:Q]*newNs[:Q]*newps[i,:Q]).sum()
                    self.assertAlmostEqual(row_sum_old,row_sum_new)
                elif i<Q and j>=Q:
                    row_sum_old = Ns[1]*ps[0,1]*js[0,1]
                    row_sum_new = (newjs[i,Q:]*newNs[Q:]*newps[i,Q:]).sum()
                elif i>=Q and j<Q:
                    row_sum_old = Ns[0]*ps[1,0]*js[1,0]
                    row_sum_new = (newjs[i,Q]*newNs[:Q]*newps[i,:Q]).sum()
                else:
                    row_sum_old = Ns[1]*ps[1,1]*js[1,1]
                    row_sum_new = (newjs[i,Q:]*newNs[Q:]*newps[i,Q:]).sum()

        self.assertTrue((newtaus[:Q]==taus[0]).all())
        self.assertTrue((newtaus[Q:]==taus[1]).all())

        self.assertTrue((newTs[:Q]==Ts[0]).all())
        self.assertTrue((newTs[Q:]==Ts[1]).all())
                    













if __name__ == '__main__':
    unittest.main()










                



