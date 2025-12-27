import numpy as np
import pyximport; pyximport.install(setup_args = {"include_dirs":np.get_include()},reload_support=True)
import unittest
from .. import cstuff
import pylab

class TestNonZeroCounter(unittest.TestCase):
    def test_init(self):
        state = pylab.randint(0,2,100).astype(bool)
        nz = cstuff.NonZeroCounter(state)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

    def test_change_first(self):
        state = pylab.randint(0,2,100).astype(bool)
        state[0] = 0
        nz = cstuff.NonZeroCounter(state)
        state[0] = 1
        nz.add(0)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
        state[0] = 0
        nz.discard(0)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

        state = pylab.randint(0,2,100).astype(bool)
        firstind = pylab.find(state==True)[0]
        nz = cstuff.NonZeroCounter(state)
        state[firstind] = False
        nz.discard(firstind)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

    def test_add_to_empty(self):
        state = pylab.zeros(1000,dtype =bool)
        nz = cstuff.NonZeroCounter(state)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

        state[200] = True
        nz.add(200)
        
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
    def test_discard_empty(self):
        # at the beginning
        state = pylab.randint(0,2,100).astype(bool)
        zero_ind = pylab.find(state==False)[0]
        nz = cstuff.NonZeroCounter(state)
        nz.discard(zero_ind)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
        # in the middle
        zeroinds = pylab.find(state==False)
        zero_ind = zeroinds[len(zeroinds)/2]
        nz.discard(zero_ind)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
        # at the end
        zero_ind = pylab.find(state==False)[-1]
        nz.discard(zero_ind)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
    def test_change_last(self):
        state = pylab.randint(0,2,100).astype(bool)
        state[99] = 0
        nz = cstuff.NonZeroCounter(state)
        state[99] = 1
        nz.add(99)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())
        state[99] = 0
        nz.discard(99)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

        state = pylab.randint(0,2,100).astype(bool)
        firstind = pylab.find(state==True)[-1]
        nz = cstuff.NonZeroCounter(state)
        state[firstind] = False
        nz.discard(firstind)
        self.assertTrue((nz.nonzero()==state.nonzero()[0]).all())

    def test_compare_nonzeros(self):
        state = pylab.randint(0,2,100).astype(bool)
        
        nz = cstuff.NonZeroCounter(state)
        
        full = nz.get_indices()
        compact = nz.nonzero()
        self.assertTrue(full.shape==state.shape)
        self.assertTrue((compact==state.nonzero()[0]).all())
        end = pylab.find(full==-1)
        self.assertTrue((compact==full[:end]).all())

        for i in range(10):
            index = pylab.randint(100)
            state[index] = 1
            nz.add(index)
            full = nz.get_indices()
            compact = nz.nonzero()
            self.assertTrue(full.shape==state.shape)
            self.assertTrue((compact==state.nonzero()[0]).all())
            end = pylab.find(full==-1)[0]
            self.assertTrue((compact==full[:end]).all())
            index = pylab.randint(100)
            state[index] = 0
            nz.discard(index)
            full = nz.get_indices()
            compact = nz.nonzero()
            self.assertTrue(full.shape==state.shape)
            self.assertTrue((compact==state.nonzero()[0]).all())
            end = pylab.find(full==-1)[0]
            self.assertTrue((compact==full[:end]).all())











if __name__=='__main__':
    unittest.main()



        


