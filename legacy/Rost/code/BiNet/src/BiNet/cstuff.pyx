"""# cython: profile=True""" 
from __future__ import division

import numpy as np
from numpy.random import randint,rand
cimport numpy as np
from libc.math cimport erfc as erfc_wrong_scale

cimport cython
cimport cpython
from cpython.array cimport array, clone

# np.int_t no longer exists in modern NumPy headers; lock the legacy code to 64-bit ints.
ctypedef np.int64_t DTYPE_t


cpdef double erfc(double x):
    return erfc_wrong_scale(x/(2)**0.5)/2.

cpdef sample_without_replacement(int sampleSize,int populationSize,int numSamples):

    if sampleSize == 1:
        return randint(0,populationSize,(numSamples,1))

    cdef np.ndarray[DTYPE_t, ndim=2] samples  = np.empty((numSamples, sampleSize),dtype=np.int64)

    

    # Use Knuth's variable names

    cdef int n = sampleSize

    cdef int N = populationSize

    cdef int i = 0

    cdef DTYPE_t t = 0 # total input records dealt with

    cdef int m = 0 # number of items selected so far
    
    cdef double u
    cdef np.ndarray[double, ndim = 1] us
    cdef int counter

    
    while i < numSamples:

        t = 0

        m = 0 
        counter = 0
        us = rand(N) # call a uniform(0,1) random number generatornp.rando

        while m < n :

            

            u = us[counter]
            counter += 1
            
            if  (N - t)*u >= n - m :

            

                t += 1

            

            else:

            

                samples[i,m] = t

                t += 1

                m += 1

                
        
        i += 1

        

    return samples


    

    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef  calc_new_states(np.ndarray[np.int8_t,ndim=1,cast=True] new_states,np.ndarray[np.int64_t,ndim=1] updates,np.ndarray[np.float64_t,ndim=2] weights,np.ndarray[np.int8_t,ndim=1,cast=True] state,np.ndarray[np.float64_t,ndim=2] input_weights,np.ndarray[np.float64_t,ndim=1] input,np.ndarray[np.float64_t,ndim=1] T,NonZeroCounter nonzerocounter):
    
    cdef int nrows = updates.shape[0]
    #cdef np.ndarray[np.int8_t,ndim=1,cast=True] new_states = np.empty(nrows,dtype=np.int8)
    cdef long[:] nonzero = nonzerocounter.indices
    cdef int i,j
    cdef long index,update
    cdef float subthreshold
    
    
    
    for i in range(nrows):
        subthreshold = 0
        update = updates[i]
        for j in range(nonzero.shape[0]):
            index = nonzero[j]
            if index<0:
                
                break

            subthreshold += weights[update,index] * state[index]

        for j in range(input_weights.shape[1]):
            subthreshold += input_weights[update,j] * input[j]
        new_states[i] = (subthreshold>T[update])*1

    for i in range(nrows):
        if new_states[i]>0:
            nonzerocounter.add(updates[i])
        else:
            nonzerocounter.discard(updates[i])

    #return new_states



cdef class NonZeroCounter:
    cdef public int maxind
    cdef public long[:] indices
    def __init__(self,np.ndarray[np.int8_t,ndim=1,cast =True] state):
        self.indices = np.zeros(state.shape[0],dtype=np.int64)
        nz = np.nonzero(state)[0]
        self.maxind = nz.shape[0]
        for i in range(self.maxind):
            self.indices[i] = nz[i]
        self.indices[self.maxind] = -1
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef long _bisect_left(self,int value):
        
        cdef long minind = 0
        cdef long maxind = self.maxind-1
        cdef long middle_ind,middle_val
        cdef long minval = self.indices[minind]
        cdef long maxval = self.indices[maxind]
        # check if value is outside range
        if value >= maxval:
            return maxind
        if (value < minval) or (minval == -1):
            return -1
        if value == minval:
            return minind
        while True:
            if maxind-minind<2:
                return minind
            middle_ind = (minind+maxind)//2
            middle_val = self.indices[middle_ind]
            if  middle_val== value:
                return middle_ind
            elif middle_val < value:
                minind = middle_ind
                continue
            else:
                maxind = middle_ind
                continue 

             
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef  add(self,int index):
        
        cdef size_t position = self._bisect_left(index)
        cdef long position_index = self.indices[position]
        cdef size_t i
        
        if position == -1:
            # index needs to be entered at beginning
            
            for i in range(self.maxind+1,0,-1):
                self.indices[i] = self.indices[i-1] 
            
            self.indices[0] = index
            
            self.maxind+=1
            self.indices[self.maxind] = -1
        elif position == (self.maxind-1) and (position_index!=index) :
           # index entered at end
           self.indices[self.maxind] = index
           self.maxind +=1
           self.indices[self.maxind] = -1
        elif position_index!=index:
            # index needs to be entered at position+1
            
            for i in range(self.maxind+1,position+1,-1):
                self.indices[i] = self.indices[i-1] 
            self.indices[position+1] = index
            
            self.maxind+=1
            self.indices[self.maxind] = -1



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef  discard(self,int index):
        
        cdef int position = self._bisect_left(index)
        cdef int position_index = self.indices[position]
        cdef int i
        if (position>-1) and (position<self.maxind) and (position_index == index):
            # position needs to be deleted
            
            for i in range(position,self.maxind):
                self.indices[i] = self.indices[i+1]
            self.maxind-=1
            self.indices[self.maxind] = -1

    
    
    def nonzero(self):
        return self.indices[:self.maxind]
    def get_indices(self):
        return np.array(self.indices)

        
