

import os
import pickle
import functools
from copy import deepcopy
from collections.abc import Hashable

import pylab
from joblib import Parallel, delayed
import multiprocessing
import unittest

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, Hashable):
            print('uncacheable: ', args)
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

class memoized_but_forgetful(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args,**kwargs):

        key = (args, frozenset(sorted(kwargs.items())))
        if not isinstance(key, Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args,**kwargs)
        if key in self.cache:
            return self.cache[key]
        else:
            # forget old value
            self.cache = {}
            value = self.func(*args,**kwargs)
            self.cache[key] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


datapath = os.path.abspath(os.path.join(os.path.dirname(__file__),'./data'))

@memoized
def get_data_file(filename):
    try:
        with open(filename, 'rb') as handle:
            all_results = pickle.load(handle)
    except:
        all_results = {}
    return all_results

def key_from_params(params,key_list):
    key = ''
    for k in key_list:
        key += '_'+params[k].__repr__()
    key =key.replace(' ','')
    return key

def params_from_keys(key,key_list):
    items = [eval(i.replace('array','')) for i in key.split('_')[1:]]
    return dict(zip(key_list,items))

def check_and_execute_hetero(param_list,func,datafile,redo = False,n_jobs = 1):
    full_datafile = os.path.join(datapath,datafile)
    all_results = get_data_file(full_datafile)
    keys =[]
    return_keys = []
    for params in param_list:
        key_list = sorted(params.keys())
        keys.append(key_from_params(params,key_list))
        return_keys.append(keys[-1])
    if not redo:
        all_keys = set(all_results.keys())
        drop_inds = []
        for i,k in enumerate(keys):
            if k in all_keys:
                drop_inds.append(i)
        for i in drop_inds[::-1]:
            keys.pop(i)
            param_list.pop(i)
    if len(param_list)>0:
        get_data_file.cache = {}
        print(len(keys),' to generate')
        
        if n_jobs ==1:
            results = [func(p) for p in param_list]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(func)(deepcopy(p)) for p in param_list)



        for k,r in zip(keys,results):
            all_results[k] = r
        with open(full_datafile,'wb') as handle:
            pickle.dump(all_results,handle,protocol = 2)
    
    return [all_results[k] for k in return_keys]

    
def check_and_execute(params,func,datafile,key_list=None,reps = None,redo = False,backup_file = None,n_jobs = 1,save = True,ignore_keys = []):
    
    
    if key_list is None:
        key_list = [k for k in sorted(params.keys()) if k not in ignore_keys]
    
    
    key = key_from_params(params,key_list)

    
    
    full_datafile = os.path.join(datapath,datafile)                                
     
                                    
                                    
                                    
    try:
        if redo:
            raise
        all_results = get_data_file(full_datafile)
        
        if reps is None and not redo:
            results = all_results[key]
            result_keys = [key]

        elif not redo:
            
            result_keys =  [key+'_'+str(r) for r in range(reps)]
            
            results = [all_results[result_key] for result_key in result_keys]
            
            
        elif redo:
            print('redo')
            raise 
    
    except:
        cache_key = (full_datafile,)
        if cache_key in get_data_file.cache.keys():
            get_data_file.cache.pop(cache_key)
        try:
            with open(full_datafile, 'rb') as handle:
                all_results = pickle.load(handle)
        except:
            
            all_results = {}  
            if save:
                with open(full_datafile,'wb') as handle:
                    pickle.dump(all_results,handle,protocol = 2)
        if reps is None:
            print('no reps')     
            results = func(params)
            all_results[key] = results
        else:
            if not redo:
                all_keys = sorted([k for k in all_results.keys() if key in k])
                try:
                    all_keys.remove(key)
                except:
                    pass
            else:
                all_keys = []
            possible_keys = [key+'_'+str(r) for r in range(reps)]
            missing_keys = [k for k in possible_keys if k not in all_keys]

            if n_jobs>1:
                print('n_jobs: ', n_jobs)
                print('careful: in parallel, randseeds need to be set')
                copied_params = [deepcopy(params) for mk in missing_keys]
                
                
                new_results = Parallel(n_jobs=n_jobs)(delayed(func)(cp) for cp in copied_params)
                #new_results = multiprocessing.Pool(processes=n_jobs).map(func,copied_params)
                
                for mk,nr in zip(missing_keys,new_results):
                    all_results[mk] = nr
            else:
                for mk in missing_keys:
                    print('loop')
                    all_results[mk] = func(deepcopy(params))
                    if save:
                        with open(full_datafile,'wb') as handle:
                            pickle.dump(all_results,handle,protocol = 2)
            
            all_keys = sorted([k for k in all_results.keys() if key in k])
            results = [all_results[k] for k in all_keys[:reps]]
            
        if save:
            with open(full_datafile,'wb') as handle:
                pickle.dump(all_results,handle,protocol = 2)

    if backup_file is not None:
        result_dict = {}
        try:
            if reps is not None:
                for rk,r in zip(result_keys,results):
                    result_dict[rk] = r
            else:
                raise
        except:
            result_dict[key] = results
        with open(backup_file,'wb') as handle:
            pickle.dump(result_dict,handle,protocol= 2)
        
    return results
    

def _recursive_in(var,val):
    if hasattr(var,'__iter__'):
        return True in [_recursive_in(v,val) for v in var]
    else:
        return var==val

def contains_nans(var):
    if hasattr(var,'__iter__'):
        return [_recursive_in(contains_nans(v),True) for v in var]
        

    return pylab.isnan(var)



class PackagedArray(object):
    def __init__(self,array):
        self.shape = array.shape
        flat_array = array.flatten()
        self.flat_shape = flat_array.shape
        self.inds = pylab.where(flat_array!=0)[0].astype(pylab.int32)
        self.data = flat_array[self.inds]
   
    def unpack(self):
        # reconstruct dense array
        flat_array = pylab.zeros(self.flat_shape,dtype=self.data.dtype)
        flat_array[self.inds] = self.data
        return flat_array.reshape(self.shape)


class TestPackagedArray(unittest.TestCase):
    def test_binary(self):
        spikes = pylab.randint(0,2,(10,1000,1000)).astype(bool)
        package = PackagedArray(spikes)
        unpacked = package.unpack()
        self.assertTrue((spikes==unpacked).all())

    def test_float(self):
        spikes = pylab.randint(0,2,(10,20,100)).astype(float)*pylab.randn(10,20,100)
        package = PackagedArray(spikes)
        unpacked = package.unpack()
        self.assertTrue((spikes==unpacked).all())

if __name__ == '__main__':
    unittest.main()
