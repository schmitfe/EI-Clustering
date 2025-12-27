from BiNet import network, mean_field, weights
import organiser
import spiketools
import pylab
import os
from copy import deepcopy
from time import time as clock
from scipy.signal import decimate, convolve2d


def simulate(original_params):
    params = deepcopy(original_params)
    try:
        pylab.seed(params['randseed'])
    except:
        pylab.seed(None)
    simulation_type = str(params.get('simulation_type', 'rost')).lower()
    if simulation_type not in {'rost', 'new'}:
        raise ValueError(f"Unknown simulation_type '{simulation_type}'. Expected 'rost' or 'new'.")

    raw_connection_type = params.get('connection_type', 'bernoulli')
    connection_type = str(raw_connection_type).replace(' ', '_').replace('-', '_').lower()
    valid_connection_types = {'bernoulli', 'poisson', 'fixed_indegree'}
    if connection_type not in valid_connection_types:
        raise ValueError(f"Unknown connection_type '{raw_connection_type}'. Expected one of {sorted(valid_connection_types)}.")

    raw_kappa = params.get('kappa', 0.0)
    kappa = 0.0 if raw_kappa is None else float(raw_kappa)
    if not 0.0 <= kappa <= 1.0:
        raise ValueError(f"kappa must be in [0, 1], got {kappa}")

    if params['spec_func'] == 'mazzucato':
        spec_func = weights.mazzucato_cluster_specs
    elif params['spec_func'] == 'doiron':
        spec_func = weights.doiron_cluster_specs
    elif params['spec_func'] == 'EI_jplus':
        spec_func = weights.EI_jplus_cluster_specs

    return_mean_cluster_rates = params.get('return_mean_cluster_rates',False)
    return_max_cluster_rates = params.get('return_max_cluster_rates',False)
    return_cluster_rates_and_spiketimes = params.get('return_cluster_rates_and_spiketimes',False)
    smooth_rates = params.get('smooth_rates',None) # std of gaussian smoothing kernel in ms
    
    downsample = params.get('downsample',None)

    net = network.BalancedNetwork(**params['network_args'])
    
    spec_args= params['spec_args']
    for k in ['Ns','ps','Ts','g']:
        spec_args[k] = params['network_args'][k]
    spec_args['taus'] = net.taus/net.taus[0]*params['effective_tau']
    if simulation_type == 'new':
        spec_args['kappa'] = kappa

    cNs,cps,cjs,cTs,ctaus = spec_func(**spec_args)

    w = weights.generate_weight_matrix(cNs,cps,cjs,delta_j=None, connection_type=connection_type)
    net.set_weights(w)
    
    w_in = pylab.ones((net.N,2))
    w_in[:net.Ns[0],0] = params['jxs'][0]
    w_in[net.Ns[0]:,0] = params['jxs'][1]
    stim_clusters = list(range(len(cNs)-2))
    pylab.shuffle(stim_clusters)
    stim_clusters = stim_clusters[:params['stim_clusters']]
    n_bins = [0]+pylab.cumsum(cNs).tolist()
    for stim_cluster in stim_clusters:
        inds = list(range(n_bins[stim_cluster],n_bins[stim_cluster+1]))
        w_in[inds,1] = params['jxs'][0]

    net.set_input_weights(w_in)
    net.initialise_state(params['init'])

    input = pylab.ones((2,int(params['input_length']*net.taus[0])))*params['mx']
    input[1,:] *= 0

    input[1,int(params['stim_start']*net.taus[0]):int(params['stim_end']*net.taus[0])] = params['stim_level']



    if smooth_rates is not None:
        tau_samples = net.taus[0]
        if downsample is not None:
            tau_samples /= int(downsample)
        tau_ms = params['effective_tau']
        dt = tau_ms/float(tau_samples)
        
        kernel = spiketools.gaussian_kernel(sigma=smooth_rates,dt=dt,nstd=2.)[None,:]
        kernel/= kernel.sum()
    
    if return_mean_cluster_rates or return_max_cluster_rates:
        
            
        output = []
        for t in range(params['trials']):
            all_states = net.forward(input,return_state=True)
           
            cluster_states = []
            for i in range(len(n_bins)-1):
                cluster_states.append(all_states[n_bins[i]:n_bins[i+1]].mean(axis=0))
            if downsample is not None:
                downsample_factor = int(downsample)
                cluster_states = decimate(pylab.array(cluster_states), downsample_factor,axis=1)
            if smooth_rates is not None:
                cluster_states = convolve2d(cluster_states,kernel,'same')
                half_width = kernel.shape[1]//2
                cluster_states[:,:half_width] = pylab.nan
                cluster_states[:,-half_width:] = pylab.nan


            output.append(cluster_states)
        output = pylab.array(output)

        if return_mean_cluster_rates:
            if 'return_kernel' in params.keys() and params['return_kernel']:
                try:
                    return output,kernel
                except:
                    return output
            return output
        else:
            excitatory_output = output[:,:params['spec_args']['Q']]
            
            max_output = excitatory_output.max(axis = 2)
            
            return max_output
    
    

    elif return_cluster_rates_and_spiketimes:
        output = []
        spiketimes = pylab.zeros((3,0))
        for t in range(params['trials']):
            all_states = net.forward(input,return_state=True,record_updates = True)
            updates = pylab.zeros_like(all_states)
            for i,update in enumerate(net.update_record):
                updates[update,i] = 1
            spikes = (updates>0)*(all_states>0)
            units,times = pylab.where(spikes)
            new_spiketimes = pylab.concatenate((times[None,:],pylab.ones((1,len(units)))*t,units[None,:]),axis=0)
            spiketimes = pylab.append(spiketimes, new_spiketimes,axis=1)
            
            net.update_record = []

            cluster_states = []
            for i in range(len(n_bins)-1):
                cluster_states.append(all_states[n_bins[i]:n_bins[i+1]].mean(axis=0))
            if downsample is not None:
                downsample_factor = int(downsample)
                cluster_states = decimate(pylab.array(cluster_states), downsample_factor,axis=1)
            if smooth_rates is not None:
                cluster_states = convolve2d(cluster_states,kernel,'same')
            output.append(cluster_states)
        return pylab.array(output),spiketimes
    


    else:
        output = [net.forward(input,return_spiketimes = True) for t in range(params['trials'])]
        units = pylab.arange(net.N)

        all_spiketimes = pylab.ones((3,0))
        for trial,spiketimes in enumerate(output):
            trial_spiketimes = pylab.zeros((3,spiketimes.shape[1]))
            trial_spiketimes[0] = spiketimes[0]
            trial_spiketimes[1,:] = trial
            trial_spiketimes[2] = spiketimes[1]
            all_spiketimes = pylab.append(all_spiketimes,trial_spiketimes, axis=1)


        # convert to ms
        all_spiketimes[0] *= params['effective_tau']/float(net.taus[0])

        return all_spiketimes

def analyse(params):
    try:
        pylab.seed(params['randseed'])
    except:
        pylab.seed(None)

    


    print('simulating networks ...')
    t1 = clock()
    spiketimes = simulate(params)
    print('done simulating: ',clock()-t1,' s')
    t1  =clock()
    print('calculating statistics')
    ffs = []
    cvs = []
    counts = []
    
    spiketimes = spiketimes[:,spiketimes[2]<params['network_args']['Ns'][0]]
    unit_ffs =[]
    unit_cvs = []
    unit_counts = []
    for unit in pylab.unique(spiketimes[2]):
        unit_spiketimes = spiketimes[:2,spiketimes[2]==unit]
        unit_ffs.append(spiketools.ff(unit_spiketimes))
        unit_cvs.append(spiketools.cv2(unit_spiketimes,pool=False))
        unit_counts.append(pylab.isfinite(unit_spiketimes[0]).sum())
    ffs = pylab.nanmean(unit_ffs)
    cvs = pylab.nanmean(unit_cvs)
    counts = pylab.nanmean(unit_counts)
    print('done calculating',clock()-t1,' s')
    return ffs,cvs,counts

def analyse_time_resolved(params):
    spiketimes = organiser.check_and_execute(params['sim_params'], simulate, params['sim_datafile'])

    results = {'spiketimes':spiketimes}

    ffs = []
    cvs = []
    rates = []
    Tmax = params['sim_params']['input_length']*params['sim_params']['effective_tau']
    time_axis = pylab.arange(Tmax)
    tlim = [0,Tmax]
    kernel = spiketools.triangular_kernel(params['rate_window'])
    units = range(sum(params['sim_params']['network_args']['Ns']))
    for unit in units:
        print(unit, ' of ',max(units))
        unit_spiketimes = spiketimes[:2,spiketimes[2]==unit]
        if unit_spiketimes.shape[1]==0:
            unit_spiketimes = pylab.append(pylab.arange(params['sim_params']['trials'])[pylab.newaxis,:]*pylab.nan, pylab.arange(params['sim_params']['trials'])[pylab.newaxis,:],axis=0)
        ff,t_ff = spiketools.kernel_fano(unit_spiketimes, params['fano_window'],tlim = tlim)

        rate,t_rate = spiketools.kernel_rate(unit_spiketimes, kernel,tlim = tlim)
        rate = rate[0]
        cv2,t_cv2 =spiketools.rate_warped_analysis(unit_spiketimes, window=params['cv_ot'], step=0.5, tlim =tlim, func = spiketools.cv2, kwargs={'pool':False},rate = (rate,t_rate))
        cv2 = spiketools.resample(cv2, t_cv2, time_axis)
        ffs.append(ff)
        cvs.append(cv2)
        rates.append(rate)

    results['ffs'] = pylab.array(ffs)
    results['t_ff'] = t_ff
    results['cvs'] = pylab.array(cvs)
    results['t_cv'] = time_axis
    results['rates'] = pylab.array(rates)
    results['t_rate'] = t_rate
    results['tlim'] = tlim



    return results
