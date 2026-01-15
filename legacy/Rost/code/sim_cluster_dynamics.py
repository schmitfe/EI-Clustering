import numpy as np
from BiNet import network, mean_field, weights
import organiser
import spiketools
import pylab
import os
from copy import deepcopy
from time import time as clock
from scipy.signal import decimate, convolve2d


def _compute_state_sample_indices(length, stride):
    stride = max(1, int(stride))
    if length <= 0:
        return pylab.zeros(0, dtype=int)
    start = min(stride - 1, max(length - 1, 0))
    return pylab.arange(start, length, stride)


def simulate(original_params):
    params = deepcopy(original_params)
    try:
        pylab.seed(params['randseed'])
    except:
        pylab.seed(None)

    simulation_type = str(params.get('simulation_type', 'new')).lower()
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

    #if params['spec_func'] == 'mazzucato':
    #    spec_func = weights.mazzucato_cluster_specs
    #elif params['spec_func'] == 'doiron':
    #    spec_func = weights.doiron_cluster_specs
    #elif params['spec_func'] == 'EI_jplus':
    spec_func = weights.EI_jplus_cluster_specs

    return_mean_cluster_rates = bool(params.get('return_mean_cluster_rates', False))
    return_max_cluster_rates = bool(params.get('return_max_cluster_rates', False))
    return_cluster_rates_and_spiketimes = bool(params.get('return_cluster_rates_and_spiketimes', False))
    return_state_dynamics = bool(params.get('return_state_dynamics', False))
    smooth_rates = params.get('smooth_rates', None)  # std of gaussian smoothing kernel in ms
    downsample = params.get('downsample', None)
    state_record_stride = int(params.get('state_record_stride', 1) or 1)

    net = network.BalancedNetwork(**params['network_args'])

    spec_args = params['spec_args']
    for k in ['Ns', 'ps', 'Ts', 'g']:
        spec_args[k] = params['network_args'][k]
    spec_args['taus'] = net.taus / net.taus[0] * params['effective_tau']
    if simulation_type == 'new':
        spec_args['kappa'] = kappa

    cNs, cps, cjs, cTs, ctaus = spec_func(**spec_args)

    w = weights.generate_weight_matrix(cNs, cps, cjs, delta_j=None, connection_type=connection_type)
    net.set_weights(w)

    w_in = pylab.ones((net.N, 2))
    w_in[:net.Ns[0], 0] = params['jxs'][0]
    w_in[net.Ns[0]:, 0] = params['jxs'][1]
    stim_clusters = list(range(len(cNs) - 2))
    pylab.shuffle(stim_clusters)
    stim_clusters = stim_clusters[:params['stim_clusters']]
    n_bins = [0] + pylab.cumsum(cNs).tolist()
    for stim_cluster in stim_clusters:
        inds = list(range(n_bins[stim_cluster], n_bins[stim_cluster + 1]))
        w_in[inds, 1] = params['jxs'][0]

    net.set_input_weights(w_in)
    net.initialise_state(params['init'])

    input = pylab.ones((2, int(params['input_length'] * net.taus[0]))) * params['mx']
    input[1, :] *= 0
    input[1, int(params['stim_start'] * net.taus[0]):int(params['stim_end'] * net.taus[0])] = params['stim_level']



    kernel = None
    if smooth_rates is not None:
        tau_samples = net.taus[0]
        if downsample is not None:
            tau_samples /= int(downsample)
        tau_ms = params['effective_tau']
        dt = tau_ms / float(tau_samples)
        kernel = spiketools.gaussian_kernel(sigma=smooth_rates, dt=dt, nstd=2.)[None, :]
        kernel /= kernel.sum()



    needs_cluster_rates = any(
        (
            return_mean_cluster_rates,
            return_max_cluster_rates,
            return_cluster_rates_and_spiketimes,
            return_state_dynamics,
        )
    )
    print(f"return_state_dynamics: {return_state_dynamics}, other rates: {needs_cluster_rates}")
    print(f"downsample: {downsample}")
    record_spike_times = bool(return_cluster_rates_and_spiketimes)
    result = {}
    if needs_cluster_rates:
        cluster_outputs = []
        sampled_states = []
        sampled_fields = []
        state_indices = None
        spiketimes = pylab.zeros((3, 0)) if record_spike_times else None
        for trial in range(params['trials']):
            forward_output = net.forward(
                input,
                return_state=True,
                record_subthreshold=False,
                record_updates=False,
                return_spiketimes=record_spike_times,
            )
            spike_log = None
            state_packet = forward_output
            if record_spike_times:
                state_packet, spike_log = forward_output
                if spike_log is not None and spike_log.size:
                    trial_block = pylab.zeros((3, spike_log.shape[1]))
                    trial_block[0] = spike_log[0]
                    trial_block[1] = trial
                    trial_block[2] = spike_log[1]
                    spiketimes = pylab.append(spiketimes, trial_block, axis=1)
            init_state, update_log, delta_log = state_packet
            all_states = net._reconstruct_states(init_state, update_log, delta_log)
            cluster_states = []
            for i in range(len(n_bins) - 1):
                cluster_states.append(all_states[n_bins[i]:n_bins[i + 1]].mean(axis=0))
            cluster_states = pylab.array(cluster_states)
            if downsample is not None:
                downsample_factor = int(downsample)
                cluster_states = decimate(cluster_states, downsample_factor, axis=1)
            if kernel is not None:
                cluster_states = convolve2d(cluster_states, kernel, 'same')
                half_width = kernel.shape[1] // 2
                if half_width > 0:
                    cluster_states[:, :half_width] = pylab.nan
                    cluster_states[:, -half_width:] = pylab.nan
            cluster_outputs.append(cluster_states)
            if return_state_dynamics:
                indices = _compute_state_sample_indices(all_states.shape[1], state_record_stride)
                if indices.size == 0 and all_states.shape[1] > 0:
                    indices = pylab.array([all_states.shape[1] - 1], dtype=int)
                state_indices = indices
                state_slice = all_states[:, indices]
                sampled_states.append(state_slice)
                input_slice = input[:, indices]
                fields = net.compute_fields_from_states(state_slice, input_slice)
                sampled_fields.append(fields)
        cluster_outputs = pylab.array(cluster_outputs)
        result["cluster_rates"] = cluster_outputs
        if return_mean_cluster_rates:
            result["mean_cluster_rates"] = cluster_outputs
        if params.get('return_kernel') and kernel is not None:
            result["kernel"] = kernel
        if return_max_cluster_rates:
            excitatory_output = cluster_outputs[:, :params['spec_args']['Q']]
            result["max_cluster_rates"] = excitatory_output.max(axis=2)
        if record_spike_times and spiketimes is not None:
            result["spike_times"] = spiketimes
            updates = getattr(net, "last_update_log", None)
            deltas = getattr(net, "last_delta_log", None)
            if updates is not None and deltas is not None:
                result["state_updates"] = np.array(updates, dtype=np.uint16, copy=True, order="F")
                result["state_deltas"] = np.array(deltas, dtype=np.int8, copy=True, order="F")
                init_state = getattr(net, "reference_state", net.state)
                result["initial_state"] = np.asarray(init_state, dtype=np.uint8)
        if return_state_dynamics:
            result["sampled_states"] = pylab.array(sampled_states)
            result["sampled_fields"] = pylab.array(sampled_fields)
            result["state_indices"] = state_indices if state_indices is not None else pylab.zeros(0, dtype=int)
        return result

    output = [net.forward(input, return_spiketimes=True) for _ in range(params['trials'])]
    all_spiketimes = pylab.ones((3, 0))
    for trial, spiketimes in enumerate(output):
        trial_spiketimes = pylab.zeros((3, spiketimes.shape[1]))
        trial_spiketimes[0] = spiketimes[0]
        trial_spiketimes[1, :] = trial
        trial_spiketimes[2] = spiketimes[1]
        all_spiketimes = pylab.append(all_spiketimes, trial_spiketimes, axis=1)
    all_spiketimes[0] *= params['effective_tau'] / float(net.taus[0])
    result["spike_times"] = all_spiketimes
    return result

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
