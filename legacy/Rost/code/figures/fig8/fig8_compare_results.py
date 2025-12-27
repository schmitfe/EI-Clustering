import sys
sys.path = ['../..' ] +sys.path +['../fig3'] 
import pylab
import numpy as np
from copy import deepcopy
import organiser
import plotting
import pickle
from sim_cluster_dynamics import simulate
organiser.datapath = '../../data'
datafile = 'fig8_data'


def do_plot(params,axes,ms = 1.,spike_alpha = 0.5,plot = True,return_rates = False,return_hist = True,n_jobs =1,hist_label ='',hist_color = 'k',hist_alpha = 0.5,log_hist = False,cmap = 'jet',state_file = '') :
    randseed = params.get('randseed',None)
    redo = params.pop('redo')
    jep = params['jep']
    jip = 1 + params['jip_ratio']*(jep-1)
    nbins = params.pop('rate_dist_bins')

    dist_reps = params.pop('rate_dist_reps')
    rate_threshold = params.pop('rate_threshold')
    smooth_rates = params.pop('smooth_rates')
    rate_dist_length = params.pop('rate_dist_length')
    plot_mode = params.pop('plot_mode')
    params['spec_args']['jplus'] = pylab.array([[jep,jip],[jip,jip]])
    params['return_cluster_rates_and_spiketimes'] = True
    params['return_max_cluster_rates'] = False
    params['return_mean_cluster_rates'] = False
    
    # print('getting spikes')
    # cluster_rates,spike_times = organiser.check_and_execute(params, simulate, datafile,redo  =redo)
    Q = params['spec_args']['Q']
    # cluster_rates = cluster_rates[0,:Q]
    # if smooth_rates is not None:
    #     params['smooth_rates'] =smooth_rates
    #     params['return_mean_cluster_rates'] = True
    #     params['return_kernel'] = True
    #     print('getting smoothed rates')
    #     smoothed_cluster_rates,kernel = organiser.check_and_execute(params, simulate, datafile,redo  =redo)
    #     params.pop('return_kernel')
    #     smoothed_cluster_rates = smoothed_cluster_rates[0,:Q]
    #
    #     print(kernel.shape)
    #     print(smoothed_cluster_rates.shape)
    #
    #
    #
    # if plot:
    #     # sort, so highest rate becomes darkest
    #     order = pylab.argsort(pylab.nanmax(cluster_rates,axis=1))[::-1]
    #
    #
    #     # if the highest cluster rate is at the start or end, swap out the indices so that
    #     # the spike trains don't occur at the edge of the plot
    #     print(order[0])
    #     if order[0] == 0 or order[0] == (Q-1):
    #         print('swapping indices')
    #
    #         swap_cluster = 1
    #         N_E = params['network_args']['Ns'][0]
    #         cluster_units = N_E/Q
    #         high_cluster_inds = pylab.arange(order[0]*cluster_units,(order[0]+1)*cluster_units)
    #         swap_cluster_inds = pylab.arange(swap_cluster*cluster_units,(swap_cluster+1)*cluster_units)
    #
    #         for new,old in zip(high_cluster_inds,swap_cluster_inds):
    #             new_inds = np.nonzero(np.ravel(spike_times[2,:]==new))
    #             old_inds = np.nonzero(np.ravel(spike_times[2,:]==old))
    #             spike_times[2,new_inds] = old
    #             spike_times[2,old_inds] = new
    #
    #
    #     colors = plotting.make_color_list(Q,cmap = cmap,maxval = 0.9)
    #     pylab.sca(axes[0])
    #     pylab.plot(spike_times[0],spike_times[2],'.k',markersize= ms,alpha = spike_alpha,rasterized = True)
    #
    #     pylab.sca(axes[1])
    #
    #
    #
    #
    #
    #     time = pylab.linspace(0,params['input_length']*params['effective_tau'], cluster_rates.shape[1])
    #     for i,q in enumerate(order):
    #         pylab.plot(time,cluster_rates[q],color = colors[i],alpha = 0.8)
    #         try:
    #             time = pylab.linspace(0,params['input_length']*params['effective_tau'], smoothed_cluster_rates.shape[1])
    #             #pylab.plot(time,smoothed_cluster_rates[q],'--',color = colors[i])
    #         except:
    #             pass
    #     pylab.ylim(0,1)
    
    
    global_randseed = params.get('global_randseed',None)
    pylab.seed(global_randseed)
    randoffset = pylab.randint(0,100000,1)
    
    rate_dist_randseeds = randoffset + pylab.arange(0,dist_reps)
    params['return_cluster_rates_and_spiketimes'] = False
    params['return_max_cluster_rates'] = False
    params['return_mean_cluster_rates'] = True
    
    hists = []
    all_rates = []
    bins = pylab.linspace(0,1, nbins)
    param_list = []
    for i,rs in enumerate(rate_dist_randseeds):
        #print i,rs
        param_list.append(deepcopy(params))
        param_list[-1]['randseed'] = rs
    print('getting repetitions of smoothed rates')
    results = organiser.check_and_execute_hetero(param_list, simulate, datafile,redo  =redo,n_jobs = n_jobs)

    for r in results:
        all_rates.append(r[0].copy())
   
    if randseed is not None:
        params['randseed'] = randseed
    else:
        params.pop('randseed')
    params.pop('return_max_cluster_rates')
    params['rate_dist_bins'] = nbins
    params['rate_dist_reps'] = dist_reps
    params['redo'] = redo
    params['rate_threshold'] = rate_threshold
    params['smooth_rates'] = smooth_rates
    params['rate_dist_length'] = rate_dist_length
    params['plot_mode'] = plot_mode
    all_rates = pylab.array(all_rates)
        
    plot_rates = []
    piece_length = int(all_rates.shape[2] * rate_dist_length/float(params['input_length']))
    start = 0
    while start < all_rates.shape[2]-1:
        
        end = start + piece_length
        
        if plot_mode == 'trial_max':
            plot_rates.append(pylab.nanmax(all_rates[:,:Q,start:end],axis=2)) 
        elif plot_mode == 'cluster_max':
            plot_rates.append(pylab.nanmax(all_rates[:,:Q,start:end],axis=1)) 
        elif plot_mode == 'all':
            plot_rates.append(all_rates[:,:Q,start:end]) 

        start = end

    plot_rates = pylab.array(plot_rates)

    
    flat_rates =plot_rates.flatten()
    
    flat_rates = flat_rates[pylab.isnan(flat_rates)==False]
    
    
    
    if plot:
        pylab.sca(axes[2])
        hist = pylab.hist(flat_rates,bins,density = True,histtype = 'stepfilled',label = hist_label,facecolor = hist_color,edgecolor = 'k',log = log_hist,linewidth = 1)[0]
        try:
            ymax = pylab.ylim()[1]
            with open(state_file, 'rb') as handle:
                states = pickle.load(handle)
            for k in states.keys():
                print(k)
                high_state = max(states[k])
                pylab.axvline(high_state,linestyle = '--',color = 'k',linewidth = 0.5)
                pylab.plot([high_state],[ymax],'ok',markersize= 4,markeredgewidth = 1)
            pylab.ylim(ymax = 1.05*ymax)
            pylab.xlim(-0.03,1.03)
        except:
            print('state file ',state_file,' could not be loaded')

    if return_hist:
        return bins,hist

    if return_rates:
        return all_rates

if __name__ == '__main__':
    n_jobs = 1
    Ns = [4000,1000]
    ps = pylab.array([[0.2,0.5],[0.5,0.5]])
    jep = 1
    ratio = 0.
    jip = 1 + (jep-1)*ratio
    color = False
    params = {'network_args':{'Ns':Ns,'ps':ps,
                             'Ts':pylab.ones(2),'g':1.2,'update_mode':'random',
                             'n_updates':1,'delta_T':0.},
              'spec_func':'EI_jplus',
              'spec_args':{'Q':20,'jplus':pylab.array([[jep,jip],[jip,jip]])},
              'stim_clusters':2,'init':0.1,'input_length':100,'trials':1,
              'stim_start':25,'stim_end':75,'stim_level':0.,'mx':0.03,'effective_tau':10.,
              'jxs':pylab.array([1,0.8])*(Ns[0]*ps[0,0])**0.5,
              'rate_dist_reps':5,#100,
              'rate_dist_bins':100,'global_randseed':1,'randseed':5,
               'smooth_rates': None,#'smooth_rates':75,
              'downsample':10,'rate_dist_length':100,'plot_mode':'cluster_max'}

    settings = [
                #{'jip_ratio':0.,'jep':2.9,'redo':False,'rate_threshold':0.0},
                #{'jip_ratio':0.75,'jep':4.0,'redo':False,'rate_threshold':0.0},
                {'jip_ratio': 0.75, 'jep': 4., 'redo': False, 'rate_threshold': 0.0, 'kappa': 1., 'simulation_type': 'new',
                'connection_type': 'bernoulli'},
                {'jip_ratio': 0.75, 'jep': 4., 'redo': False, 'rate_threshold': 0.0, 'kappa': 1., 'simulation_type': 'new', 'connection_type': 'poisson'}
                ]
    plot = True
    if plot:
        fig = plotting.nice_figure(fig_size_mm = [plotting.biol_cyb_fig_widths[2],plotting.biol_cyb_fig_widths[2]],backend = 'ps')
        ncols =2
        nrows = 3
        gs = pylab.GridSpec(nrows,ncols,top=0.95,bottom=0.1,hspace = 0.47,left = 0.1,right = 0.9,height_ratios = [2,1,1])
        subplotspec = gs.new_subplotspec((0,0), colspan=1,rowspan=1)
        ax1 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax1, 'a')

        subplotspec = gs.new_subplotspec((0,1), colspan=1,rowspan=1)
        ax2 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax2, 'b')

        subplotspec = gs.new_subplotspec((1,0), colspan=1,rowspan=1)
        ax3 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax3, 'c')

        subplotspec = gs.new_subplotspec((1,1), colspan=1,rowspan=1)
        ax4 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax4, 'd')

        subplotspec = gs.new_subplotspec((2,0), colspan=1,rowspan=1)
        ax5 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax5, 'e')

        subplotspec = gs.new_subplotspec((2,1), colspan=1,rowspan=1)
        ax6 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label(ax6, 'f')
        
        axes = [[ax1,ax3,ax5],[ax2,ax4,ax6]]
        
    else:
        axes = [[None]*3,[None]*3]

    hists = []
    hist_colors = ['0.5','0.5']
    hist_labels = ['$R_J = 0$','R_J = 3/4$']
    fixed_point_files = ['../fig4/cluster_states_2.9_0.pickle',
         '../fig4/cluster_states_4.0_0.75.pickle']
    for setno,setting in enumerate(settings):
        for k in setting.keys():
            params[k] = setting[k]
        params['return_mean_cluster_rates'] = False
        if color:
            cmap = 'jet'
        else:
            cmap = 'gray'
        rates = do_plot(params,axes[setno],plot=plot,return_rates=True,return_hist =False,n_jobs = n_jobs,hist_color = hist_colors[setno],hist_label = hist_labels[setno],
                        cmap = cmap,state_file = fixed_point_files[setno],log_hist = False)
    

    """
    pylab.figure()
    for bins,hist in hists:
        bins = 0.5*(bins[1:]+bins[:-1])
        pylab.plot(bins,hist)  
        """

    pylab.sca(ax1)
    pylab.ylabel('$unit$')
    pylab.yticks(range(0,5001,1000))
    xticks = pylab.xticks()
    pylab.xticks(pylab.linspace(0,xticks[0][-1],5),[0,250,500,750,1000])
    pylab.xlabel('$t [ms]$')
    pylab.sca(ax2)
    pylab.yticks(range(0,5001,1000),['']*6)
    pylab.xticks(pylab.linspace(0,xticks[0][-1],5),[0,250,500,750,1000])
    pylab.xlabel('$t [ms]$')

    pylab.sca(ax3)
    pylab.ylabel('$m_c$')
    yticks = pylab.yticks()
    pylab.ylim(-0.01,1.01)
    pylab.xticks([0,250,500,750,1000])
    pylab.xlabel('$t [ms]$')
    pylab.sca(ax4)
    pylab.xlabel('$t [ms]$')
    
    pylab.sca(ax5)
    pylab.yticks([])
    pylab.ylabel(r'$p\left( m_{c max} \right)$')
    pylab.xlabel('$m_{c max}$')

    pylab.sca(ax6)
    pylab.yticks([])
    pylab.xlabel('$m_{c max}$')
    

    figname = 'fig8'
    if color:
        figname += '_color'
    pylab.savefig(figname + '.pdf',dpi  =1200)
    pylab.savefig(figname + '.eps',dpi  =1200)
    #pylab.show()




