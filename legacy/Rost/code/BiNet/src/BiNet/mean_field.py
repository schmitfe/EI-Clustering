import numbers
import numpy as np
from scipy.special import  erfc as erfc_wrong_scale
from scipy.special import erf
from scipy.optimize import bisect as fzero
from scipy.integrate import quad
from scipy.optimize import root as fsolve
from scipy.optimize import fminbound,minimize,fmin,fmin_tnc
import functools                                                                
import pickle 
from joblib import Parallel,delayed


small = 1e-30


                                                                  


def memoize(func):                                                          
    """                                                                         
    This is a caching decorator. It caches the function results for             
    all the arguments combinations, so use it with care. It does not matter      whether the arguments    
    are passed as keywords or not.                                              
    """                                                                         
    cache = {}                                                                  

    @functools.wraps(func)                                                      
    def cached(*args, **kwargs):                                                
        arg_names = func.__code__.co_varnames                                   
        arg_dict = {}                                                           
        for i, arg in enumerate(args):                                          
            arg_dict[arg_names[i]] = args[i]                                     
        arg_dict.update(**kwargs)                                                
        key = pickle.dumps(arg_dict, protocol=pickle.HIGHEST_PROTOCOL)                                            
        if key not in cache:                                                    
            cache[key] = func(*args, **kwargs)                                  
        return cache[key]                                                       
    return cached 


def condition(Ns,ps,g,jxs):
    if g >1:
        return jxs[0]/jxs[1] > g*np.sqrt(ps[0,0]/ps[1,0])
    elif g<1:
        return jxs[0]/jxs[1] < g*np.sqrt(ps[0,0]/ps[1,0])
    else:
        return False

def m_steady_state_theoretical_finite(Ns,ps,js,Ts,jxs,mx):
    assert len(Ns)==2, 'steady state rates only work for two-population models'
    J = _mean_weights(Ns,ps,js)
    #print J
    me = (Ts[1]*J[0,1] -Ts[0]*J[1,1] +mx*(J[1,1]*jxs[0]-J[0,1]*jxs[1]))/(J[0,1]*J[1,0]-J[0,0]*J[1,1])
    mi = (Ts[1]*J[0,0]-Ts[0]*J[1,0]+mx*(jxs[0]*J[1,0]-jxs[1]*J[0,0]))/(J[1,1]*J[0,0]-J[0,1]*J[1,0])
    
    return np.array([me,mi])

def m_steady_state_theoretical(Ns,ps,js,jxs,mx):
    assert len(Ns)==2, 'steady state rates only work for two-population models'
    J = _mean_weights(Ns,ps,js)
    me = mx * (jxs[1]*J[0,1] -jxs[0]*J[1,1])/(J[0,0]*J[1,1]-J[0,1]*J[1,0])
    mi = mx * (jxs[1]*J[0,0] -jxs[0]*J[1,0])/(J[0,1]*J[1,0]-J[0,0]*J[1,1])
    return np.array([me,mi])

def m_steady_state(ms,Ns,ps,js,Ts,taus,jxs,mx,delta_j = None,freeze = None,precission = 1e-15,constrain_equal =None):
    Ns = np.array(Ns)
    if isinstance(freeze, numbers.Integral):
        freeze = [freeze]
    elif freeze is None:
        freeze = []
    # if no constraints are given, each variable gets its own 
    if constrain_equal is None:
        constrain_equal = [[i] for i in range(len(ms))]
    
    
    frozen = np.zeros_like(ms).astype(bool)
    if freeze is not None:
        frozen[freeze] = True
    
    unique_ms = []
    unique_m_groups = []
    all_groups_to_solve = []
    for ce in constrain_equal:
        
        intersection =list(set(freeze).intersection(set(ce)))
        if len(intersection)>0:
            # if any variables have an equality constraint with a frozen one, they are also frozen to the same value
            frozen[ce] = True
            ms[ce] = ms[intersection[0]]
            if len(intersection)>1:
                print('warning: only using first frozen value')
        else:
            # otherwise they are set to the same value
            unique_ms.append(ms[ce].mean())
            unique_m_groups.append(ce)
            all_groups_to_solve += ce

    unique_ms = np.array(unique_ms)

    m_solutions = ms.copy()
    
    
    def minfunc(solve_ms):
        
        # heavily penalize out of bounds solutions
        if min(solve_ms)<0:
            distance = abs(min(solve_ms))
            return 10. + distance * 1e10
        if max(solve_ms)>1:
            distance = max(solve_ms)-1
            return 10. +distance * 1e10
        
        for m,inds in zip(solve_ms,unique_m_groups):
            m_solutions[inds] = m
        #print solve_ms,unique_m_groups
        #print m_solutions
        residual =  dm_dt(m_solutions,taus,js,ps,Ns,Ts,jxs,mx,delta_j=delta_j)
        return sum(residual[all_groups_to_solve]**2)


    def minjac(solve_ms):
        
        for m,inds in zip(solve_ms,unique_m_groups):
            m_solutions[inds] = m
        #print solve_ms,unique_m_groups
        #print m_solutions
        jac = jac_m(m_solutions,taus,js,ps,Ns,Ts,jxs,mx,delta_j=delta_j)
        # df/dm_b = sum(2(m_a)dm_a/dm_b)
        residual =  dm_dt(m_solutions,taus,js,ps,Ns,Ts,jxs,mx,delta_j=delta_j)
        df_dm =2*np.dot(jac,residual)
        
        # pick the rows that were solved for
        #print df_dm
        
        df_dm =   np.array([df_dm[g][0] for g in unique_m_groups])
        
        return df_dm
        
    
    
    if len(unique_ms)>0:
        
        
        result = fmin(minfunc,unique_ms,disp  =False,full_output = True,maxiter =100000,maxfun = 100000,xtol=precission,ftol=precission)#,full_output=True,xtol = precission,ftol=precission)#,bounds = [(0,1) for n in range(not_frozen)])#,method = 'slsqp',options={'ftol':1e-10})
        
        func_val = result[1]
        success = result[-1]==0
        result = result[0] 
        

        for m,inds in zip(result,unique_m_groups):
            m_solutions[inds] = m

        converged = (func_val< precission)
    else:
        converged  =True
    




       
    return m_solutions,converged


def EFR2D(Ns,ps,js,Ts,taus,jxs,mx,delta_j=None,N_ms = 1000,fix = [0,1],n_retry = 10,constrain_equal = None,reverse = False,precission  =1e-15,m_init = 0.5,bounds = [0,1],min_rate = 1e-10,n_jobs = 1):
    """ this is not (yet) strictly as 2d efr..."""
    Ns = np.array(Ns)
    n_pops = len(Ns)
    if reverse:
        m_in1 = np.linspace(bounds[1],bounds[0],N_ms)
        m_in2 = np.linspace(bounds[1],bounds[0],N_ms)
    else:
        m_in1 = np.linspace(bounds[0],bounds[1],N_ms)
        m_in2 = np.linspace(bounds[0],bounds[1],N_ms)


    m1_grid,m2_grid = np.meshgrid(m_in1,m_in2)
    m_grid = np.concatenate([m1_grid[:,:,None],m2_grid[:,:,None]],axis=2)
    # calculate the efrs for each row in parallel
    efrs = Parallel(n_jobs=n_jobs)(delayed(EFR)(Ns,ps,js,Ts,taus,jxs,mx,delta_j,fix = fix[0],n_retry = n_retry,\
                                         constrain_equal = constrain_equal,reverse = reverse,precission  =precission,
                                         m_init = m_init,passive_fix = [fix[1],m2],min_rate = min_rate,m_in = m_in1) for m2 in m_in2)

    # use only the non-focus rates from the efrs and then calculate the focus ones propperly
    m_others = np.zeros((N_ms,N_ms,n_pops))
    for i in range(N_ms):
        m_others[i,:,1:] = efrs[i][1].T
    m_others[:,:,:2] = m_grid
    

    m_out = np.zeros_like(m_grid)
    derivatives = np.zeros_like(m_grid)

    for i in range(N_ms):
        for j in range(N_ms):
            mu = _mu(js, ps, Ns, m_others[i,j], Ts, jxs, mx)
        
            s = _alpha(js, ps, Ns, m_others[i,j])
            m_out[i,j,:] = _m_k(mu,s)[fix]
            derivatives[i,j,:] = dm_dt(m_others[i,j], taus, js, ps, Ns, Ts, jxs, mx)[fix]


    return m_grid,m_others[:,:,2:],m_out,derivatives




    
    






def EFR(Ns,ps,js,Ts,taus,jxs,mx,delta_j=None,N_ms = 1000,fix = 0,n_retry = 10,constrain_equal = None,reverse = False,precission  =1e-15,m_init = 0.5,passive_fix = None,bounds = [0,1],min_rate = 1e-10,m_in = None):
    """ effective response function as in amit&mascaro 1999. 
        
        fix:             index of the focus population 
        constrain_equal: list of lists, populations in sublists are constraint to equal values
        passive_fix:     None or (pop,val). additional population to be fixed to val. mainly useful for 2d efr. 
        """
    Ns = np.array(Ns) 
    
    if m_in is None:
        if reverse:
            m_in = np.linspace(bounds[1],bounds[0],N_ms)
        else:
            m_in = np.linspace(bounds[0],bounds[1],N_ms)

    m_out = np.zeros_like(m_in)
    n_pops = len(Ns)
    m_between = np.zeros((n_pops-1,len(m_in)))

    if passive_fix is not None:
        all_fixes = [fix,passive_fix[0]]
    else:
        all_fixes = [fix]

    # the populations that are not fixed
    others = np.array([i for i in range(n_pops) if i not in all_fixes])
    
    # find a good starting point
    if np.isscalar(m_init):
        m_start = np.ones(n_pops)*m_init
        m_start,converged = m_steady_state(m_start,Ns,ps,js,Ts,taus,jxs,mx,delta_j,freeze =None,constrain_equal=constrain_equal,precission = 1e-5)
        if not converged:
            print('Fail')
            
    else:
        m_start = m_init

    
    for i ,m in enumerate(m_in):
        
        #m_start = np.ones(n_pops)*m
        m_start[fix] = m
        if passive_fix is not None:
            
            m_start[passive_fix[0]] = passive_fix[1]
        m_others,converged =  m_steady_state(m_start,Ns,ps,js,Ts,taus,jxs,mx,delta_j,freeze = all_fixes,constrain_equal=constrain_equal,precission = precission)
        
        if len(others)>0 and (not converged or (m_others[others]<min_rate).any()):

            print('not converged for ', m)
            # failed to converge
            # try some other starting values
            m_tries = np.linspace(0.02,0.98, n_retry)
            for j in range(n_retry):
                m_start = (np.random.rand(n_pops)-0.5)*0.01+m_tries[j]
                m_start[m_start>=0.99] = 0.99
                m_start[m_start<=0.01] = 0.01
                m_start[fix] = m
                if passive_fix is not None:
                    m_start[passive_fix[0]] = passive_fix[1]
                m_others,converged =  m_steady_state(m_start,Ns,ps,js,Ts,taus,jxs,mx,delta_j,freeze = all_fixes,constrain_equal=constrain_equal,precission = precission)
                print(j, converged)
                if converged and (m_others[others]>min_rate).all(): # zero means trouble...
                    break
            if not converged:
                
                m_out[i] = np.nan 
                continue
        # remember the other values
        if len(others)>0:
            m_between[others-1,i] = m_others[others]
        if passive_fix is not None:
            
            m_between[passive_fix[0]-1,i] = m_others[passive_fix[0]]
        mu = _mu(js, ps, Ns, m_others, Ts, jxs, mx)
        
        s = _alpha(js, ps, Ns, m_others)
             
        m_out_vec = _m_k(mu,s)
        m_out[i] = m_out_vec[fix]
        if not np.isnan(m_out_vec).any():
            m_start = m_out_vec.copy()
        else:
            m_start = np.ones(n_pops)*m
    if reverse:
        m_in = m_in[::-1]
        m_bewteen = m_between[:,::-1]
        m_out = m_out[::-1]
    return m_in,m_between,m_out




def _mean_weights(Ns,ps,js):
    
    return ps * Ns[np.newaxis,:] *js /np.sqrt(Ns.sum())

def _weight_vars(Ns,ps,js,delta_j=0.):
    Js = js/np.sqrt(sum(Ns))
    if delta_j is None:
        delta_j = 0.
    if np.isscalar(delta_j):
        delta_j =np.ones_like(ps)*delta_j
    delta_j =np.absolute(Js*delta_j)

    var_p = ps*(1-ps)
    #var_p = ps
    e_p = ps
    var_j = delta_j**2 / 12.
    e_j = Js


    
    return (var_p*var_j + var_p*e_j**2 + var_j * e_p**2)*Ns[np.newaxis,:]

def _mu(js,ps,Ns,ms,Ts,jxs,mx):
    mw = _mean_weights(Ns,ps,js)
    mu =  np.dot(_mean_weights(Ns,ps,js),ms) -Ts +jxs*mx

    #print 'mu', Ns,ps,js,ms,Ts,jxs,mx,mu
    return mu
    

def _alpha(js,ps,Ns,ms,delta_j = None):
    mv = _weight_vars(Ns,ps,js,delta_j)
    return np.dot(_weight_vars(Ns,ps,js,delta_j),ms)+small

def _beta(js,ps,Ns,qs,delta_j =None):
    
    return np.dot(_weight_vars(Ns,ps,js,delta_j),qs)

@memoize
def _q_k(js,ps,Ns,ms,Ts,jxs,mx,delta_j = None):
    mu = _mu(js,ps,Ns,ms,Ts,jxs,mx)
    alpha = _alpha(js,ps,Ns,ms,delta_j)
    
    #print mu,alpha

    def func(qs):

        beta = _beta(js,ps,Ns,qs,delta_j)
        #print alpha,beta,qs
        res = np.zeros_like(qs)
        for i in range(len(qs)):
            def intfunc(x):
                val =np.exp(-x**2/2.)/np.sqrt(2*np.pi) * erfc((-mu[i]+np.sqrt(beta[i])*x)/np.sqrt(alpha[i]-beta[i]))**2
                return val
            res[i] = quad(intfunc,-100,100)[0]
        return res - qs

    return fsolve(func, ms**2,method='hybr')

@memoize
def _mk_x(x,js,ps,Ns,ms,Ts,jxs,mx,delta_j = None):
    mu = _mu(js,ps,Ns,ms,Ts,jxs,mx)
    alpha = _alpha(js,ps,Ns,ms,delta_j)
    qs = _q_k(js,ps,Ns,ms,Ts,jxs,mx,delta_j).x
    beta = _beta(js,ps,Ns,qs,delta_j)
    

    return erfc((-mu[:,None]+np.sqrt(beta[:,None])*x)/np.sqrt(alpha-beta)[:,None])

def _rho_k(m,js,ps,Ns,ms,Ts,jxs,mx,delta_j = None,steps = 10000):
    rhos = np.zeros_like(ms)
    xs = np.linspace(-20,20,steps)[None,:]
    delta_x = np.diff(xs[0,:])[0]
    
    mk_xs = _mk_x(xs,js,ps,Ns,ms,Ts,jxs,mx,delta_j)
    
    for k in range(len(rhos)):
        index = np.argmin(np.absolute(mk_xs[k,:]-m))
        x = xs[0,index]
        #print mk_xs.shape
        #print m,mk_xs[k,index]
        rhos = np.exp(-x**2/2.)/np.sqrt(2*np.pi)* delta_x
    return rhos

def rate_dists(js,ps,Ns,ms,Ts,jxs,mx,delta_j=None,bins=100,steps = 10000):
    m_range = np.linspace(0,1,bins)
    delta_m = m_range[1]-m_range[0]
    dists = np.zeros((len(Ns),len(m_range)))
    for i,m in enumerate(m_range):
        dists[:,i] = _rho_k(m, js, ps, Ns, ms, Ts, jxs, mx,delta_j,steps = steps)

    dists /= dists.sum(axis=1)[:,None]*delta_m
    
    return dists,m_range

def _m_k(mu,s,delta_theta=None):

    if delta_theta is None:

        return erfc(- mu /(np.sqrt(s+small)))
        
    else:
        # for inhonogeneous thresholds, the erfc needs to be integrated over the distribution...
        if np.isscalar(delta_theta):
            delta_theta = np.ones(mu.shape[0])*delta_theta

        mks = np.zeros(len(mu))
        for i in range(mks.shape[0]):
            if delta_theta[i]>0:
                def func(theta):
                    return erfc((-mu[i]+theta)/(np.sqrt(s[i])+small))
                mks[i] = quad(func,-0.5*delta_theta[i],0.5*delta_theta[i])[0]/delta_theta[i]
            else:
                mks[i] = erfc(- mu[i] /(np.sqrt(s[i])+small))
        return mks

def dm_dt(ms,taus,js,ps,Ns,Ts,jxs,mx,delta_theta=None,delta_j = None):
    mu = _mu(js,ps,Ns,ms,Ts,jxs,mx)

    s = _alpha(js,ps,Ns,ms,delta_j)
    m_k = _m_k(mu,s,delta_theta)
    return -(ms -m_k)/taus

def jac_m(ms,taus,js,ps,Ns,Ts,jxs,mx,delta_j = None):
    Ns = np.array(Ns)
    mean_weights = _mean_weights(Ns,ps,js)
    weight_vars = _weight_vars(Ns,ps,js,delta_j)
    mu = _mu(js,ps,Ns,ms,Ts,jxs,mx)
    s = _alpha(js,ps,Ns,ms,delta_j)**0.5
    jac = np.zeros_like(ps)
    for alpha in range(len(ms)):
        for beta in range(len(ms)):
            jac[alpha,beta] = -Derfc(-mu[alpha]/s[alpha])*(mean_weights[alpha,beta]*s[alpha] -mu[alpha] * weight_vars[alpha,beta]/(2*s[alpha]))/s[alpha]**2
    jac-= np.eye(len(ms))       
    jac /= taus[:,np.newaxis]
    
    return jac


def erfc(x):
    return (1-erf(x/np.sqrt(2)))/2.
    #return erfc_wrong_scale(x/np.sqrt(2))/2.

def Derfc(x):
    
    return -np.exp(-x**2/2.)/np.sqrt(2*np.pi)

def arrow_field(func,n_arrows):
    mes = np.linspace(0,1,n_arrows)
    mis = np.linspace(0,1,n_arrows)
    grid  = np.array(np.meshgrid(mes,mis)) 
    deltas = np.zeros_like(grid)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            deltas[:,i,j] = func(grid[:,i,j])
    
    return grid,deltas
  
def nullclines(func,n_points=1000):
    grid= np.linspace(0,1,n_points)
    ncs = []
    for g in grid:
        gncs = []
        for k in [0,1]:
            opt_func = lambda x: func(np.array([g,x]))[k]
            try:
                opt = fzero(opt_func,0.,1.,disp = True)
            except:
                opt = np.nan
            gncs.append( opt)
        ncs.append(gncs)
    return grid,np.array(ncs)
        
def integrate(func,m0,T,dt=0.1):
    steps = int(T/dt)
    ms= [m0]
    for step in range(steps):
        ms.append(ms[-1]+dt*func(ms[-1]))
    ms = np.array(ms).T
    return ms

def integrate_to_convergence(func,m0,maxit=1000,delta = 1e-5,verbose =False,dt=0.1,return_full = False,solve =False,
                             jac=None,solver_method='hybr',solver_options = {'maxfev':10000}):
    
    converged = False
    ms=[m0]
    for iteration in range(maxit):
        new_m = integrate(func,ms[-1],dt,dt)[:,-1]
        residual = ((ms[-1]-new_m)**2).sum()**0.5
        ms.append(new_m)
        if verbose:
            print('iteration ', iteration + 1, ' residual: ', residual)
        if residual <delta:
            converged = True
            break

    if verbose:
        if converged:
            print('converged to residual=', residual, ' after ', iteration + 1, 'steps')
        else:
            print('not converged: residual=', residual)
    if solve:
        if jac is None:
            result =fsolve(func,new_m,method=solver_method,options =solver_options)
        else:
            result = fsolve(func,new_m,jac=jac,method=solver_method,options =solver_options)
        if verbose:
            print(result)
        new_m = result['x']
        converged = result['success']
        ms.append(new_m)

    if return_full:
        return np.array(ms).T,iteration,converged

    return new_m,converged







def tune_jxs(m_target,Ns,ps,js,Ts,taus,jxs,mx,m_precission = 1e-15,jx_precission= 0.01):
    #m_steady_state(ms,Ns,ps,js,Ts,taus,jxs,mx,delta_j = None,freeze = None,,constrain_equal =None):
    m_target =np.array(m_target)
    
    def minfunc(solve_jxs):
        ms,converged= m_steady_state(m_target,Ns,ps,js,Ts,taus,solve_jxs,mx,precission=m_precission)
        return ((ms-m_target)**2).sum()

    result = fmin(minfunc,jxs,disp  =False,full_output = True,maxiter =100000,maxfun = 100000,xtol=jx_precission,ftol=m_precission)
    
    return np.array(result[0])
