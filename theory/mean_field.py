import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import firing_rate
from cells.cell_construct import built_up_neuron_params
from theory.tf import build_up_afferent_synaptic_input
import numpy as np
import matplotlib.pylab as plt
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import set_plot, Brown
from scipy.integrate import odeint

def input_output(neuron_params, SYN_POPS, RATES, COEFFS, already_SI=False):
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=already_SI)
    if already_SI:
        Vthre = neuron_params['Vthre']
    else:
        Vthre = 1e-3*neuron_params['Vthre']
        
    Proba = Proba_g_P(muV, sV, gV, Vthre)
    Fout = firing_rate(muV, sV, gV, Tv, Proba, COEFFS)
    return Fout

def solve_mean_field_first_order(Model,
                                 DYN_SYSTEM = {
                                     'RecExc': {'aff_pops':['AffExc']},
                                     'RecInh': {'aff_pops':['AffExc']}
                                 },
                                 t = np.arange(10),
                                 X0 = None,
                                 T=5e-3):

    DYN_KEYS = [key for key in DYN_SYSTEM.keys()] # for a quick access to the dynamical system variables

    # initialize neuronal and synaptic params
    for key in DYN_KEYS:
        DYN_SYSTEM[key]['nrn_params'] = built_up_neuron_params(Model, key)
        DYN_SYSTEM[key]['syn_input'] = build_up_afferent_synaptic_input(Model, DYN_KEYS+DYN_SYSTEM[key]['aff_pops'], key)
        
    
    # --- CONSTRUCT THE DIFFERENTIAL OPERATOR --- #
    def dX_dt(X, t, DYN_KEYS, DYN_SYSTEM):
        dX_dt = []
        for i, key0 in enumerate(DYN_KEYS):
            RATES = {}
            for x, key in zip(X, DYN_KEYS):
                RATES['F_'+key] = x
            for aff_key, func in zip(DYN_SYSTEM[key0]['aff_pops'],DYN_SYSTEM[key0]['aff_pops_funcs']):
                RATES['F_'+aff_key] = func(t)
            Fout = input_output(DYN_SYSTEM[key0]['nrn_params'], DYN_SYSTEM[key0]['syn_input'], RATES, Model['COEFFS_'+key0])
            dX_dt.append((Fout-X[i])/T)
        return dX_dt
    # ------------------------------------------- #

    if X0 is None:
        X0 = np.ones(len(DYN_KEYS))
        
    X = odeint(dX_dt, X0, t, args=(DYN_KEYS, DYN_SYSTEM))
    
    return X

def show_phase_space(Model,
                     KEY1='RecExc', KEY2='RecInh',
                     F1 = np.logspace(-2., 2.1, 100),
                     F2 = np.logspace(-2., 2.2, 400),
                     KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
                     KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5],
                     with_trajectory = [0.02, 10.],
                     t = np.arange(1000)*1e-5):

        
    Model['RATES'] = {}
    POP_STIM1 = [KEY1, KEY2]+KEY_RATES1
    POP_STIM2 = [KEY1, KEY2]+KEY_RATES2
    # neuronal and synaptic params
    neuron_params1 = built_up_neuron_params(Model, KEY1)
    SYN_POPS1 = build_up_afferent_synaptic_input(Model, POP_STIM1, KEY1)
    neuron_params2 = built_up_neuron_params(Model, KEY2)
    SYN_POPS2 = build_up_afferent_synaptic_input(Model, POP_STIM2, KEY2)
    
    # initialize rates
    RATES1, RATES2 = {}, {}
    for f, pop in zip(VAL_RATES1, KEY_RATES1):
        RATES1['F_'+pop] = f
    for f, pop in zip(VAL_RATES2, KEY_RATES2):
        RATES2['F_'+pop] = f
    
    # ---------------------------
    # computing the vector field
    # ---------------------------
    FE, FI = np.meshgrid(F1, F2)
    ZE, ZI = 0.*FE, 0.*FE
    F1_nullcline, F2_nullcline = 0.*F1, 0.*F1
    ZE = np.exp(-(FE-.1*FE.mean())**2)
    ZI = np.exp(-(FI-.1*FI.mean())**2)
    for kk in range(len(F1)):
        for ll in range(len(F2)):
            # pop 1
            RATES1['F_'+KEY1], RATES1['F_'+KEY2] = FE[ll, kk], FI[ll, kk]
            Fout1 = input_output(neuron_params1, SYN_POPS1, RATES1, Model['COEFFS_'+str(KEY1)])
            ZE[ll, kk] = Fout1-FE[ll, kk]
            # pop 2
            RATES2['F_'+KEY1], RATES2['F_'+KEY2] = FE[ll, kk], FI[ll, kk]
            Fout2 = input_output(neuron_params2, SYN_POPS2, RATES2, Model['COEFFS_'+str(KEY2)])
            ZI[ll, kk] = Fout2-FI[ll, kk]
        # ---------------------------
        # finding the nullclines, detected as change of sign of derivative
        # ---------------------------
        i0 = np.arange(len(F2)-1)[(np.sign(ZE[1:,kk])!=np.sign(ZE[:-1,kk]))]
        if len(i0)>0:
            F1_nullcline[kk] = F2[i0[0]-1]
        i0 = np.arange(len(F2)-1)[(np.sign(ZI[1:,kk])!=np.sign(ZI[:-1,kk]))]
        if len(i0)>0:
            F2_nullcline[kk] = F2[i0[0]-1]

    if with_trajectory is not None:
        X0 = with_trajectory
        X = run_trajectory(Model, X0=X0, t=t, 
                           KEY1=KEY1, KEY2=KEY2,
                           KEY_RATES1 = KEY_RATES1, VAL_RATES1=VAL_RATES1,
                           KEY_RATES2 = KEY_RATES2, VAL_RATES2=VAL_RATES2)
        f1_fp, f2_fp = X.T[0][-1], X.T[1][-1]

    # then we get the other quantities:
    RATES1['F_'+KEY1], RATES1['F_'+KEY2] = f1_fp, f2_fp
    RATES2['F_'+KEY1], RATES2['F_'+KEY2] = f1_fp, f2_fp
        
    output = {'F_'+KEY1:f1_fp, 'F_'+KEY2:f2_fp}
    output['muV_'+KEY1], output['sV_'+KEY1],\
        output['gV_'+KEY1], output['Tv_'+KEY1],\
        output['Isyn_'+KEY1] = getting_statistical_properties(
            neuron_params1, SYN_POPS1, RATES1, already_SI=False, with_Isyn=True, with_current_based=True)
    output['muV_'+KEY2], output['sV_'+KEY2],\
        output['gV_'+KEY2], output['Tv_'+KEY2],\
        output['Isyn_'+KEY2] = getting_statistical_properties(
            neuron_params2, SYN_POPS2, RATES2, already_SI=False, with_Isyn=True, with_current_based=True)
        
    # plot of phase space
    fig1, ax1 = plt.subplots(1, figsize=(4,3.3))
    plt.subplots_adjust(left=.2, bottom=.2)
    
    F1_nullcline[F1_nullcline==0], F2_nullcline[F2_nullcline==0] = np.nan, np.nan # masking 0 points for plotting
    ax1.streamplot(np.log(F1)/np.log(10), np.log(F2)/np.log(10), ZE, ZI, density=0.4, color='lightgray', linewidth=1)
    ax1.plot(np.log(F1)/np.log(10), np.log(F1_nullcline)/np.log(10), 'g-', lw=3, label=r'$\partial_t \nu_e=0$')
    ax1.plot(np.log(F1)/np.log(10), np.log(F2_nullcline)/np.log(10), 'r-', lw=3, label=r'$\partial_t \nu_i=0$')
    if len(i0)>0:
        ax1.plot([np.log(f1_fp)/np.log(10)],[np.log(f2_fp)/np.log(10)], 'ko', mfc='none', ms=10, label='stable FP')

    if with_trajectory is not None:
        ax1.plot(np.log(X.T[0])/np.log(10), np.log(X.T[1])/np.log(10), ':', color=Brown, label='trajectory')
    
    ax1.legend(loc='best', frameon=False, prop={'size':'small'})
    set_plot(ax1, xlabel='$\\nu_e$ (Hz)', ylabel='$\\nu_i$ (Hz)',
             yticks=[-1, 0, 1], yticks_labels=['0.1', '1', '10'],
             xticks=[-1, 0, 1], xticks_labels=['0.1', '1', '10'])
    return fig1, output


def find_fp(Model,
            DYN_SYSTEM = {
                'RecExc': {'aff_pops':['AffExc']},
                'RecInh': {'aff_pops':['AffExc']}
            },
            t = None,
            X0 = None,
            T=5e-3):
    if t is None:
        t = np.arange(5000.)*T/10.
    if X0 is None:
        X0 = np.ones(len(DYN_SYSTEM.keys()))
        
    X = solve_mean_field_first_order(Model, DYN_SYSTEM, t=t, X0=X0)

    return X[-1,:]

def get_full_statistical_quantities(Model, X,
                                    KEY1='RecExc', KEY2='RecInh',
                                    KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
                                    KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5]):

    Model['RATES'] = {}
    POP_STIM1 = [KEY1, KEY2]+KEY_RATES1
    POP_STIM2 = [KEY1, KEY2]+KEY_RATES2
    # neuronal and synaptic params
    neuron_params1 = built_up_neuron_params(Model, KEY1)
    SYN_POPS1 = build_up_afferent_synaptic_input(Model, POP_STIM1, KEY1)
    neuron_params2 = built_up_neuron_params(Model, KEY2)
    SYN_POPS2 = build_up_afferent_synaptic_input(Model, POP_STIM2, KEY2)
    
    # initialize rates that are static external parameters
    RATES1 = {'F_'+KEY1:X[0], 'F_'+KEY2:X[1]} # RATES
    RATES2 = {'F_'+KEY1:X[0], 'F_'+KEY2:X[1]} # RATES
    for f, pop in zip(VAL_RATES1, KEY_RATES1):
        RATES1['F_'+pop] = f
    for f, pop in zip(VAL_RATES2, KEY_RATES2):
        RATES2['F_'+pop] = f
        
    output = {'F_'+KEY1:X[0], 'F_'+KEY2:X[1]}
    output['muV_'+KEY1], output['sV_'+KEY1],\
        output['gV_'+KEY1], output['Tv_'+KEY1],\
        output['Isyn_'+KEY1] = getting_statistical_properties(
            neuron_params1, SYN_POPS1, RATES1, already_SI=False, with_Isyn=True)
    output['muV_'+KEY2], output['sV_'+KEY2],\
        output['gV_'+KEY2], output['Tv_'+KEY2],\
        output['Isyn_'+KEY2] = getting_statistical_properties(
            neuron_params2, SYN_POPS2, RATES2, already_SI=False, with_Isyn=True)
    
    return output

if __name__=='__main__':

    # show_phase_space(Model,
    #                  with_trajectory=[0.01, 0.01],
    #                  t = np.arange(5000)*1e-5, 
    #                  KEY1='RecExc', KEY2='RecInh',
    #                  KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
    #                  KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    # X = find_fp(Model,
    #             t = np.arange(5000)*1e-5, 
    #             KEY1='RecExc', KEY2='RecInh',
    #             KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
    #             KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    
    # fig1, _, _ = find_fp(Model, plot=True)
    # output = get_full_statistical_quantities(Model, X,
    #                                 KEY1='RecExc', KEY2='RecInh',
    #                                 KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
    #                                 KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    # print(output)
    
    # fig1.savefig('/Users/yzerlaut/Desktop/temp.svg')
    # plt.show()
    
    exec(open('../configs/The_Spectrum_of_Asynch_Dyn_2018/model.py').read())
    Model['COEFFS_RecExc'] = np.load('../configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecExc.npy')
    Model['COEFFS_RecInh'] = np.load('../configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecInh.npy')

    def func0(t):
        return 5.+nonstat*t*10.
    
    nonstat=0

    tstop, dt = 1, 5e-4

    DYN_SYSTEM = {
        'RecExc': {'aff_pops':['AffExc'], 'aff_pops_funcs':[func0]},
        'RecInh': {'aff_pops':['AffExc'], 'aff_pops_funcs':[func0]}
    }
    # be careful you'll need to label everything with respect to the DYN_KEYS (that change over different runs)
    print([key for key in DYN_SYSTEM.keys()])

    X0 = find_fp(Model, DYN_SYSTEM)
    print(X0)
    nonstat=0
    
    X = solve_mean_field_first_order(Model, DYN_SYSTEM, np.arange(int(tstop/dt))*dt, X0=X0)

    from graphs.my_graph import *

    fig, ax = plot(Y=[X[:,0], X[:,1]], COLORS=[Green, Red], fig_args={'figsize':(1.,.3)})
    show()
