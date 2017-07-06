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

def input_output(neuron_params, SYN_POPS, RATES, COEFFS):
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=False)
    Proba = Proba_g_P(muV, sV, gV, 1e-3*neuron_params['Vthre'])
    Fout = firing_rate(muV, sV, gV, Tv, Proba, COEFFS)
    return Fout

def build_up_differential_operator_first_order(Model,
                                               KEY1='RecExc', KEY2='RecInh',
                                               KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
                                               KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5],
                                               T=5e-3):
    """
    simple first order system
    """

    Model['RATES'] = {}
    POP_STIM1 = [KEY1, KEY2]+KEY_RATES1
    POP_STIM2 = [KEY1, KEY2]+KEY_RATES2
    # neuronal and synaptic params
    neuron_params1 = built_up_neuron_params(Model, KEY1)
    SYN_POPS1 = build_up_afferent_synaptic_input(Model, POP_STIM1, KEY1)
    neuron_params2 = built_up_neuron_params(Model, KEY2)
    SYN_POPS2 = build_up_afferent_synaptic_input(Model, POP_STIM2, KEY2)
    
    # initialize rates that are static external parameters
    RATES1, RATES2 = {}, {}
    for f, pop in zip(VAL_RATES1, KEY_RATES1):
        RATES1['F_'+pop] = f
    for f, pop in zip(VAL_RATES2, KEY_RATES2):
        RATES2['F_'+pop] = f
        
    def TF1(X):
        RATES1['F_'+KEY1], RATES1['F_'+KEY2] = X[0], X[1] # two pop only
        return input_output(neuron_params1, SYN_POPS1, RATES1, Model['COEFFS_'+str(KEY1)])

    def TF2(X):
        RATES2['F_'+KEY1], RATES2['F_'+KEY2] = X[0], X[1] # two pop only
        return input_output(neuron_params2, SYN_POPS2, RATES2, Model['COEFFS_'+str(KEY2)])
    
    # the differential operator is an array of functions
    def A0(X):
        return 1./T*(TF1(X)-X[0])
    def A1(X):
        return 1./T*(TF2(X)-X[1])

    return [A0, A1]


def run_trajectory(Model,
                   X0 = [0.05, 100.],
                   KEY1='RecExc', KEY2='RecInh',
                   KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
                   KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5],
                   t = np.arange(2)*1e-10,
                   plot=False):
    
    Operator = build_up_differential_operator_first_order(Model, 
                                                          KEY1=KEY1, KEY2=KEY2,
                                                          KEY_RATES1 = KEY_RATES1, VAL_RATES1=VAL_RATES1,
                                                          KEY_RATES2 = KEY_RATES2, VAL_RATES2=VAL_RATES2)
    def dX_dt(X, t=0):
        return Operator[0](X), Operator[1](X)
    
    X = odeint(dX_dt, X0, t)         # we don't need infodict here
    
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
            neuron_params1, SYN_POPS1, RATES1, already_SI=False, with_Isyn=True)
    output['muV_'+KEY2], output['sV_'+KEY2],\
        output['gV_'+KEY2], output['Tv_'+KEY2],\
        output['Isyn_'+KEY2] = getting_statistical_properties(
            neuron_params2, SYN_POPS2, RATES2, already_SI=False, with_Isyn=True)
        
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
            KEY1='RecExc', KEY2='RecInh',
            KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
            KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5],
            X0 = [0.01, 0.01], t = np.arange(5000)*1e-5):

    X = run_trajectory(Model, X0=X0, t=t, 
                       KEY1=KEY1, KEY2=KEY2,
                       KEY_RATES1 = KEY_RATES1, VAL_RATES1=VAL_RATES1,
                       KEY_RATES2 = KEY_RATES2, VAL_RATES2=VAL_RATES2)

    return X.T[0][-1], X.T[1][-1]

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

    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import Model
    Model['COEFFS_RecExc'] = np.load('../../sparse_vs_balanced/data/COEFFS_RecExc.npy')
    Model['COEFFS_RecInh'] = np.load('../../sparse_vs_balanced/data/COEFFS_RecInh.npy')
    
    # show_phase_space(Model,
    #                  with_trajectory=[0.01, 0.01],
    #                  t = np.arange(5000)*1e-5, 
    #                  KEY1='RecExc', KEY2='RecInh',
    #                  KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
    #                  KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    X = find_fp(Model,
                t = np.arange(5000)*1e-5, 
                KEY1='RecExc', KEY2='RecInh',
                KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
                KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    
    # fig1, _, _ = find_fp(Model, plot=True)
    output = get_full_statistical_quantities(Model, X,
                                    KEY1='RecExc', KEY2='RecInh',
                                    KEY_RATES1 = ['AffExc'], VAL_RATES1=[15.],
                                    KEY_RATES2 = ['AffExc'], VAL_RATES2=[15.])
    print(output)
    
    # fig1.savefig('/Users/yzerlaut/Desktop/temp.svg')
    plt.show()
