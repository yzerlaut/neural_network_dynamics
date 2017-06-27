import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import firing_rate
from cells.cell_construct import built_up_neuron_params
from theory.tf import build_up_afferent_synaptic_input
import numpy as np
import matplotlib.pylab as plt
from graphs.my_graph import set_plot

def input_output(neuron_params, SYN_POPS, RATES, COEFFS):
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=False)
    Proba = Proba_g_P(muV, sV, gV, 1e-3*neuron_params['Vthre'])
    Fout = firing_rate(muV, sV, gV, Tv, Proba, COEFFS)
    return Fout

def find_fp(Model,
            KEY1='RecExc', KEY2='RecInh',
            F1 = np.logspace(-2., 2.1, 100),
            F2 = np.logspace(-2., 2.2, 400),
            KEY_RATES1 = ['AffExc'], VAL_RATES1=[4.],
            KEY_RATES2 = ['AffExc', 'DsInh'], VAL_RATES2=[4., .5],
            plot=False):

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

    # find the fixed point: crossing of the two nullclines !
    # i0 = np.argwhere((F1_nullcline[:-1]>F2_nullcline[:-1]) &\
    #                  (F2_nullcline[1:]>F1_nullcline[1:]) &\
    #                  (F1_nullcline[:-1]>0) & (F2_nullcline[:-1]>0)).flatten()
    i0 = np.argwhere((F2_nullcline>F1_nullcline)).flatten()
    
    if (len(i0)==0) or (i0[0]==0):
        # no nullcline crossing, no dynamical system solution of recurrent system, firing is just effect of input
        RATES1['F_'+KEY1], RATES1['F_'+KEY2] = 0, 0
        f1_fp = input_output(neuron_params1, SYN_POPS1, RATES1, Model['COEFFS_'+str(KEY1)])
        RATES2['F_'+KEY1], RATES1['F_'+KEY2] = 0, 0
        f2_fp = input_output(neuron_params2, SYN_POPS2, RATES2, Model['COEFFS_'+str(KEY2)])
    else:
        f1_fp, f2_fp = F1[i0[0]], F2_nullcline[i0[0]]

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
    if plot:
        F1_nullcline[F1_nullcline==0], F2_nullcline[F2_nullcline==0] = np.nan, np.nan # masking 0 points for plotting
        fig1, ax1 = plt.subplots(1, figsize=(4,3.3))
        plt.subplots_adjust(left=.2, bottom=.2)
        ax1.streamplot(np.log(F1)/np.log(10), np.log(F2)/np.log(10), ZE, ZI, density=0.4, color='lightgray', linewidth=1)
        ax1.plot(np.log(F1)/np.log(10), np.log(F1_nullcline)/np.log(10), 'g-', lw=3, label=r'$\partial_t \nu_e=0$')
        ax1.plot(np.log(F1)/np.log(10), np.log(F2_nullcline)/np.log(10), 'r-', lw=3, label=r'$\partial_t \nu_i=0$')
        if len(i0)>0:
            ax1.plot([np.log(f1_fp)/np.log(10)],[np.log(f2_fp)/np.log(10)], 'ko', mfc='none', ms=10, label='stable FP')
        ax1.legend(loc='best', frameon=False, prop={'size':'small'})
        set_plot(ax1, xlabel='$\\nu_e$ (Hz)', ylabel='$\\nu_i$ (Hz)',
                 yticks=[-1, 0, 1], yticks_labels=['0.1', '1', '10'],
                 xticks=[-1, 0, 1], xticks_labels=['0.1', '1', '10'])
        return fig1, output
    else:
        return output

if __name__=='__main__':

    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import Model
    
    from neural_network_dynamics.theory.fitting_tf import fit_data
    exc_data = np.load('../../params_scan/data_exc1.npy').item()
    Model['COEFFS_RecExc'] = fit_data(exc_data, order=2)
    inh_data = np.load('../../params_scan/data_inh0.npy').item()
    Model['COEFFS_RecInh'] = fit_data(inh_data, order=1)
    fig1, _, _ = find_fp(Model, plot=True)
    # fig1.savefig('/Users/yzerlaut/Desktop/temp.svg')
    plt.show()
