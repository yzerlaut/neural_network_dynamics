import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import firing_rate
from cells.cell_construct import built_up_neuron_params
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def build_up_afferent_synaptic_input(Model, POP_STIM, NRN_KEY=None):

    if NRN_KEY is None:
        NRN_KEY = Model['NRN_KEY']
    
    SYN_POPS = []
    for source_pop in POP_STIM:
        if len(source_pop.split('Exc'))>1:
            Erev, Ts = Model['Ee'], Model['Tse']
        elif len(source_pop.split('Inh'))>1:
            Erev, Ts = Model['Ei'], Model['Tsi']
        else:
            print(' /!\ AFFERENT POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev, Ts = Model['Ee'], Model['Tse']
        SYN_POPS.append({'name':source_pop, 'Erev': Erev, 'N': Model['N_'+source_pop],
                         'Q': Model['Q_'+source_pop+'_'+NRN_KEY],
                         'pconn': Model['p_'+source_pop+'_'+NRN_KEY],
                         'Tsyn': Ts})
    return SYN_POPS
    
def TF(RATES, Model, NRN_KEY):

    neuron_params = built_up_neuron_params(Model, NRN_KEY)
    SYN_POPS = build_up_afferent_synaptic_input(Model, Model['POP_STIM'], NRN_KEY)
    ### OUTPUT OF ANALYTICAL CALCULUS IN SI UNITS !! -> from here SI, be careful...
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=False)
    Proba = Proba_g_P(muV, sV, gV, 1e-3*neuron_params['Vthre'])
    
    Fout_th = firing_rate(muV, sV, gV, Tv, Proba, Model['COEFFS'])

    return Fout_th


def make_tf_plot(data,
                 xkey='F_RecExc', ckey='F_RecInh', output_key='Fout',
                 ckey_label='$\\nu_{i}$ (Hz)',
                 col_key = 'F_AffExc', col_key_label = '$\\nu_a$', col_key_unit = 'Hz', col_subsmpl=None,
                 row_key = 'F_DsInh', row_key_label = '$\\nu_d$', row_key_unit = 'Hz', row_subsmpl=None,
                 ylim=[1e-2, 100], yticks=[0.01, 0.1, 1, 10], yticks_labels=['0.01', '0.1', '1', '10'], ylabel='$\\nu_{out}$ (Hz)',
                 xticks=[0.1, 1, 10], xticks_labels=['0.1', '1', '10'], xlabel='$\\nu_{e}$ (Hz)',
                 logscale=True, cmap=cm.copper,
                 with_theory=False, th_discret=20):
    
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from graphs.my_graph import set_plot, build_bar_legend
    import matplotlib.pylab as plt

    # limiting the data within the range
    Fout_mean, Fout_std = data[output_key+'_mean'], data[output_key+'_std']

    if col_subsmpl is None:
        col_subsmpl = np.arange(len(np.unique(data[col_key])))
    if row_subsmpl is None:
        row_subsmpl = np.arange(len(np.unique(data[row_key])))
        print(row_subsmpl)
    def make_row_fig(cond, AX, with_top_label=False, fd=0):
        F1, Fe, Fi = data[col_key][cond], data[xkey][cond], data[ckey][cond]
        mFout, sFout = Fout_mean[cond], Fout_std[cond]
        for i, f1 in enumerate(np.unique(F1)[col_subsmpl]):
            i0 = np.argwhere(F1==f1).flatten()
            for j, fi in enumerate(np.unique(Fi[i0])):
                i1 = np.argwhere((Fi[i0]==fi)).flatten()
                cond2 =  (mFout[i0][i1]>=ylim[0]) & (mFout[i0][i1]<ylim[1])
                AX[i].errorbar(Fe[i0][i1][cond2],
                               Fout_mean[i0][i1][cond2],
                               yerr=Fout_std[i0][i1][cond2],
                               fmt='o', ms=4,
                               color=cmap(j/len(np.unique(Fi[i0]))))
                # # now analytical estimate
                if with_theory:
                    RATES = {xkey:np.concatenate([np.linspace(f1, f2, th_discret, endpoint=False)\
                                                       for f1, f2 in zip(Fe[i0][i1][:-1], Fe[i0][i1][1:])])}
                    for pop, f in zip([col_key, ckey, row_key],[f1, fi, fd]) :
                        RATES[pop] = f*np.ones(len(RATES[xkey]))
                    Fout_th = TF(RATES, data['Model'], data['Model']['NRN_KEY'])
                    th_cond = (Fout_th>ylim[0]) & (Fout_th<ylim[1])
                    AX[i].plot(RATES[xkey][th_cond],
                               Fout_th[th_cond], '-',
                               color=cmap(j/len(np.unique(Fi[i0]))), lw=3, alpha=.8)
            if with_top_label:
                AX[i].set_title(col_key_label+'='+str(round(f1,1))+col_key_unit)
            if logscale:
                AX[i].set_yscale('log')
                AX[i].set_xscale('log')
            ylabel2, yticks_labels2 = '', []
            if i==0: # if first column
                ylabel2, yticks_labels2 = ylabel, yticks_labels
            set_plot(AX[i], ylim=[.9*ylim[0], ylim[1]],
                     yticks=yticks, xticks=xticks, ylabel=ylabel2, xlabel=xlabel,
                     yticks_labels=yticks_labels2, xticks_labels=xticks_labels)
        Fi = Fi
        Fi_log = np.log(Fi)/np.log(10)
        cax = inset_axes(AX[-1], width="20%", height="90%", loc=3)
        cb = build_bar_legend(np.unique(Fi_log), cax , cmap,
                          ticks_labels=[str(round(fi,1)) for fi in np.unique(Fi)],
                              label=ckey_label)
        AX[-1].axis('off')

    fig, AX = plt.subplots(len(row_subsmpl), len(col_subsmpl)+1,
                           figsize=(2.5*len(col_subsmpl)+2, 2*len(row_subsmpl)))
    AX = AX.reshape((len(row_subsmpl), len(col_subsmpl)+1))

    for j, l in enumerate(np.unique(data[row_key])[row_subsmpl]):
        make_row_fig(data[row_key]==l, AX[j,:], with_top_label=(j==0), fd=l)
        
    return fig

