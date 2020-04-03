import numpy as np
import datavyz

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pylab as plt


Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def color(pop_key):
    if pop_key=='RecExc':
        return Green
    if pop_key=='RecInh':
        return Red
    if pop_key=='AffExc':
        return Grey
    if pop_key=='DsInh':
        return Purple

def plot_single_cell_sim(data, XTICKS = None,
                         COLORS = [Green, Blue, Red, Orange], savefig=''):
    Model = data['Model']
    Vthre = Model[Model['NRN_KEY']+'_Vthre']
    fig, N1, N2 = plt.figure(figsize=(4,5)), 5, 1
    plt.subplots_adjust(left=.15, bottom=.05, top=.96, right=.99)
    ax1 = plt.subplot2grid((N1, N2), (0, 0))
    ax2 = plt.subplot2grid((N1, N2), (1, 0), rowspan=2)
    ax3 = plt.subplot2grid((N1, N2), (3, 0), rowspan=2)
    # presynaptic raster
    j = 0
    for i in range(len(Model['POP_STIM'])):
        prespikes = data['t_prespikes'][i]
        Npre = Model['N_'+Model['POP_STIM'][i]]
        y = j+np.random.randint(Npre, size=len(prespikes))
        ax1.plot(prespikes, y, 'o', color=color(Model['POP_STIM'][i]), ms=2.)
        j += Npre
    ax1.set_yticks([]);ax1.set_ylabel('presynaptic\n neurons')
    # synaptic currents
    ax2.plot(np.arange(len(data['Vm'][0]))*float(Model['dt']), data['Ie'][0], lw=1, color=Green)
    ax2.plot(np.arange(len(data['Vm'][0]))*float(Model['dt']), data['Ii'][0], lw=1, color=Red)
    ax2.set_ylabel('synaptic\ncurrents (nA)')
    # Vm and spikes
    for i in range(Model['N_SEED']):
        ax3.plot(np.arange(len(data['Vm'][i]))*float(Model['dt']), data['Vm'][i], lw=1., color='k')
    for t in data['tspikes']: ax3.plot([t,t], [Vthre, Vthre+5], ':', lw=2, color='k')
    for label, func in zip(['time (ms)', '$V_m$ (mV)'], [ax3.set_xlabel, ax3.set_ylabel]): func(label) # labeling
    ax3.set_yticks([-70,-60,-50])
    if XTICKS is not None:
        ax1.set_xticks(XTICKS)
        ax2.set_xticks(XTICKS)
        ax3.set_xticks(XTICKS)
    ax2.set_xticklabels([])
    ax1.set_xticklabels([])
    return fig

def make_tf_plot_2_variables(data,
                             xkey='F_RecExc', ckey='F_RecInh', output_key='Fout',
                             ckey_label='$\\nu_{i}$ (Hz)',
                             cond=None,
                             ylim=[1e-2, 100], yticks=[0.01, 0.1, 1, 10], yticks_labels=['0.01', '0.1', '1', '10'], ylabel='$\\nu_{out}$ (Hz)',
                             xticks=[0.1, 1, 10], xticks_labels=['0.1', '1', '10'], xlabel='$\\nu_{e}$ (Hz)',
                             logscale=True, cmap=cm.copper, ax=None, acb=None, ge=None,
                             with_top_label=False,
                             with_theory=False, th_discret=20):
    
    # limiting the data within the range
    Fout_mean, Fout_std = data[output_key+'_mean'], data[output_key+'_std']

    # graph settings
    if ge is None:
        ge = datavyz.main.graph_env()
    if ax is None:
        fig, ax, acb = ge.figure(with_space_for_bar_legend=True)

    

    if cond is None:
        cond = np.ones(len(data[xkey]), dtype=bool)
        
    F0, F1 = data[xkey][cond], data[ckey][cond]
    
    mFout, sFout = Fout_mean[cond], Fout_std[cond]
    for i, f1 in enumerate(np.unique(F1)):
        i1 = np.argwhere(F1==f1).flatten()
        
        cond1 =  (mFout[i1]>=ylim[0]) & (mFout[i1]<ylim[1])
        ax.errorbar(F0[i1][cond1],
                    Fout_mean[i1][cond1],
                    yerr=Fout_std[i1][cond1],
                    fmt='o', ms=4,
                    color=cmap(i/len(np.unique(F1[i1]))))
        
        # # # now analytical estimate
        # if with_theory:
        #     RATES = {xkey:np.concatenate([np.linspace(f1, f2, th_discret, endpoint=False)\
        #                                        for f1, f2 in zip(Fe[i1][i1][:-1], Fe[i1][i1][1:])])}
        #     for pop, f in zip([col_key, ckey, row_key],[f1, fi, fd]) :
        #         RATES[pop] = f*np.ones(len(RATES[xkey]))
        #     Fout_th = TF(RATES, data['Model'], data['Model']['NRN_KEY'])
        #     th_cond = (Fout_th>ylim[0]) & (Fout_th<ylim[1])
        #     AX[i].plot(RATES[xkey][th_cond],
        #                Fout_th[th_cond], '-',
        #                color=cmap(j/len(np.unique(Fi[i1]))), lw=5, alpha=.7)
        
    if with_top_label:
        ax.set_title(col_key_label+'='+str(round(f1,1))+col_key_unit)
    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ylabel2, yticks_labels2 = '', []
    if i==0: # if first column
        ylabel2, yticks_labels2 = ylabel, yticks_labels
    ge.set_plot(ax, ylim=[.9*ylim[0], ylim[1]],
                yticks=yticks, xticks=xticks, ylabel=ylabel2, xlabel=xlabel,
                yticks_labels=yticks_labels2, xticks_labels=xticks_labels)

    F1_log = np.log(F1)/np.log(10)
    # cax = inset_axes(AX[-1], width="20%", height="90%", loc=3)
    F1_ticks, F1_ticks_labels = [], []
    for k in np.arange(-2, 3):
        for l in range(1, 10):
            F1_ticks.append(np.log(l)/np.log(10) +k)
            if l%10==1:
                F1_ticks_labels.append(str(l*np.exp(k*np.log(10))))
            else:
                F1_ticks_labels.append('')

    cb = ge.build_bar_legend(F1_ticks, acb , cmap,
                             bounds = [np.log(F1.min())/np.log(10),
                                       np.log(F1.max())/np.log(10)],
                             ticks_labels=F1_ticks_labels,
                             color_discretization=100,
                             label=ckey_label)


    return fig, ax, acb

