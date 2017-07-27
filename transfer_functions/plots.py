import matplotlib.pylab as plt
import numpy as np

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
