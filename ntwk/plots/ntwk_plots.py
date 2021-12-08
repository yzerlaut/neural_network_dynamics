import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from PIL import Image # BITMAP (png, jpg, ...)
import numpy as np
import matplotlib.pylab as plt

from datavyz import graph_env_manuscript as ge
from analyz.processing.signanalysis import gaussian_smoothing


def find_pop_keys(data):
    ii, pops = 0, []
    while str(ii) in data.keys():
        pops.append(data[str(ii)]['name'])
        ii+=1
    return pops
    
def find_num_of_key(data,pop_key):
    ii, pops = 0, []
    while str(ii) in data.keys():
        pops.append(data[str(ii)]['name'])
        ii+=1
    i0 = np.argwhere(np.array(pops)==pop_key)[0][0]
    return i0


def raster_subplot(data, ax,
                   POP_KEYS, COLORS, tzoom,
                   graph_env):

    n = 0
    for i, tpop in enumerate(POP_KEYS):

        cond = (data['tRASTER_%s' % tpop]>tzoom[0]) & (data['tRASTER_%s' % tpop]<tzoom[1])
        ax.plot(data['tRASTER_%s' % tpop][cond], n+data['iRASTER_%s' % tpop][cond],
                   'o', ms=1, c=COLORS[i], alpha=.5)
        ax.plot(tzoom[1]*np.ones(2), [n,n+data['N_%s' % tpop]], 'w.', ms=0.01)
        n += data['N_%s' % tpop]
        
    graph_env.set_plot(ax, xlim=tzoom, ylabel='neuron ID',
                       xticks_labels=[], yticks=[0,n], ylim=[0,n])


def membrane_potential_subplots(data, AX,
                                POP_KEYS, COLORS, tzoom,
                                graph_env,
                                subsampling=1,
                                Vm_is_the_last_one=True):
    
    for i, tpop in enumerate(POP_KEYS):
        
        if ('VMS_%s' % tpop) in data:

            t = np.arange(len(data['VMS_%s' % tpop][0]))*data['dt']
            cond = (t>=tzoom[0]) & (t<=tzoom[1])
            
            for v in data['VMS_%s' % tpop]:
                AX[i].plot(t[cond][::subsampling], v[::subsampling], '-', lw=1, c=COLORS[i])
                
            graph_env.annotate(AX[i], ' %s' % tpop, (1.,.5), xycoords='axes fraction',
                        color=COLORS[i], bold=True, size='large')
        else:

            graph_env.annotate(AX[i], '$V_m$ of %s not recorded' % tpop,
                               (.5,.5), xycoords='axes fraction',
                               color=COLORS[i], bold=True, size='large', ha='center', va='center')

        if tpop==POP_KEYS[-1] and Vm_is_the_last_one:
            graph_env.set_plot(AX[i], xlabel='time (ms)', ylabel='Vm (mV)', xlim=tzoom)
        else:
            graph_env.set_plot(AX[i], ylabel='Vm (mV)', xlim=tzoom, xticks_labels=[])

def population_activity_subplot(data, ax,
                                POP_KEYS, COLORS, tzoom,
                                graph_env,
                                subsampling=1,
                                with_smoothing=0, lw=2,
                                fout_min=0.01, with_log_scale=False):

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    for pop_key, color in zip(POP_KEYS, COLORS):
        if with_smoothing>0:
            ax.plot(t[cond][::subsampling],
                    gaussian_smoothing(data['POP_ACT_'+pop_key][cond], int(with_smoothing/data['dt']))[::subsampling]+fout_min,
                    color=color, lw=lw)
        else:
            ax.plot(t[cond][::subsampling],
                    data['POP_ACT_'+pop_key][cond][::subsampling]+fout_min,
                    color=color, lw=lw)

    if with_log_scale:
        # ax.set_yscale("log", nonposy='clip')
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                           xlim=tzoom,
                           # ylim=[fout_min, ax.get_ylim()[1]],
                           # yticks=[0.01, 1, 100],
                           # yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                           yscale='log')
    else:
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                           num_yticks=4, xlim=tzoom)
    
    return ax

def input_rate_subplot(data, ax,
                       POP_KEYS, COLORS,
                       tzoom,
                       graph_env,
                       subsampling=2):

    """
    ned to be improved to cover different afferent->target sets of waveforms
    """
    colors=['k', plt.cm.tab10(5), plt.cm.tab10(6)]
    
    rates, afferents, afferents_targets = {}, [], []
    for key in data:
        if len(key.split('Rate_'))>1:
            _, afferent, target = key.split('_')
            # rates['%s_%s' % (afferent, target)] = data[key]

            if afferent not in rates:
                rates[afferent] = data[key]
    if len(rates.keys())>0:
        ll = []
        for i, afferent in enumerate(rates):
            t = np.arange(len(rates[afferent]))*data['dt']
            cond = (t>tzoom[0]) & (t<tzoom[1])
            ll.append(ax.plot(t[cond][::subsampling], rates[afferent][cond][::subsampling], label=afferent, color=colors[i]))

        ax.legend(frameon=False, fontsize=graph_env.fontsize)
        
        graph_env.set_plot(ax, ['left'],
                           ylabel='input (Hz)',
                           num_yticks=3, xlim=tzoom)
    else:
        graph_env.annotate(ax, 'no time varying input', (.5, .5), ha='center', va='center')
        ax.axis('off')
        

def activity_plots(data,
                   POP_KEYS = None,
                   COLORS = None,
                   tzoom=[0, np.inf],
                   smooth_population_activity=0.,
                   pop_act_log_scale=False,
                   subsampling=2,
                   graph_env=ge, ax=None,
                   fig_args=dict(hspace=0.5, right = 5.)):

    AE = [[[4,1]],
          [[4,2]]] # axes extent
    
    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)

    for pop in POP_KEYS:
        if ('VMS_%s' % pop) in data:
            AE.append([[4,1]])
        Vm_is_the_last_one = True
            
    if ('POP_ACT_%s' % pop) in data: # just checking on the last one
        AE.append([[4,2]])
        Vm_is_the_last_one = False
        
    tzoom=[np.max([tzoom[0], 0.]), np.min([tzoom[1], data['tstop']])]

    if COLORS is None:
        COLORS = graph_env.colors

    fig, AX = graph_env.figure(axes_extents=AE, **fig_args)

    input_rate_subplot(data, AX[0],
                       POP_KEYS, COLORS, tzoom,
                       graph_env)
    raster_subplot(data, AX[1],
                   POP_KEYS, COLORS, tzoom,
                   graph_env)

    if ('VMS_%s' % pop) in data:
        membrane_potential_subplots(data, AX[2:],
                                    POP_KEYS, COLORS, tzoom,
                                    graph_env, subsampling=subsampling,
                                    Vm_is_the_last_one=Vm_is_the_last_one)
    if ('POP_ACT_%s' % pop) in data:
        population_activity_subplot(data, AX[-1],
                                    POP_KEYS, COLORS,
                                    tzoom,
                                    graph_env,
                                    with_smoothing=smooth_population_activity,
                                    subsampling=subsampling,
                                    with_log_scale=pop_act_log_scale)
        

    return fig, AX

def raster_and_Vm_plot(data,
                       POP_KEYS = None,
                       COLORS = None,
                       tzoom=[0, np.inf],
                       smooth_population_activity=0.,
                       Vm_subsampling=2,
                       graph_env=ge, ax=None):

    AE = [[[4,2]]] # axes extent
    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
        
    for pop in POP_KEYS:
        if ('VMS_%s' % pop) in data:
            AE.append([[4,1]])

    tzoom=[np.max([tzoom[0], 0.]), np.min([tzoom[1], data['tstop']])]

    if COLORS is None:
        COLORS = graph_env.colors

    fig, AX = graph_env.figure(axes_extents=AE, hspace=0.5, right = 5.)

    if ('VMS_%s' % pop) in data:
        raster_subplot(data, AX[0],
                       POP_KEYS, COLORS, tzoom,
                       graph_env)
        membrane_potential_subplots(data, AX[1:],
                                    POP_KEYS, COLORS, tzoom,
                                    graph_env, subsampling=Vm_subsampling)
    else:
        raster_subplot(data, AX,
                       POP_KEYS, COLORS, tzoom,
                       graph_env)

    return fig, AX


def twin_plot_raster_pop_act(data,
                             POP_KEYS = None,
                             COLORS = None,
                             tzoom=[0, np.inf],
                             with_smoothing=10.,
                             with_log_scale=False, fout_min=1e-2,
                             lw=2,
                             raster_ms=2, raster_alpha=0.5,
                             graph_env=ge, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = graph_env.colors
        
    tzoom=[np.max([tzoom[0], 0.]), np.min([tzoom[1], data['tstop']])]
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    if ax is None:
        fig, ax = graph_env.figure(axes_extents=[[[4,2]]], hspace=0.5, right = 5.)
    else:
        fig = None

    t = np.arange(len(data['POP_ACT_'+POP_KEYS[0]]))*data['dt']
    for pop_key, color in zip(POP_KEYS, COLORS):
        if with_smoothing>0:
            ax.plot(t[cond],
                    gaussian_smoothing(data['POP_ACT_'+pop_key][cond], int(with_smoothing/data['dt'])),
                    color=color, lw=lw)
        else:
            ax.plot(t[cond], data['POP_ACT_'+pop_key][cond], color=color, lw=lw)
            
    # now spikes
    ax2 = ax.twinx()
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    n=0
    for i, tpop in enumerate(POP_KEYS):

        cond = (data['tRASTER_%s' % tpop]>tzoom[0]) & (data['tRASTER_%s' % tpop]<tzoom[1])
        ax2.plot(data['tRASTER_%s' % tpop][cond], n+data['iRASTER_%s' % tpop][cond],
                 '.', ms=raster_ms, c=COLORS[i], alpha=raster_alpha)
        ax2.plot(tzoom[1]*np.ones(2), [n,n+data['N_%s' % tpop]], 'w.', ms=0.01)
        n += data['N_%s' % tpop]

    if with_log_scale:
        # ax.set_yscale("log", nonposy='clip')
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
                    ylim=[fout_min, ax.get_ylim()[1]],
                    yticks=[0.01, 1, 100],
                    yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                    yscale='log')
    else:
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    num_yticks=4,
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    graph_env.set_plot(ax2, ['right'], ylabel='neuron ID',
                       xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    
    return fig, ax

    
######################################
#### RASTER PLOT
######################################

def raster(data,
           POP_KEYS = None, COLORS=None,
           NMAXS = None,
           tzoom=[0, np.inf],
           Nnrn=500, Tbar=50,
           ms=1, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = ['C'+str(i) for i in range(len(POP_KEYS))]
    if NMAXS is None:
        NMAXS = np.array([float(data['N_'+pop]) for pop in POP_KEYS])

    if ax is None:
        _, ax = plt.subplots(1, figsize=(3.2,2))
        plt.subplots_adjust(left=.05, bottom=.2)
        
    # raster activity
    nn = 0
    for n, pop_key, color, nmax in zip(range(len(NMAXS)), POP_KEYS, COLORS, NMAXS):
        try:
            cond = (data['tRASTER_'+pop_key]>tzoom[0]) & (data['tRASTER_'+pop_key]<tzoom[1]) & (data['iRASTER_'+pop_key]<nmax)
            ax.plot(data['tRASTER_'+pop_key][cond], NMAXS[:n].sum()+data['iRASTER_'+pop_key][cond], '.', color=color, ms=ms)
        except ValueError:
            pass
    ax.plot(tzoom[0]*np.ones(2), [0, Nnrn], lw=5, color='gray')
    ax.annotate(str(Nnrn)+' neurons',\
                 (-0.1, 0.5), rotation=90, fontsize=12, xycoords='axes fraction')
    ax.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms',
                 (0., -0.1), fontsize=12, xycoords='axes fraction')
    set_plot(ax, [], yticks=[], xticks=[],
             xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
             ylim=[0, NMAXS.sum()])
    return ax

######################################
#### TIME-VARYING ACTIVITIES
######################################

def pop_act(data,
            POP_KEYS = None, COLORS=None,
            with_smoothing=0,
            with_log_scale=False,
            fout_min=0.01,
            tzoom=[0, np.inf],
            lw=2,
            graph_env=ge, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = graph_env.colors
        
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    if ax is None:
        fig, ax = graph_env.figure(axes_extents=[[[4,1]]], hspace=0.5, right = 5.)

    for pop_key, color in zip(POP_KEYS, COLORS):
        if with_smoothing>0:
            ax.plot(t[cond],
                    gaussian_smoothing(data['POP_ACT_'+pop_key][cond], int(with_smoothing/data['dt'])),
                    color=color, lw=lw)
        else:
            ax.plot(t[cond], data['POP_ACT_'+pop_key][cond], color=color, lw=lw)

    if with_log_scale:
        # ax.set_yscale("log", nonposy='clip')
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
                    ylim=[fout_min, ax.get_ylim()[1]],
                    yticks=[0.01, 1, 100],
                    yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                    yscale='log')
    else:
        graph_env.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    num_yticks=4,
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    
    return ax


def find_spikes_from_Vm(t, Vm, data, pop_key):
    threshold, reset = data[pop_key+'_Vthre'], data[pop_key+'_Vreset']
    if pop_key+'_delta_v' in data:
        threshold += data[pop_key+'_delta_v']
    # adding spikes
    tspikes = t[1:][np.argwhere((Vm[1:]-Vm[:-1])<(.9*(reset-threshold)))]
    return tspikes, threshold
    

def few_Vm_plot(data,
                POP_KEYS = None, COLORS=None, NVMS=None,
                tzoom=[0, np.inf],
                vpeak=-40, vbottom=-80, shift=20.,
                Tbar=50., Vbar=20.,
                lw=1, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = ['C'+str(i) for i in range(len(POP_KEYS))]
    if NVMS is None:
        NVMS = np.array([range(len(data['VMS_'+pop_key])) for pop_key in POP_KEYS], dtype=object)
        
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    if ax is None:
        _, ax = plt.subplots(figsize=(5,3))
        plt.subplots_adjust(left=.15, bottom=.1, right=.99)
    
    cond = (t>tzoom[0]) & (t<tzoom[1])

    nn = 0
    for VmID, pop_key, color in zip(NVMS, POP_KEYS, COLORS):
        rest = data[pop_key+'_El']
        for i in VmID:
            nn+=1
            Vm = data['VMS_'+pop_key][i].flatten()
            ax.plot(t[cond], Vm[cond]+shift*nn, color=color, lw=lw)
            # adding spikes
            tspikes, threshold = find_spikes_from_Vm(t, Vm, data, pop_key)
            Scond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
            for ts in tspikes[Scond]:
                ax.plot([ts, ts], shift*nn+np.array([threshold, vpeak]), '--', color=color, lw=lw)
            ax.plot([t[cond][0], t[cond][-1]], shift*nn+np.array([rest, rest]), ':', color=color, lw=lw)

    y0 = ax.get_ylim()[0]
    ax.plot([tzoom[0],tzoom[0]+Tbar], y0*np.ones(2),
                 lw=2, color='k')
    ax.annotate(str(int(Tbar))+' ms',
                 (0., -0.1), fontsize=12, xycoords='axes fraction')
    ax.plot([tzoom[0],tzoom[0]], y0+np.arange(2)*Vbar,
                 lw=2, color='k')
    ax.annotate(str(int(Vbar))+' mV',
                 (-0.1, 0.5), rotation=90, fontsize=12, xycoords='axes fraction')
    ge.set_plot(ax, [], xticks=[], yticks=[],
                xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    
    return ax

def exc_inh_balance(data, pop_key='Exc'):
    
    NVm= len(data['VMS_'+str(pop_key)])
    
    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(3.5,2))
    plt.subplots_adjust(left=.4, bottom=.2, hspace=1., wspace=1., right=.99)

    # removings the current points where clamped at reset potential (creates artificially strong exc currents)
    CONDS =[]
    for i in range(NVm):
        CONDS.append(data['VMS_'+str(pop_key)][i]!=data[str(find_num_of_key(data,pop_key))+'_params']['Vreset'])
        
    # excitation
    mean = np.mean([data['ISYNe_'+str(pop_key)][i][CONDS[i]].mean() for i in range(NVm)])
    std = np.std([data['ISYNe_'+str(pop_key)][i][CONDS[i]].mean() for i in range(NVm)])
    ax.bar([0], mean, yerr=std, edgecolor='g', facecolor='w', lw=3,
           error_kw={'ecolor':'g','linewidth':3}, capsize=3)

    # inhibition
    mean = -np.mean([data['ISYNi_'+str(pop_key)][i][CONDS[i]].mean() for i in range(NVm)])
    std = np.std([data['ISYNi_'+str(pop_key)][i][CONDS[i]].mean() for i in range(NVm)])
    ax.bar([1], mean, yerr=std, edgecolor='r', facecolor='w', lw=3,
           error_kw={'ecolor':'r','linewidth':3}, capsize=3)
    
    set_plot(ax, ylabel='currents \n (abs. value, pA)',
             xticks=[0,1], xticks_labels=['exc.', 'inh.'])

    Gl = data[str(find_num_of_key(data,pop_key))+'_params']['Gl']
    # excitation
    mean = np.mean([data['GSYNe_'+str(pop_key)][i].mean() for i in range(NVm)])/Gl
    std = np.std([data['GSYNe_'+str(pop_key)][i].mean() for i in range(NVm)])/Gl
    ax2.bar([0], mean, yerr=std, edgecolor='g', facecolor='w', lw=3,
           error_kw={'ecolor':'g','linewidth':3}, capsize=3)

    # inhibition
    mean = np.mean([data['GSYNi_'+str(pop_key)][i].mean() for i in range(NVm)])/Gl
    std = np.std([data['GSYNi_'+str(pop_key)][i].mean() for i in range(NVm)])/Gl
    ax2.bar([1], mean, yerr=std, edgecolor='r', facecolor='w', lw=3,
           error_kw={'ecolor':'r','linewidth':3}, capsize=3)
    
    set_plot(ax2, ylabel='$G_{syn}$/$g_L$',
             xticks=[0,1], xticks_labels=['exc.', 'inh.'])
    
    return fig

def histograms(data, pop_key='Exc'):
    
    NVm= len(data['VMS_'+str(pop_key)])
    
    fig, AX = plt.subplots(2, 2, figsize=(4,3))
    plt.subplots_adjust(bottom=.3, top=.99, right=.99)

    ######## VM ########
    # excitation
    for i in range(NVm):
        hist, be = np.histogram(data['VMS_'+exc_pop_key][i], bins=20, normed=True)
        AX[0, 0].plot(.5*(be[1:]+be[:-1]), hist, color=G, lw=.5)
    hist, be = np.histogram(np.concatenate(data['VMS_'+exc_pop_key]), bins=20, normed=True)
    AX[0, 0].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    set_plot(AX[0, 0], ['bottom'], yticks=[],
             xticks=[-70, -60, -50], xticks_labels=[], xlim=[-75,-45])
    
    # inhibition
    for i in range(NVm):
        hist, be = np.histogram(data['VMS_'+inh_pop_key][i], bins=20, normed=True)
        AX[1, 0].plot(.5*(be[1:]+be[:-1]), hist, color=R, lw=.5)
    hist, be = np.histogram(np.concatenate(data['VMS_'+inh_pop_key]), bins=20, normed=True)
    AX[1, 0].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=R, lw=0,
                 width=be[1]-be[0], facecolor=R, alpha=.3)
    set_plot(AX[1, 0], ['bottom'], xlabel='$V_m$ (mV)', yticks=[],
             xticks=[-70, -60, -50], xlim=[-75,-45])

    ######## CURRENTS ########
    # on excitatory population
    cond = np.concatenate(data['VMS_'+exc_pop_key])!=data[str(find_num_of_key(data,'Exc'))+'_params']['Vreset'] # removing clamping at reset
    hist, be = np.histogram(np.concatenate(data['ISYNe_'+exc_pop_key])[cond], bins=20, normed=True)
    AX[0, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    hist, be = np.histogram(np.concatenate(data['ISYNi_'+exc_pop_key])[cond], bins=20, normed=True)
    AX[0, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=R, lw=0,
                 width=be[1]-be[0], facecolor=R, alpha=.3)
        
    # on inhibitory population
    cond = np.concatenate(data['VMS_'+inh_pop_key])!=data[str(find_num_of_key(data,'Inh'))+'_params']['Vreset'] # removing clamping at reset
    hist, be = np.histogram(np.concatenate(data['ISYNe_'+inh_pop_key])[cond], bins=20, normed=True)
    AX[1, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    hist, be = np.histogram(np.concatenate(data['ISYNi_'+inh_pop_key])[cond], bins=20, normed=True)
    AX[1, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=R, lw=0,
                 width=be[1]-be[0], facecolor=R, alpha=.3)
    imax = np.amax(np.concatenate([AX[0,1].get_xlim(), AX[1,1].get_xlim()]))
    set_plot(AX[0, 1], ['bottom'], yticks=[], xlim=[-imax, imax],
             num_xticks=3, xticks_labels=[])
    set_plot(AX[1, 1], ['bottom'], yticks=[], xlabel='currents (pA)',
             xlim=[-imax, imax], num_xticks=3)
    fig.text(0,.8,'exc. cells', fontsize=11)
    fig.text(0,.4,'inh. cells', fontsize=11)
    return fig


def assemble_quantities(data, filename,
                        exc_pop_key='Exc',
                        inh_pop_key='Inh',
                        tzoom=[800,1200]):

    new_im = Image.new('RGBA', (830, 1140), (255,255,255,255))

    # title
    fig = plt.figure(figsize=(3,.3))
    fig.text(0., 0.2,
             '$\\nu_{aff}$='+str(round(data['F_AffExc'][0]))+'Hz, $\\nu_{dsnh}$='+\
             str(round(data['F_DsInh'][0]))+'Hz', fontsize=14)
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (100, 0))
    # raster
    fig = raster_fig(data, tzoom=tzoom)
    fig.text(0., .9, '(i)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (10, 40))
    # pop activity
    try:
        fig = pop_act(data)
    except KeyError:
        fig = plt.figure(figsize=(3,.3))
    fig.text(0., .9, '(ii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (200, 40))
    # exc inh balance
    fig = exc_inh_balance(data, pop_key=exc_pop_key)
    fig.text(0.1, .9, '(iii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (450, 40))
    # Vm traces excitation
    fig = Vm_Isyn_fig(data, pop_key=exc_pop_key, tzoom=tzoom)
    fig.text(0.2, .92, '(iv) excitatory cells sample', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (20, 250))
    # Vm traces inhibition
    fig = Vm_Isyn_fig(data, pop_key=inh_pop_key, tzoom=tzoom)
    fig.text(0.2, .92, '(v) inhibitory cells sample', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (20, 560))
    fig = histograms(data)
    fig.text(0.2, .92, '(vi)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (20, 860))
    # raster much zoomed
    fig = raster_fig(data, tzoom=[tzoom[0], tzoom[0]+20], Tbar=5)
    fig.text(0., .9, '(vii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (500, 850))
    # Vm much zoomed
    fig = few_Vm_fig(data, pop_key=exc_pop_key, tzoom=[tzoom[0], tzoom[0]+20], Tbar=5)
    fig.text(0., .7, '(viii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (500, 1050))
    
    # closing everything
    plt.close('all')
    new_im.save(filename)

    
if __name__=='__main__':
    import sys
    sys.path.append('../../')
    
    from params_scan.aff_exc_aff_dsnh_params_space import get_scan
    args, F_aff, F_dsnh, DATA = get_scan(\
                    '../../params_scan/data/scan.zip')
    assemble_quantities(DATA[-1],
                        'data/0.png',
                        exc_pop_key='RecExc',
                        inh_pop_key='RecInh')

    
