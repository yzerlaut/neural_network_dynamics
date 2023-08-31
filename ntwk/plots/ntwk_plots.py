import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from PIL import Image # BITMAP (png, jpg, ...)
import numpy as np
from matplotlib.pylab import figure, subplot2grid

from utils.signal_processing import gaussian_smoothing
from utils import plot_tools as pt

def find_pop_keys(data, with_Aff_keys=False):
    ii, pops = 0, []
    while str(ii) in data.keys():
        pops.append(data[str(ii)]['name'])
        ii+=1

    if with_Aff_keys:
        aff_pops = []
        for key in data:
            if ('N_' in key) and (key.replace('N_', '') not in pops):
                aff_pops.append(key.replace('N_', ''))
        return pops, aff_pops
    else:
        return pops

def find_num_of_key(data,pop_key):
    ii, pops = 0, []
    while str(ii) in data.keys():
        pops.append(data[str(ii)]['name'])
        ii+=1
    i0 = np.argwhere(np.array(pops)==pop_key)[0][0]
    return i0


def connectivity_matrix(Model,
                        REC_POPS=None,
                        AFF_POPS=None,
                        COLORS=None,
                        colormap=pt.viridis,
                        blank_zero=False,
                        nticks=4,
                        ax=None):

    if (REC_POPS is None) and (AFF_POPS is None):
        REC_POPS, AFF_POPS = find_pop_keys(Model, with_Aff_keys=True)
    elif ('REC_POPS' in Model) and ('AFF_POPS' in Model):
        REC_POPS, AFF_POPS = list(Model['REC_POPS']), list(Model['AFF_POPS'])
    if COLORS is None:
        COLORS = [pt.tab10(i) for i in range(10)]

    pconnMatrix = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS)))

    for i, source_pop in enumerate(REC_POPS+AFF_POPS):
        for j, target_pop in enumerate(REC_POPS):
            if ('p_%s_%s' % (source_pop, target_pop) in Model) and (Model['p_%s_%s' % (source_pop, target_pop)]>0):
                pconnMatrix[i,j] = Model['p_%s_%s' % (source_pop, target_pop)]
            else:
                if blank_zero:
                    pconnMatrix[i,j] = -np.inf
                else:
                    pconnMatrix[i,j] = 0

    if ax is None:
        fig, ax = pt.figure(right=5.)

    cond = np.isfinite(pconnMatrix)
    Lims = [np.round(100*pconnMatrix[cond].min(),1),np.round(100*pconnMatrix[cond].max(),1)]

    pt.matrix(100*pconnMatrix.T, origin='upper',
                     colormap=colormap,
                     ax=ax, vmin=Lims[0], vmax=Lims[1])

    n, m = len(REC_POPS)+len(AFF_POPS), len(REC_POPS)
    for i, label in enumerate(REC_POPS+AFF_POPS):
        pt.annotate(ax, label, (-0.2, .9-i/n),\
                           ha='right', va='center', color=COLORS[i], size='small')
    for i, label in enumerate(REC_POPS):
        pt.annotate(ax, label, (i/m+.25, -0.1),\
                           ha='right', va='top', color=COLORS[i], rotation=65, size='small')


    pt.annotate(ax, 'pre', (1., .5), rotation=90, va='center', size='xx-small')
    pt.annotate(ax, 'post', (0.5, 1.), ha='center', size='xx-small')

    ticks = np.linspace(Lims[0], Lims[1], nticks)
    acb = pt.bar_legend(fig,
                               colorbar_inset=dict(rect=[.72,.3,.03,.5], facecolor=None),
                               colormap=colormap,
                               bounds=[Lims[0], Lims[1]],
                               ticks=ticks,
                               ticks_labels=['%.1f'%t for t in ticks],
                               label='$p_{conn}$ (%)', size='small')

    pt.set_plot(ax,
                ['left', 'bottom'],
                tck_outward=0,
                xticks=.75*np.arange(0,4)+.75/2.,
                xticks_labels=[],
                xlim_enhancement=0, ylim_enhancement=0,
                yticks=.83*np.arange(0,6)+.85/2.,
                yticks_labels=[])

    return fig, ax, acb


def raster_subplot(data, ax,
                   POP_KEYS, COLORS, tzoom,
                   Nmax_per_pop_cond=None,
                   subsampling=10,
                   with_annot=True,
                   ms=0.5):

    if Nmax_per_pop_cond is None:
        Nmax_per_pop_cond = []
        for pop in POP_KEYS:
            if 'N_%s' % pop in data:
                Nmax_per_pop_cond.append(data['N_%s' % pop])
            else:
                Nmax_per_pop_cond.append(np.max(data['iRASTER_%s' % pop]))

    n = 0
    for i, tpop in enumerate(POP_KEYS):
        cond = (data['tRASTER_%s' % tpop]>tzoom[0]) &\
                (data['tRASTER_%s' % tpop]<tzoom[1]) &\
                (data['iRASTER_%s' % tpop]<Nmax_per_pop_cond[i])

        ax.plot(data['tRASTER_%s' % tpop][cond][::subsampling],
                n+data['iRASTER_%s' % tpop][cond][::subsampling],
                '.', ms=ms, color=COLORS[i])

        ax.plot(tzoom[1]*np.ones(2), [n,n+Nmax_per_pop_cond[i]],
                'w.', ms=0.01)
        n += Nmax_per_pop_cond[i]

    ax.set_xlim(tzoom)
    ax.set_ylim([0,n])
    ax.set_yticks([])
    if with_annot:
        ax.annotate('%i '%n, (0,n), ha='right', va='center')
        ax.annotate('1 ', (0,0), ha='right', va='center')
        ax.annotate('neurons', (0, 0.5), ha='right', va='center',
                    rotation=90, xycoords='axes fraction')

    # pt.set_plot(ax, xlim=tzoom, ylabel='neuron ID',
                       # xticks_labels=[], yticks=[0,n], ylim=[0,n])


def shifted_Vms_subplot(data, ax,
                        POP_KEYS, COLORS, tzoom,
                        v_shift=10,
                        Tbar=50,
                        spike_peak = None):

    shift = 0

    for i, tpop in zip(range(len(POP_KEYS)), POP_KEYS):

        if ('VMS_%s' % tpop) in data:

            t = np.arange(len(data['VMS_%s' % tpop][0]))*data['dt']
            cond = (t>=tzoom[0]) & (t<=tzoom[1])

            for v in data['VMS_%s' % tpop]:

                ax.plot(t[cond], v[cond]+shift, '-', lw=1, c=COLORS[i])

                if spike_peak is not None:
                    tspikes, threshold = find_spikes_from_Vm(t, v, data, tpop)
                    scond = (tspikes>=tzoom[0]) & (tspikes<=tzoom[1])
                    for ts in tspikes[scond]:
                        ax.plot([ts,ts],[threshold+shift,spike_peak+shift],
                                '--', c=COLORS[i], lw=0.5)
                shift += v_shift

    pt.draw_bar_scales(ax,
                       Xbar=Tbar, Xbar_label='%ims'%Tbar,
                       # Ybar_rotation=90,
                       Ybar=v_shift,
                       Ybar_label='%imV'%v_shift)

    ax.axis('off')

def membrane_potential_subplots(data, AX,
                                POP_KEYS, COLORS, tzoom,
                                subsampling=1,
                                clip_spikes=False,
                                Vm_is_the_last_one=True):

    for i, tpop, ax in zip(range(len(POP_KEYS)), POP_KEYS, AX):

        if ('VMS_%s' % tpop) in data:

            t = np.arange(len(data['VMS_%s' % tpop][0]))*data['dt']
            cond = (t>=tzoom[0]) & (t<=tzoom[1])

            for v in data['VMS_%s' % tpop]:
                if clip_spikes:
                    tspikes, threshold = find_spikes_from_Vm(t, v, data, tpop) # getting spikes
                    tt, vv = clip_spikes_from_Vm(t[cond], v[cond], tspikes)
                    ax.plot(tt[::subsampling], vv[::subsampling], '-', lw=1, c=COLORS[i])
                else:
                    ax.plot(t[cond][::subsampling], v[cond][::subsampling], '-', lw=1, c=COLORS[i])

            ax.annotate(' %s' % tpop, (1.,.5), xycoords='axes fraction',
                           color=COLORS[i])
        else:

            ax.annotate('$V_m$ of %s not recorded' % tpop,
                           (.5,.5), xycoords='axes fraction',
                           color=COLORS[i], ha='center', va='center')

        ax.set_xlim(tzoom)
        ax.set_ylabel('$V_m$ (mV)')
        if tpop==POP_KEYS[-1] and Vm_is_the_last_one:
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticklabels([])


def Vm_subplots_mean_with_single_trace(data, AX,
                                       POP_KEYS, COLORS, tzoom,
                                       traces_ID=None,
                                       subsampling=1,
                                       clip_spikes=False):

    if traces_ID is None:
        traces_ID = np.zeros(len(POP_KEYS), dtype=int)

    for i, tpop in enumerate(POP_KEYS):

        t = np.arange(len(data['VMS_%s' % tpop][0]))*data['dt']
        cond = (t>=tzoom[0]) & (t<=tzoom[1])

        if ('VMS_%s' % tpop) in data:

            # mean
            V = np.array(data['VMS_%s' % tpop])
            pt.plot(t[cond][::subsampling], V.mean(axis=0)[cond][::subsampling],
                           sy=V.std(axis=0)[cond][::subsampling],
                           lw=0.5, color=COLORS[i], ax=AX[i], no_set=True)

            # single trace
            v = data['VMS_%s' % tpop][traces_ID[i]]

            if clip_spikes:
                tspikes, threshold = find_spikes_from_Vm(t, v, data, tpop) # getting spikes
                tt, vv = clip_spikes_from_Vm(t[cond], v[cond], tspikes)
                AX[i].plot(tt[::subsampling], vv[::subsampling], '-', lw=1, c=COLORS[i])
            else:
                AX[i].plot(t[cond][::subsampling], v[cond][::subsampling], '-', lw=1, c=COLORS[i])

        pt.set_plot(AX[i], ylabel='mV', xlim=tzoom)


def population_activity_subplot(data, ax,
                                POP_KEYS, COLORS, tzoom,
                                subsampling=1,
                                with_smoothing=0, lw=0.5,
                                fout_min=0.01,
                                with_log_scale=False):

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
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                           xlim=tzoom,
                           # ylim=[fout_min, ax.get_ylim()[1]],
                           # yticks=[0.01, 1, 100],
                           # yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                           yscale='log')
    else:
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                           num_yticks=4, xlim=tzoom)

    return ax

def input_rate_subplot(data, ax,
                       POP_KEYS, COLORS,
                       tzoom,
                       with_label=True,
                       subsampling=2):

    """
    ned to be improved to cover different afferent->target sets of waveforms
    """

    colors=['k', pt.tab10(5), pt.tab10(6)]

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
            ll.append(ax.plot(t[cond][::subsampling], rates[afferent][cond][::subsampling],
                              label=afferent, color=colors[i]))

        if with_label:
            ax.legend(frameon=False)

        pt.set_plot(ax, ['left'],
                           ylabel='input (Hz)',
                           num_yticks=3, xlim=tzoom)
    else:
        pt.annotate(ax, 'no time varying input', (.5, .5), ha='center', va='center')
        ax.axis('off')


def activity_plots(data,
                   POP_KEYS = None,
                   COLORS = None,
                   tzoom=[0, np.inf],
                   ax=None,
                   smooth_population_activity=0.,
                   pop_act_log_scale=False,
                   subsampling=2,
                   Vm_plot_args=dict(subsampling=2),
                   raster_plot_args=dict(subsampling=10),
                   fig_args=dict(hspace=0.5, right = 5.)):

    AE = [[[4,1]],
          [[4,2]]] # axes extent

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)

    for pop in POP_KEYS:
        if ('VMS_%s' % pop) in data:
            AE.append([[4,1]])
        Vm_plot_args['Vm_is_the_last_one'] = True

    if ('POP_ACT_%s' % pop) in data: # just checking on the last one
        AE.append([[4,2]])
        Vm_plot_args['Vm_is_the_last_one'] = False

    tzoom=[np.max([tzoom[0], 0.]), np.min([tzoom[1], data['tstop']])]

    if COLORS is None:
        COLORS = [pt.tab10(i) for i in range(10)]

    fig, AX = pt.figure(axes_extents=AE, **fig_args)

    input_rate_subplot(data, AX[0],
                       POP_KEYS, COLORS, tzoom)
    raster_subplot(data, AX[1],
                   POP_KEYS, COLORS, tzoom,
                   **raster_plot_args)

    if ('VMS_%s' % pop) in data:
        membrane_potential_subplots(data, AX[2:],
                                    POP_KEYS, COLORS, tzoom,
                                    **Vm_plot_args)
    if ('POP_ACT_%s' % pop) in data:
        population_activity_subplot(data, AX[-1],
                                    POP_KEYS, COLORS,
                                    tzoom,
                                    with_smoothing=smooth_population_activity,
                                    subsampling=subsampling,
                                    with_log_scale=pop_act_log_scale)

    return fig, AX


def raster_and_Vm(data,
                  POP_KEYS = None,
                  COLORS = None,
                  tzoom=[0, np.inf],
                  figsize=(5,3),
                  smooth_population_activity=0.,
                  Vm_subsampling=2):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)

    n, fig2 = 0, None
    for pop in POP_KEYS:
        if ('VMS_%s' % pop) in data:
            n+=1

    if n>0:
        fig = figure(figsize=(figsize[0],figsize[1]/2*n))
        AX = [subplot2grid((n+2,1), (0,0), rowspan=2)]
        for i in range(n):
            AX.append(subplot2grid((n+2,1), (i+2,0)))
    else:
        fig, [AX] = pt.figure((1,1), figsize=figsize,
                            keep_shape=True)

    tzoom=[np.max([tzoom[0], 0.]),
           np.min([tzoom[1], data['tstop']])]

    if COLORS is None:
        COLORS = [pt.tab10(i) for i in range(10)]

    raster_subplot(data, AX[0],
                   POP_KEYS, COLORS, tzoom)
    if n>0:
        AX[0].set_xticklabels([])
        membrane_potential_subplots(data, AX[1:],
                                    POP_KEYS, COLORS, tzoom,
                                    subsampling=Vm_subsampling)

    return fig, AX


def twin_plot_raster_pop_act(data,
                             POP_KEYS = None,
                             COLORS = None,
                             tzoom=[0, np.inf],
                             ax=None,
                             with_smoothing=10.,
                             with_log_scale=False, fout_min=1e-2,
                             lw=2,
                             raster_ms=2, raster_alpha=0.5):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = [pt.tab10(i) for i in range(10)]

    tzoom=[np.max([tzoom[0], 0.]), np.min([tzoom[1], data['tstop']])]
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    if ax is None:
        fig, ax = pt.figure(axes_extents=[[[4,2]]], hspace=0.5, right = 5.)
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
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
                    ylim=[fout_min, ax.get_ylim()[1]],
                    yticks=[0.01, 1, 100],
                    yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                    yscale='log')
    else:
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    num_yticks=4,
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    pt.set_plot(ax2, ['right'], ylabel='neurons',
                       xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])

    return fig, ax


######################################
#### RASTER PLOT
######################################

def raster(data,
           POP_KEYS = None, COLORS=None,
           NMAXS = None,
           tzoom=[0, np.inf],
           bar_scales_args=dict(Xbar=50, Xbar_label='50ms',
                                Ybar=500, Ybar_label='500 neurons',
                                loc='bottom-left'),
           subsampling=1,
           ms=1, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = ['C'+str(i) for i in range(len(POP_KEYS))]
    if NMAXS is None:
        NMAXS = np.array([float(data['N_'+pop]) for pop in POP_KEYS])

    if ax is None:
        _, ax = pt.subplots(1, figsize=(3.2,2))
        pt.subplots_adjust(left=.05, bottom=.2)

    # raster activity
    nn = 0
    for n, pop_key, color, nmax in zip(range(len(NMAXS)), POP_KEYS, COLORS, NMAXS):
        try:
            cond = (data['tRASTER_'+pop_key]>tzoom[0]) & (data['tRASTER_'+pop_key]<tzoom[1]) & (data['iRASTER_'+pop_key]<nmax)
            #ax.plot(data['tRASTER_'+pop_key][cond][::subsampling], np.sum(NMAXS[:n])+data['iRASTER_'+pop_key][cond][::subsampling], '.', color=color, ms=ms)
            pt.scatter(data['tRASTER_'+pop_key][cond][::subsampling], np.sum(NMAXS[:n])+data['iRASTER_'+pop_key][cond][::subsampling], color=color, ms=ms, ax=ax, no_set=True)
        except ValueError:
            pass

    pt.set_plot(ax, [], yticks=[], xticks=[],
                       xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
                       ylim=[0, np.sum(NMAXS)])
    if bar_scales_args is not None:
        pt.draw_bar_scales(ax, **bar_scales_args)

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
            ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = [pt.tab10(i) for i in range(10)]

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    if ax is None:
        fig, ax = pt.figure(axes_extents=[[[4,1]]], hspace=0.5, right = 5.)

    for pop_key, color in zip(POP_KEYS, COLORS):
        if with_smoothing>0:
            ax.plot(t[cond],
                    gaussian_smoothing(data['POP_ACT_'+pop_key][cond], int(with_smoothing/data['dt'])),
                    color=color, lw=lw)
        else:
            ax.plot(t[cond], data['POP_ACT_'+pop_key][cond], color=color, lw=lw)

    if with_log_scale:
        # ax.set_yscale("log", nonposy='clip')
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
                    ylim=[fout_min, ax.get_ylim()[1]],
                    yticks=[0.01, 1, 100],
                    yticks_labels=['$10^{-2}$', '$10^0$', '$10^{2}$'],
                    yscale='log')
    else:
        pt.set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)',
                    num_yticks=4,
                    xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])

    return ax


def find_spikes_from_Vm(t, Vm, data, pop_key):
    threshold, reset = data[pop_key+'_Vthre'], data[pop_key+'_Vreset']
    if pop_key+'_delta_v' in data:
        threshold += data[pop_key+'_delta_v']
    # adding spikes
    tspikes = t[1:][np.argwhere((Vm[1:]-Vm[:-1])<(.8*(reset-threshold)))]
    return tspikes, threshold

def clip_spikes_from_Vm(t, Vm, tspikes,
                        clip_window=10):

    clipped = np.zeros(len(t), dtype=bool)
    for ts in tspikes:
        clipped[(t>=ts) & (t<ts+clip_window)] = True
    return t[~clipped], Vm[~clipped]


def few_Vm_plot(data,
                POP_KEYS = None, COLORS=None, NVMS=None,
                tzoom=[0, np.inf],
                clip_spikes=False,
                vpeak=-40, vbottom=-80, shift=20.,
                bar_scales_args=dict(Xbar=50, Xbar_label='50ms', Ybar=20, Ybar_label='20mV'),
                subsampling=1,
                lw=1, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = ['C'+str(i) for i in range(len(POP_KEYS))]
    if NVMS is None:
        NVMS = np.array([range(len(data['VMS_'+pop_key])) for pop_key in POP_KEYS], dtype=object)

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    if ax is None:
        _, ax = pt.subplots(figsize=(5,3))
        pt.subplots_adjust(left=.15, bottom=.1, right=.99)

    cond = (t>tzoom[0]) & (t<tzoom[1])

    nn = 0
    for VmID, pop_key, color in zip(NVMS, POP_KEYS, COLORS):
        rest = data[pop_key+'_El']
        for i in VmID:
            nn+=1
            Vm = data['VMS_'+pop_key][i].flatten()
            tspikes, threshold = find_spikes_from_Vm(t, Vm, data, pop_key) # getting spikes
            if clip_spikes:
                tt, vv = clip_spikes_from_Vm(t[cond], Vm[cond], tspikes)
                ax.plot(tt[::subsampling], vv[::subsampling]+shift*nn, color=color, lw=lw)
            else:
                ax.plot(t[cond][::subsampling], Vm[cond][::subsampling]+shift*nn,
                        color=color, lw=lw)
            # adding spikes
            Scond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
            for ts in tspikes[Scond]:
                ax.plot([ts, ts], shift*nn+np.array([threshold, vpeak]), '--', color=color, lw=lw)
            ax.plot([t[cond][0], t[cond][-1]], shift*nn+np.array([rest, rest]), ':', color=color, lw=lw)

    pt.set_plot(ax, [], xticks=[], yticks=[],
                xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])])
    if bar_scales_args is not None:
        pt.draw_bar_scales(ax, **bar_scales_args)

    return ax

def exc_inh_balance(data, pop_key='Exc'):

    NVm= len(data['VMS_'+str(pop_key)])

    fig, [ax, ax2] = pt.subplots(1, 2, figsize=(3.5,2))
    pt.subplots_adjust(left=.4, bottom=.2, hspace=1., wspace=1., right=.99)

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

    fig, AX = pt.subplots(2, 2, figsize=(4,3))
    pt.subplots_adjust(bottom=.3, top=.99, right=.99)

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
    fig = figure(figsize=(3,.3))
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
        fig = figure(figsize=(3,.3))
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
    pt.close('all')
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


