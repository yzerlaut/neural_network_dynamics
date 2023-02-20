import numpy as np


from utils import plot_tools as pt
from ntwk.theory.tf import TF

def plot_single_cell_sim(data,
                         XTICKS = None,
                         fig_args={'figsize':(1.8,.6), 'hspace':.5, 'right':.5, 'left':.6},
                         COLORS = [],
                         savefig='',
                         dVthre=10,
                         ms=1,
                         subsampling=2):
    
    if len(COLORS)==0:
        COLORS = [pt.cm.tab10(i) for i in range(10)]

    fig, [ax1, ax2, ax3] = figure(axes=(3,1), **fig_args)
        
    Model = data['Model']
    Vthre = Model[Model['NRN_KEY']+'_Vthre']
    N1, N2 = 5, 1
    
    # presynaptic raster
    j = 0
    for i in range(len(Model['POP_STIM'])):
        prespikes = data['t_prespikes'][i]*1e-3
        Npre = Model['N_'+Model['POP_STIM'][i]]
        y = j+np.random.randint(Npre, size=len(prespikes))
        ax1.plot(prespikes, y, 'o', color=COLORS[i], ms=ms)
        j += Npre

    t = np.arange(len(data['Vm'][0]))*float(Model['dt'])
    
    # synaptic currents
    ax2.plot(t[::subsampling], data['Ie'][0][::subsampling], lw=1, color=ge.green)
    ax2.plot(t[::subsampling], data['Ii'][0][::subsampling], lw=1, color=ge.red)
    
    # Vm and spikes
    for i in range(Model['N_SEED']):
        ax3.plot(t[::subsampling], data['Vm'][i][::subsampling], lw=1., color='k')
    for tt in data['tspikes']:
        ax3.plot([tt,tt], [Vthre, Vthre+dVthre], ':', lw=1, color='k')
        
    graph_env.set_plot(ax1, ['left'], yticks=[], ylabel='%i\npresynaptic\n neurons' % j, xlim=[t[0], t[-1]])
    graph_env.set_plot(ax2, ['left'], ylabel='$I_{syn}$ (pA)', xlim=[t[0], t[-1]])
    graph_env.set_plot(ax3, xlabel='time (ms)', ylabel='$V_m$ (mV)', xlim=[t[0], t[-1]])
    
    return fig, [ax1, ax2, ax3]

def tf_2_variables(data,
                 xkey='F_RecExc', ckey='F_RecInh', output_key='Fout',
                 cond=None,
                 ckey_label='$\\nu_{i}$ (Hz)',
                 ylim=[1e-2, 100],
                 yticks=[0.01, 0.1, 1, 10],
                 yticks_labels=['0.01', '0.1', '1', '10'],
                 yscale='linear',
                 ylabel='$\\nu_{out}$ (Hz)',
                 xlim=None,
                 xticks=None,
                 xticks_labels=None,
                 xscale='linear',
                 cscale='log',
                 xlabel='$\\nu_{e}$ (Hz)',
                 cmap=pt.cm.copper, 
                 ax=None, acb=None,
                 fig_args={'figsize':(2., 1.3), 'right':0.7},
                 with_top_label=False,
                 ms=2, lw_th=2, alpha_th=0.7,
                 with_theory=False, th_discret=20):
    
    # limiting the data within the range
    Fout_mean, Fout_std = data[output_key+'_mean'], data[output_key+'_std']

    if ax is None:
        fig, ax = pt.figure(**fig_args)
        acb = pt.inset(ax, [1.2,0.1,0.1,0.85])
    else:
        fig = None

    if cond is None:
        cond = np.ones(len(data[xkey]), dtype=bool)
        
    F0, F1 = data[xkey][cond], data[ckey][cond]

    # color scale
    if cscale=='log':
        color_scale = (np.log(F1)-np.log(F1.min()))/(np.log(F1.max())-np.log(F1.min()))
    else:
        color_scale = np.linspace(0, 1, len(F1))

    mFout, sFout = Fout_mean[cond], Fout_std[cond]
    for i, f1 in enumerate(np.unique(F1)):
        i1 = np.argwhere(F1==f1).flatten()
        
        cond1 =  (mFout[i1]>=ylim[0]) & (mFout[i1]<ylim[1])
        ax.errorbar(F0[i1][cond1],
                    Fout_mean[i1][cond1],
                    yerr=Fout_std[i1][cond1],
                    fmt='o', ms=ms,
                    color=cmap(color_scale[i]))
        
        # # # now analytical estimate
        if with_theory:
            RATES = {xkey:np.concatenate([np.linspace(f1, f2, th_discret, endpoint=False)\
                                               for f1, f2 in zip(F0[i1][:-1], F0[i1][1:])])}
            RATES[ckey] = f1*np.ones(len(RATES[xkey]))
            Fout_th = TF(RATES, data['Model'], data['Model']['NRN_KEY'])
            th_cond = (Fout_th>ylim[0]) & (Fout_th<ylim[1])
            ax.plot(RATES[xkey][th_cond],
                    Fout_th[th_cond], '-',
                    color=cmap(color_scale[i]), lw=lw_th, alpha=alpha_th)
        
    if with_top_label:
        ax.set_title(col_key_label+'='+str(round(f1,1))+col_key_unit)
        
    for key in ['xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale']:
        exec('ax.set_%s(%s)' % (key,key))

    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # graph_env.set_plot(ax,
                # xlim=xlim,
                # ylim=ylim,
                # xticks=xticks,
                # yticks=yticks,
                # xlabel=xlabel,
                # ylabel=ylabel,
                # xscale=xscale,
                # yscale=yscale,
                # yticks_labels=yticks_labels,
                # xticks_labels=xticks_labels)

    # F1_ticks, F1_ticks_labels = [], []

    # cb = graph_env.build_bar_legend_continuous(acb , cmap,
                                               # bounds = [F1.min(), F1.max()],
                                               # scale='log')
    # cb.set_label(ckey_label)#, labelpad=labelpad, fontsize=fontsize, color=color)

    return fig, ax, acb


def tf_2_variables_3d(data,
                     Model=None,
                     xkey='F_RecInh', ykey='F_RecExc', zkey='Fout',
                     cond=None,
                     zlim=[0,100],
                     xlabel='$\\nu_{i}$ (Hz)',
                     ylabel='$\\nu_{e}$ (Hz)',
                     zlabel='$\\nu_{out}$ (Hz)',
                     ax=None,
                     x=None,
                     lw_th=3, alpha_th=0.5):
    
    # limiting the data within the range
    Fout_mean, Fout_std = data[zkey+'_mean'], data[zkey+'_std']

    if ax is None:
        fig, ax = pt.subplots()
    else:
        fig = None

    if cond is None:
        cond = np.ones(len(data[xkey]), dtype=bool)

    # to rotate the 
    if False:
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view)

    ax.view_init(10, -55)

    F0, F1 = data[ykey][cond], data[xkey][cond]

    ax.xaxis.labelpad=-10
    ax.yaxis.labelpad=-9
    ax.zaxis.labelpad=-10
    ax.tick_params(axis='x', which='major', pad=-6)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-4)

    mFout, sFout = Fout_mean[cond], Fout_std[cond]
    for i, f1 in enumerate(np.unique(F1)):
        i1 = np.argwhere(F1==f1).flatten()
        cond1 =  (mFout[i1]>=zlim[0]) & (mFout[i1]<zlim[1])
        ax.plot(np.ones(len(F0[i1][cond1]))*f1,
                F0[i1][cond1], 
                Fout_mean[i1][cond1],# 'o',
                color='gray', lw=2)
        
    print(data.keys())
    # # # now analytical estimate
    if ('COEFFS' in data['Model']):
        if x is None:
            x = np.linspace(np.min(F0), np.max(F0), 200)
        RATES = {}
        for key in data:
            if 'F_' in key:
                RATES[key] = 0*x+data[key][0] # to deal with other RATES like AffExc
        RATES[xkey], RATES[ykey] = x, x
        Fout_th = TF(RATES, data['Model'], data['Model']['NRN_KEY'])
        th_cond = (Fout_th<=zlim[1])
        ax.plot(RATES[xkey][th_cond],
                RATES[xkey][th_cond],
                Fout_th[th_cond], 'k--', lw=1)

    for key in ['xlabel', 'ylabel', 'zlabel']:
        exec('ax.set_%s(%s)' % (key,key))

    return fig, ax

