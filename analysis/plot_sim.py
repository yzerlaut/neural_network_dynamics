import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *

def raster_fig(data,
             tzoom=[0, np.inf],
             COLORS=['g', 'r', 'k', 'y'], NVm=3, Nnrn=500, Tbar=50):
    
    fig, ax = plt.subplots(1, figsize=(2.3,2))
    plt.subplots_adjust(left=.05, bottom=.2)
    # raster activity
    nn = 0
    cond = (data['tRASTER_Exc']>tzoom[0]) & (data['tRASTER_Exc']<tzoom[1])
    plt.plot(data['tRASTER_Exc'][cond], data['iRASTER_Exc'][cond], '.', color='g', ms=1)
    nn+= data['iRASTER_Exc'].max()
    cond = (data['tRASTER_Inh']>tzoom[0]) & (data['tRASTER_Inh']<tzoom[1])
    plt.plot(data['tRASTER_Inh'][cond], nn+data['iRASTER_Inh'][cond], '.', color='r', ms=1)
    plt.plot(tzoom[0]*np.ones(2), [0, Nnrn], lw=5, color='gray')
    plt.annotate(str(Nnrn)+' neurons',\
                 (0, .7), rotation=90, fontsize=14, xycoords='figure fraction')
    # plt.ylabel(str(Nnrn)+' neurons', fontsize=14)
    plt.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    plt.annotate(str(Tbar)+' ms',
                 (0.2, 0.1), fontsize=14, xycoords='figure fraction')
    # plt.xlabel(str(Tbar)+' ms            ', fontsize=14)
    set_plot(ax, [], yticks=[], xticks=[])
    return fig

def preraster_fig(data,
             tzoom=[0, np.inf],
             COLORS=['g', 'r', 'k', 'y'], NVm=3, Nnrn=500, Tbar=50):
    
    fig, ax = plt.subplots(1, figsize=(2.3,2))
    plt.subplots_adjust(left=.05, bottom=.2)
    # raster activity
    nn = 0
    cond = (data['tRASTER_Exc']>tzoom[0]) & (data['tRASTER_Exc']<tzoom[1])
    plt.plot(data['tRASTER_Exc'][cond], data['iRASTER_Exc'][cond], '.', color='g', ms=1)
    nn+= data['iRASTER_Exc'].max()
    cond = (data['tRASTER_Inh']>tzoom[0]) & (data['tRASTER_Inh']<tzoom[1])
    plt.plot(data['tRASTER_Inh'][cond], nn+data['iRASTER_Inh'][cond], '.', color='r', ms=1)
    plt.plot(tzoom[0]*np.ones(2), [0, Nnrn], lw=5, color='gray')
    plt.annotate(str(Nnrn)+' neurons',\
                 (0, .7), rotation=90, fontsize=14, xycoords='figure fraction')
    # plt.ylabel(str(Nnrn)+' neurons', fontsize=14)
    plt.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    plt.annotate(str(Tbar)+' ms',
                 (0.2, 0.1), fontsize=14, xycoords='figure fraction')
    # plt.xlabel(str(Tbar)+' ms            ', fontsize=14)
    set_plot(ax, [], yticks=[], xticks=[])
    return fig

def Vm_Isyn_fig(data, pop_key='Exc',
                tzoom=[0, np.inf], NVm=3,
                COLORS=['g', 'r', 'k', 'y'], Nnrn=500, Tbar=50, vpeak=-40):

    NVm= max([NVm, len(data['VMS_'+str(pop_key)])])
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    fig, AX = plt.subplots(2, NVm, figsize=(4*NVm, 4))
    cond = (t>tzoom[0]) & (t<tzoom[1])
    for i in range(NVm):
        AX[0,i].plot(t[cond], data['ISYNe_'+str(pop_key)][i][cond], color='g')
        AX[0,i].plot(t[cond], data['ISYNi_'+str(pop_key)][i][cond], color='r')
        AX[1,i].plot(t[cond], data['VMS_'+str(pop_key)][i][cond], color='k')
        AX[0,i].set_xticklabels([]);AX[1,i].set_xticklabels([])
        AX[1,i].set_ylim([-75, vpeak])
        if i>0:
            AX[0,i].set_yticklabels([])
            AX[1,i].set_yticklabels([])
    # adding spikes
    for i in range(NVm):
        tspikes = data['tRASTER_'+str(pop_key)][np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
        cond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
        for ts in tspikes[cond]:
            AX[1,i].plot([ts, ts], [-50, vpeak], 'k--')
    
    AX[0,0].set_ylabel('current (pA)')
    AX[1,0].set_ylabel('$V_m$ (mV)')
    imax = 0
    for ax in AX[0,:]:
        imax = max(np.abs(ax.get_ylim()))
    for ax in AX[0,:]:
        ax.set_ylim([-imax, imax])
            
    return fig

def exc_inh_balance(data, pop_key='Exc'):
    
    NVm= len(data['VMS_'+str(pop_key)])
    
    fig, ax = plt.subplots(1,figsize=(1.2,2.))
    
    # excitation
    mean = np.mean([data['ISYNe_'+str(pop_key)][i].mean() for i in range(NVm)])
    std = np.std([data['ISYNe_'+str(pop_key)][i].mean() for i in range(NVm)])
    ax.bar([0], mean, yerr=std, edgecolor='g', facecolor='w', lw=3,
           error_kw={'ecolor':'g','linewidth':3}, capsize=3)

    # inhibition
    mean = -np.mean([data['ISYNi_'+str(pop_key)][i].mean() for i in range(NVm)])
    std = np.std([data['ISYNi_'+str(pop_key)][i].mean() for i in range(NVm)])
    ax.bar([1], mean, yerr=std, edgecolor='r', facecolor='w', lw=3,
           error_kw={'ecolor':'r','linewidth':3}, capsize=3)
    
    # ax.bar([1], -np.mean([np.mean(ISYNi[0][i].Ii/ntwk.pA) for i in range(4)]), color='r')
    set_plot(ax, ylabel='mean currents \n (abs. value, pA)',
             xticks=[0,1], xticks_labels=['exc.', 'inh.'])

def assemble_quantities(data):

    ntwk.raster_fig(bs_data, tzoom=[800,870]);
