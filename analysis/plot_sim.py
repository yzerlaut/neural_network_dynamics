import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
from PIL import Image # BITMAP (png, jpg, ...)

B, O, G, R, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

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

######################################
#### RASTER PLOT
######################################
def raster(data,
           POP_KEYS = None, COLORS=None,
           NMAXS = None, tzoom=[0, np.inf],
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
                 (0, -.1), rotation=90, fontsize=14, xycoords='axes fraction')
    ax.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms',
                 (0., -0.1), fontsize=14, xycoords='axes fraction')
    set_plot(ax, [], yticks=[], xticks=[],
             xlim=[tzoom[0], min([ax.get_xlim()[1], tzoom[1]])],
             ylim=[0, NMAXS.sum()])
    return ax

######################################
#### TIME-VARYING ACTIVITIES
######################################

# for smoothing
from scipy.ndimage.filters import gaussian_filter1d
def gaussian_smoothing(signal, idt_sbsmpl=10):
    return gaussian_filter1d(signal, idt_sbsmpl)

def pop_act(data,
            POP_KEYS = None, COLORS=None,
            with_smoothing=0,
            tzoom=[0, np.inf],
            lw=2, ax=None):

    if POP_KEYS is None:
        POP_KEYS = find_pop_keys(data)
    if COLORS is None:
        COLORS = ['C'+str(i) for i in range(len(POP_KEYS))]
        
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4,2.5))
        plt.subplots_adjust(left=.3, bottom=.3)

    for pop_key, color in zip(POP_KEYS, COLORS):
        if with_smoothing>0:
            ax.plot(t[cond],
                    gaussian_smoothing(data['POP_ACT_'+pop_key][cond], int(with_smoothing/data['dt'])),
                    color=color, lw=lw)
        else:
            ax.plot(t[cond], data['POP_ACT_'+pop_key][cond], color=color, lw=lw)
            
    set_plot(ax, ylabel='pop. act. (Hz)', xlabel='time (ms)')
    
    return ax


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
        NVMS = np.array([range(len(data['VMS_'+pop_key])) for pop_key in POP_KEYS])
        
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    if ax is None:
        _, ax = plt.subplots(figsize=(5,3))
        plt.subplots_adjust(left=.15, bottom=.1, right=.99)
    
    cond = (t>tzoom[0]) & (t<tzoom[1])

    nn = 0
    for VmID, pop_key, color in zip(NVMS, POP_KEYS, COLORS):
        threshold, rest = data[pop_key+'_Vthre'], data[pop_key+'_El']
        if pop_key+'_delta_v' in data:
            threshold += data[pop_key+'_delta_v']
        for i in VmID:
            nn+=1
            ax.plot(t[cond], data['VMS_'+pop_key][i][cond]+shift*nn, color=color, lw=lw)
            # adding spikes
            tspikes = data['tRASTER_'+str(pop_key)][np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
            Scond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
            for ts in tspikes[Scond]:
                ax.plot([ts, ts], shift*nn+np.array([threshold, vpeak]), '--', color=color, lw=lw)
            ax.plot([t[cond][0], t[cond][-1]], shift*nn+np.array([rest, rest]), ':', color=color, lw=lw)

    y0 = ax.get_ylim()[0]
    ax.plot([tzoom[0],tzoom[0]+Tbar], y0*np.ones(2),
                 lw=2, color='k')
    ax.annotate(str(int(Tbar))+' ms', (tzoom[0], .9*y0), fontsize=14)
    ax.plot([tzoom[0],tzoom[0]], y0+np.arange(2)*Vbar,
                 lw=2, color='k')
    ax.annotate(str(int(Vbar))+' mV', (tzoom[0], y0+Vbar), fontsize=14)
    set_plot(ax, [], xticks=[], yticks=[])
    
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

    
