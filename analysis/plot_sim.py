import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
from PIL import Image # BITMAP (png, jpg, ...)

G, R = 'g', 'r'

def raster_fig(data,
             tzoom=[0, np.inf],
             COLORS=['g', 'r', 'k', 'y'], NVm=3, Nnrn=500, Tbar=50):
    
    fig, ax = plt.subplots(1, figsize=(2.3,2))
    plt.subplots_adjust(left=.05, bottom=.2)
    # raster activity
    nn = 0
    try:
        cond = (data['tRASTER_Exc']>tzoom[0]) & (data['tRASTER_Exc']<tzoom[1])
        plt.plot(data['tRASTER_Exc'][cond], data['iRASTER_Exc'][cond], '.', color=G, ms=1)
        nn+= data['iRASTER_Exc'].max()
        cond = (data['tRASTER_Inh']>tzoom[0]) & (data['tRASTER_Inh']<tzoom[1])
        plt.plot(data['tRASTER_Inh'][cond], nn+data['iRASTER_Inh'][cond], '.', color=R, ms=1)
    except ValueError:
        pass
    plt.plot(tzoom[0]*np.ones(2), [0, Nnrn], lw=5, color='gray')
    plt.annotate(str(Nnrn)+' neurons',\
                 (0, .7), rotation=90, fontsize=14, xycoords='figure fraction')
    # plt.ylabel(str(Nnrn)+' neurons', fontsize=14)
    plt.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    plt.annotate(str(Tbar)+' ms',
                 (0.2, 0.1), fontsize=14, xycoords='figure fraction')
    # plt.xlabel(str(Tbar)+' ms            ', fontsize=14)
    set_plot(ax, [], yticks=[], xticks=[], xlim=tzoom,
             ylim=[0, data['0']['N']+data['1']['N']])
    return fig

def pop_act(data, tdiscard=200, Tbar=50):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = t>tdiscard
    
    fig, ax = plt.subplots(1, figsize=(2.6,2))
    plt.subplots_adjust(left=.33, bottom=.2)

    BOTTOM, dmin = -3, 0.5 # 0.01 taken as the lower value for bar plot
    ax.bar([0], np.log(data['faff'])/np.log(10)-BOTTOM)
    ax.bar([1], np.log(data['fdsnh'])/np.log(10)-BOTTOM)
    for i, f, color in zip(range(2,4), [data['POP_ACT_Exc'], data['POP_ACT_Inh']], [R, G]):
        mean = f[cond].mean()
        if mean<0.001:
            mean = 0.001*(1+dmin) # to get a visible value at 0
        lmean = np.log(mean)/np.log(10)
        std1 = lmean-np.log(max([1e-2, mean-f[cond].std()]))/np.log(10)
        std2 = np.log(mean+f[cond].std())/np.log(10)-lmean
        ax.bar([i], [lmean-BOTTOM], yerr=[[std1], [std2]])
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-0.09,0), (0.05,0.08), **kwargs)
    ax.plot((-0.09,0), (0.065,0.095), **kwargs)
    ax.plot([0], [2.1-BOTTOM], 'w.', alpha=.1)
    set_plot(ax, ['left'], xticks=np.arange(4),
             yticks=np.array([-3,-2,-1,0,1,2])-BOTTOM, ylabel='pop. act. (Hz)',
             yticks_labels=['0', '0.01', '0.1', '1', '10', '100'],
             xticks_labels=['$\\nu_a$', '$\\nu_d$', '$\\nu_e$', '$\\nu_i$'])
    
    return fig

def Vm_Isyn_fig(data, pop_key='Exc',
                tzoom=[0, np.inf], NVm=3,
                COLORS=['g', 'r', 'k', 'y'], Nnrn=500, Tbar=50,
                vpeak=-40, vbottom=-80):

    NVm= max([NVm, len(data['VMS_'+str(pop_key)])])
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    fig, AX = plt.subplots(2, NVm, figsize=(2*NVm, 3))
    plt.subplots_adjust(left=.15, bottom=.1, right=.99)
    
    cond = (t>tzoom[0]) & (t<tzoom[1])
    for i in range(NVm):
        AX[0,i].plot(t[cond], data['ISYNe_'+str(pop_key)][i][cond], color='g')
        AX[0,i].plot(t[cond], data['ISYNi_'+str(pop_key)][i][cond], color='r')
        AX[1,i].plot(t[cond], data['VMS_'+str(pop_key)][i][cond], color='k')
        AX[0,i].set_xticklabels([]);AX[1,i].set_xticklabels([])
        AX[1,i].set_ylim([vbottom, vpeak])
        if i>0:
            AX[0,i].set_yticklabels([])
            AX[1,i].set_yticklabels([])
    # adding spikes
    for i in range(NVm):
        tspikes = data['tRASTER_'+str(pop_key)][np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
        cond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
        for ts in tspikes[cond]:
            AX[1,i].plot([ts, ts], [-50, vpeak], 'k--')
    
    AX[0,0].plot([tzoom[0],tzoom[0]+Tbar], AX[0,0].get_ylim()[0]*np.ones(2),
                 lw=5, color='gray')
    AX[0,0].annotate(str(Tbar)+' ms', (tzoom[0], .9*AX[0,0].get_ylim()[0]), fontsize=14)
    
    imax = 0
    for ax in AX[0,:]:
        imax = max([imax, max(np.abs(ax.get_ylim()))])
    for ax in AX[0,:]:
        ax.plot(tzoom, [-imax, imax], 'w.', alpha=0.01)
        ax.set_ylim([-imax, imax])
    for ax in AX[1,:]:
        ax.plot(tzoom, [vbottom, vpeak], 'w.', alpha=0.01)
        
        
    for i in range(NVm):
        AX[1,i].set_title('cell '+str(i+1))
        if i==0:
            set_plot(AX[0,i], ['left'], ylabel='current (pA)', xticks=[], num_yticks=3)
            set_plot(AX[1,i], ['left'], ylabel='$V_m$ (mV)', xticks=[],
                     yticks=[-70,-60,-50])
        else:
            set_plot(AX[0,i], ['left'], xticks=[], num_yticks=3)
            set_plot(AX[1,i], ['left'], xticks=[], yticks=[-70,-60,-50])
    return fig

def few_Vm_fig(data, pop_key='Exc',
               tzoom=[0, np.inf], NVm=3,
               COLORS=['g', 'r', 'k', 'y'], Nnrn=500, Tbar=50,
               vpeak=-40, vbottom=-80):

    NVm= max([NVm, len(data['VMS_'+str(pop_key)])])
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    fig, ax = plt.subplots(figsize=(3,1))
    # plt.subplots_adjust(left=.15, bottom=.1, right=.99)
    
    cond = (t>tzoom[0]) & (t<tzoom[1])
    for i in range(NVm):
        ax.plot(t[cond], data['VMS_'+str(pop_key)][i][cond]+10*i, color='k')
    # adding spikes
    for i in range(NVm):
        tspikes = data['tRASTER_'+str(pop_key)][np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
        cond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
        for ts in tspikes[cond]:
            ax.plot([ts, ts], 10*i+np.array([-50, vpeak]), 'k--')
    ax.plot([tzoom[0],tzoom[0]+Tbar], ax.get_ylim()[0]*np.ones(2),
                 lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms', (tzoom[0], .9*ax.get_ylim()[0]), fontsize=14)
    set_plot(ax, [], xticks=[], yticks=[])
    
    return fig

def exc_inh_balance(data, pop_key='Exc'):
    
    NVm= len(data['VMS_'+str(pop_key)])
    
    fig, ax = plt.subplots(1,figsize=(2.5,2))
    plt.subplots_adjust(left=.5, bottom=.2)
    
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
    
    set_plot(ax, ylabel='mean currents \n (abs. value, pA)',
             xticks=[0,1], xticks_labels=['exc.', 'inh.'])
    return fig

def histograms(data, pop_key='Exc'):
    
    NVm= len(data['VMS_'+str(pop_key)])
    
    fig, AX = plt.subplots(2, 2, figsize=(4,3))
    plt.subplots_adjust(bottom=.3, top=.99, right=.99)

    ######## VM ########
    # excitation
    for i in range(NVm):
        hist, be = np.histogram(data['VMS_Exc'][i], bins=20, normed=True)
        AX[0, 0].plot(.5*(be[1:]+be[:-1]), hist, color=G, lw=.5)
    hist, be = np.histogram(np.concatenate(data['VMS_Exc']), bins=20, normed=True)
    AX[0, 0].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    set_plot(AX[0, 0], ['bottom'], yticks=[],
             xticks=[-70, -60, -50], xticks_labels=[], xlim=[-75,-45])
    
    # inhibition
    for i in range(NVm):
        hist, be = np.histogram(data['VMS_Inh'][i], bins=20, normed=True)
        AX[1, 0].plot(.5*(be[1:]+be[:-1]), hist, color=R, lw=.5)
    hist, be = np.histogram(np.concatenate(data['VMS_Inh']), bins=20, normed=True)
    AX[1, 0].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=R, lw=0,
                 width=be[1]-be[0], facecolor=R, alpha=.3)
    set_plot(AX[1, 0], ['bottom'], xlabel='$V_m$ (mV)', yticks=[],
             xticks=[-70, -60, -50], xlim=[-75,-45])

    ######## CURRENTS ########
    # on excitatory population
    hist, be = np.histogram(np.concatenate(data['ISYNe_Exc']), bins=20, normed=True)
    AX[0, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    hist, be = np.histogram(np.concatenate(data['ISYNi_Exc']), bins=20, normed=True)
    AX[0, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=R, lw=0,
                 width=be[1]-be[0], facecolor=R, alpha=.3)
        
    # on excitatory population
    hist, be = np.histogram(np.concatenate(data['ISYNe_Inh']), bins=20, normed=True)
    AX[1, 1].bar(.5*(be[1:]+be[:-1]), hist, edgecolor=G, lw=0,
                 width=be[1]-be[0], facecolor=G, alpha=.3)
    hist, be = np.histogram(np.concatenate(data['ISYNi_Inh']), bins=20, normed=True)
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


def assemble_quantities(data, filename, tzoom=[800,1200]):

    new_im = Image.new('RGBA', (830, 1140), (255,255,255,255))

    # title
    fig = plt.figure(figsize=(3,.3))
    fig.text(0., 0.2,
             '$\\nu_{aff}$='+str(data['faff'])+'Hz, $\\nu_{dsnh}$='+\
             str(data['fdsnh'])+'Hz', fontsize=14)
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
    fig = pop_act(data)
    fig.text(0., .9, '(ii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (250, 40))
    # exc inh balance
    fig = exc_inh_balance(data, pop_key='Exc')
    fig.text(0.5, .9, '(iii)', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (550, 40))
    # Vm traces excitation
    fig = Vm_Isyn_fig(data, pop_key='Exc', tzoom=tzoom)
    fig.text(0.2, .92, '(iv) excitatory cells sample', fontsize=14, weight='bold')
    fig.savefig('temp.png')
    im = Image.open('temp.png')
    new_im.paste(im, (20, 250))
    # Vm traces inhibition
    fig = Vm_Isyn_fig(data, pop_key='Inh', tzoom=tzoom)
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
    fig = few_Vm_fig(data, pop_key='Exc', tzoom=[tzoom[0], tzoom[0]+20], Tbar=5)
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
                    '../../params_scan/scan.zip')
    assemble_quantities(DATA[-1], 'data/0.png')

    
