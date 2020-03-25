import sys
import numpy as np
from earlyVis_model import earlyVis_model
from plots import plot

try:
    Runcase=int(sys.argv[-1])
except ValueError:
    Runcase='all'


if (Runcase==10) or (Runcase is 'model-doc') or (Runcase is 'all'):
    """
    demo fig of SEM model
    """
    model = earlyVis_model()
    model.init_eye_movement('saccadic')
    plot_sim = plot(model=model)
    fig, _ = plot_sim.SEM_plot()
    # fig, AX = plot_sim.ge.figure(axes_extents=[[[2,1]],
    #                                            [[2,1]],
    #                                            [[2,1]],
    #                                            [[2,2]]],
    #                              figsize=(.9,.45), hspace=0.5, left=.9, right=.5, bottom=1.2)
    
    # t = np.arange(1500)*1e-3
    # model.t, model.dt = t, 1e-3

    # s = np.array([1 if tt>0.4 else 0 for tt in t])
    # s[(t>0.1) & (t<0.2)] = -1
    # AX[0].plot(t, s, 'k-')
    # plot_sim.ge.set_plot(AX[0], ['left'], ylabel='  input\n  signal',
    #                      yticks=[-1,0,1], xlim=[t[0], t[-1]])
    # r, a = model.temporal_filtering(t, s)
    # AX[1].plot(t, a, '-', color=plot_sim.ge.purple, label='a(t)')
    # AX[1].plot(t, r, '-', color=plot_sim.ge.blue, lw=2, label='r(t)')
    # plot_sim.ge.annotate(AX[1], 'r(t)', (.6,-.1), color=plot_sim.ge.blue)
    # plot_sim.ge.annotate(AX[1], 'a(t)', (.6,1), color=plot_sim.ge.purple)
    # plot_sim.ge.set_plot(AX[1], ['left'], ylabel='processed\nsignal', xlim=[t[0], t[-1]])
    # model.RATES = [model.compute_rates(r)]
    # AX[2].plot(t, model.RATES[0], 'k-')
    # plot_sim.ge.set_plot(AX[2], ['left'], ylabel='Rate  \n(Hz)  ', xlim=[t[0], t[-1]])
    # for i in range(10):
    #     model.Poisson_process_transform(seed=i+20)
    #     for spk in model.SPIKES[0]:
    #         AX[3].plot([spk,spk], [0.1*i,0.1*(i+1)], color=plot_sim.ge.brown)
    # plot_sim.ge.set_plot(AX[3], ylabel='Spike output\n(for 10 seeds)', yticks=[], xlabel='time (s)',
    #                      ylim=[0,1], xlim=[t[0], t[-1]])
    fig.savefig('docs/figs/SEM-model.png')
    plot_sim.show()
    
if (Runcase==9) or (Runcase is 'model-doc') or (Runcase is 'all'):
    """
    demo fig of LNP model
    """
    model = earlyVis_model()
    plot_sim = plot(model=model)
    fig, AX = plot_sim.ge.figure(axes_extents=[[[2,1]],
                                               [[2,1]],
                                               [[2,1]],
                                               [[2,2]]],
                                 figsize=(.9,.45), hspace=0.5, left=.9, right=.5, bottom=1.2)
    
    t = np.arange(1500)*1e-3
    model.t, model.dt = t, 1e-3

    s = np.array([1 if tt>0.4 else 0 for tt in t])
    s[(t>0.1) & (t<0.2)] = -1
    AX[0].plot(t, s, 'k-')
    plot_sim.ge.set_plot(AX[0], ['left'], ylabel='  input\n  signal',
                         yticks=[-1,0,1], xlim=[t[0], t[-1]])
    r, a = model.temporal_filtering(t, s)
    AX[1].plot(t, a, '-', color=plot_sim.ge.purple, label='a(t)')
    AX[1].plot(t, r, '-', color=plot_sim.ge.blue, lw=2, label='r(t)')
    plot_sim.ge.annotate(AX[1], 'r(t)', (.6,-.1), color=plot_sim.ge.blue)
    plot_sim.ge.annotate(AX[1], 'a(t)', (.6,1), color=plot_sim.ge.purple)
    plot_sim.ge.set_plot(AX[1], ['left'], ylabel='processed\nsignal', xlim=[t[0], t[-1]])
    model.RATES = [model.compute_rates(r)]
    AX[2].plot(t, model.RATES[0], 'k-')
    plot_sim.ge.set_plot(AX[2], ['left'], ylabel='Rate  \n(Hz)  ', xlim=[t[0], t[-1]])
    for i in range(10):
        model.Poisson_process_transform(seed=i+20)
        for spk in model.SPIKES[0]:
            AX[3].plot([spk,spk], [0.1*i,0.1*(i+1)], color=plot_sim.ge.brown)
    plot_sim.ge.set_plot(AX[3], ylabel='Spike output\n(for 10 seeds)', yticks=[], xlabel='time (s)',
                         ylim=[0,1], xlim=[t[0], t[-1]])
    fig.savefig('docs/figs/LNP-model.png')
    plot_sim.show()
if (Runcase==8) or (Runcase is 'model-doc') or (Runcase is 'all'):
    model = earlyVis_model()
    model.draw_cell_RF_properties(model.Ncells, clustered_features=True)
    plot_sim = plot(model=model)
    icells = np.random.choice(np.arange(model.Ncells), 12)
    fig = plot_sim.plot_RF_properties(cell_list=icells)
    fig.savefig('docs/figs/RF.png')
    plot_sim.show()
if (Runcase==7) or (Runcase is 'model-doc') or (Runcase is 'all'):
    model = earlyVis_model()
    model.draw_cell_RF_properties(model.Ncells, clustered_features=True)
    plot_sim = plot(model=model)
    fig = plot_sim.show_cell_props_distrib()
    fig.savefig('docs/figs/cell-props.png')
    plot_sim.show()
if (Runcase==6) or (Runcase is 'all'):
    model = earlyVis_model(from_file='data/drifting-grating.npz')
    plot_sim = plot(model=model)
    plot_sim.protocol_plot()
    plot_sim.show()
if (Runcase==5) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('drifting-grating', '', seed=3)
    model.save_data('data/drifting-grating.npz')
if (Runcase==4) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('sparse-noise', 'saccadic', seed=3)
    model.save_data('data/sparse-noise-saccadic.npz')
if (Runcase==3) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('dense-noise', '', seed=3)
    model.save_data('data/dense-noise.npz')
if (Runcase==2) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('sparse-noise', '', seed=3)
    model.save_data('data/sparse-noise.npz')
if (Runcase==1) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('grating', 'saccadic', seed=3)
    model.save_data('data/grating-saccadic.npz')
if (Runcase==0) or (Runcase is 'all'):
    model = earlyVis_model()
    model.full_process('drifting-grating', 'saccadic', seed=3)
    model.save_data('data/drifting-grating-saccadic.npz')
