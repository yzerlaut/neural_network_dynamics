import sys
import numpy as np
from earlyVis_model import earlyVis_model
from stimuli import visual_stimulus
from plots import plot

try:
    Runcase=int(sys.argv[-1])
except ValueError:
    print('Provide aither an integer, the key "all", or a specific key (see run.py)')
    Runcase=str(sys.argv[-1])


np.random.seed(1)
Nshow = 8

if (Runcase==19) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/dense-noise.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-dense-noise.png')
    ps.show()
    
if (Runcase==18) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/sparse-noise.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-sparse-noise.png')
    ps.show()
    
if (Runcase==15) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/static-grating.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-static-grating.png')
    ps.show()
    
if (Runcase==14) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/drifting-grating.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-drifting-grating.png')
    ps.show()
    
if (Runcase==15) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/static-grating.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-static-grating.png')
    ps.show()
    
if (Runcase==14) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/drifting-grating.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-drifting-grating.png')
    ps.show()
    
if (Runcase==13) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model(from_file='data/grating-saccadic.npz')
    ps = plot(model=model)
    fig = ps.protocol_plot(cell_plot=np.random.choice(np.arange(model.Ncells), Nshow, replace=False))
    fig.savefig('docs/figs/response-static-grating-saccadic.png')
    ps.show()
    
if (Runcase==12) or (Runcase=='model-doc') or (Runcase=='all'):
    """
    demo fig of dynamic stimuli
    """
    model = earlyVis_model()
    ps = plot(model=model)
    
    fig, AX = ps.ge.figure(axes=(5,6), figsize=(.9,.7),
                           wspace=0.2, hspace=0.2, left=1.5, bottom=0.5, top=0.4, right=0.1)

    stim = visual_stimulus('drifting-grating')
    params = ' ($f$, $c$, $\\theta$, $\Psi$, $v_{dg}$)'
    ps.ge.annotate(AX[0][0], 'drifting\ngratings\n%s' % params, (-0.55,0.5), va='center', ha='center')
    ps.show_visual_stim_snapshots(stim, np.linspace(0, 1.2, 6), AX[0],
                                  with_time_annotation=True)

    stim = visual_stimulus('sparse-noise')
    params = ' ($\Gamma_{SN}$, $\sigma_{SN}$, $D_{SN}$, $J_{SN}$)'
    ps.ge.annotate(AX[1][0], 'sparse\nnoise\n%s' % params, (-0.55,0.5), va='center', ha='center')
    ps.show_visual_stim_snapshots(stim, np.linspace(0, 1.2, 6), AX[1],
                                  with_time_annotation=False)

    stim = visual_stimulus('dense-noise')
    params = ' ($\sigma_{DN}$, $D_{DN}$, $J_{DN}$)'
    ps.ge.annotate(AX[2][0], 'dense\nnoise\n%s' % params, (-0.55,0.5), va='center', ha='center')
    ps.show_visual_stim_snapshots(stim, np.linspace(0, 1.2, 6), AX[2],
                                  with_time_annotation=False)

    stim = visual_stimulus('gaussian-blob')
    params = ' ($c_{GB}$, $\sigma_{GB}$, $A_{GB}$,\n $\\tau_{DN}$, $t_{gb}$)'
    ps.ge.annotate(AX[3][0], 'gaussian blob\nappearance\n%s' % params,
                   (-0.55,0.5), va='center', ha='center')
    ps.show_visual_stim_snapshots(stim, np.linspace(0, 1.2, 6), AX[3],
                                  with_time_annotation=False)

    stim = visual_stimulus('center-surround')
    params = ' ($D_C$, $T_C$, $D_S$, $T_S$)'
    ps.ge.annotate(AX[4][0], 'centered-surround\nprotocols\n%s' % params,
                   (-0.55,0.5), va='center', ha='center')
    ps.show_visual_stim_snapshots(stim, np.linspace(0, 1.2, 6), AX[4],
                                  with_time_annotation=False)

    
    sax = ps.ge.arrow(fig, x0=0.1, y0=.1, dx=.87, dy=0.,
                      width=0.01, head_width=0.06)
    ps.ge.annotate(fig, 'time', (.5, .03), ha='center')
    
    fig.savefig('docs/figs/dynamic-stimuli.png')
    ps.show()
    
    
if (Runcase==11) or (Runcase=='model-doc') or (Runcase=='all'):
    """
    demo fig of static stimuli
    """
    model = earlyVis_model()
    stim = visual_stimulus('static-full-field-gratings')
    ps = plot(model=model)
    fig, AX = ps.ge.figure(axes=(5,6), figsize=(.9,.7), wspace=0.2, hspace=0.2, left=1.5, bottom=0., top=0.4, right=0.1)

    # full-field gratings
    params = ' ($f$, $c$, $\\theta$, $\Psi$)'
    ps.ge.annotate(AX[0][0], 'full-field\ngratings\n%s' % params, (-0.55,0.5), va='center', ha='center')
    ps.screen_plot(stim.full_array[1,:,:], ax=AX[0][0])
    for i, f, theta, psi, c in zip(range(6),
                              [0.1, 0.1, 0.1, 0.2, 0.02, 0.02],
                              [np.pi/6., np.pi/2., 3*np.pi/4., 3*np.pi/4., 3*np.pi/4., 3*np.pi/4.],
                              [np.pi, np.pi, np.pi, np.pi, np.pi, 0],
                              [1, 0.5, 1, 0.7, 0.8, 0.6]):
        stim.static_full_field_grating(theta=theta, spatial_freq=f, contrast=c, spatial_phase=psi)
        ps.screen_plot(stim.full_array[1,:,:], ax=AX[0][i], with_scale_bar=np.invert(bool(i)))

    # center gratings
    params = r'($f$, $c_c$, $\theta_c$, $\Psi_c$, $\vec{x_c}$, $r_c$)'
    ps.ge.annotate(AX[1][0], 'center\ngratings\n%s' % params, (-0.55,0.5), va='center', ha='center')
    for i, f, theta, psi, center, s, c in zip(range(6),
                                      [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
                                      [np.pi/6., np.pi/2., 3*np.pi/4., 3*np.pi/4., 3*np.pi/4., np.pi/8.],
                                      [np.pi, np.pi, 0, 0, np.pi, np.pi],
                                      [(40,20),(20,20),(40,20),(50,40),(20,40),(40,30)],
                                      [7,7,7,10,5,20],
                                      [1, 0.5, 1, 0.7, 1, 0.2]):
        stim.static_center_grating(center_theta=theta, center_spatial_freq=f, center=center,
                                   center_radius=s, center_contrast=c, center_spatial_phase=psi)

        ps.screen_plot(stim.full_array[1,:,:], ax=AX[1][i], with_scale_bar=np.invert(bool(i)))

    # # surround gratings
    params = r'($f$,$c_s$,$\theta_s$,$\Psi_s$,$\vec{x_c}$,$r_c$,$r_s$)'
    ps.ge.annotate(AX[2][0], 'surround\ngratings\n%s' % params, (-0.55,0.5), va='center', ha='center')
    # ps.screen_plot(stim.full_array[1,:,:], ax=AX[2][0])
    for i, f, c, theta, center, r1, r2 in zip(range(6),
                                              [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
                                              [1,1,0.5,1,1,1],
                                              [np.pi/6., np.pi/2., 3*np.pi/4., 3*np.pi/4., 3*np.pi/4., np.pi/8.],
                                           [(40,20),(20,20),(40,20),(50,40),(20,40),(40,30)],
                                           [10,10,10,3,5,20],
                                           [20,25,30,8,25,25]):
        stim.static_surround_grating(surround_theta=theta, surround_spatial_freq=f,
                                     center=center, center_radius=r1, surround_radius=r2,
                                     surround_contrast=c)
        ps.screen_plot(stim.full_array[1,:,:], ax=AX[2][i], with_scale_bar=np.invert(bool(i)))

    # # # center-surround gratings
    params = '($f$,$c_c$,$c_s$,$\\theta_c$,$\\theta_s$,\n'+r'$\Psi_c$,$\Psi_s$,$\vec{x_c}$,$r_c$,$r_s$)'
    ps.ge.annotate(AX[3][0], 'center-surround\ngratings\n%s' % params, (-0.55,0.5), va='center', ha='center')
    for i, f, c1, c2, theta1, theta2, psi, center, r1, r2 in zip(range(6),
                                           [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
                                           [1, 0.5, 1, 0.7, 1, 0.2],
                                           [1,1,0.5,1,1,1],
                                           [np.pi/6., np.pi/2., 3*np.pi/4., 3*np.pi/4., 3*np.pi/4., np.pi/8.],
                                           [np.pi/2., np.pi/4., 3*np.pi/4, np.pi/4., np.pi/4., np.pi],
                                           [np.pi, np.pi, 0, 0, np.pi, np.pi],
                                           [(40,20),(20,20),(40,20),(50,40),(20,40),(40,30)],
                                           [10,10,10,3,5,20],
                                           [20,25,30,8,25,25]):
        stim.static_center_surround_grating(center_theta=theta1, surround_theta=theta2,
                                            center_contrast=c1, surround_contrast=c2,
                                            surround_spatial_freq=f, center_spatial_freq=f,
                                            surround_spatial_phase=psi, center_spatial_phase=psi,
                                            center=center,
                                            center_radius=r1, surround_radius=r2)
        ps.screen_plot(stim.full_array[1,:,:], ax=AX[3][i], with_scale_bar=np.invert(bool(i)))

    
    # # # natural images
    ps.ge.annotate(AX[4][0], 'natural\nimages', (-0.55,0.5), va='center', ha='center')
    ps.screen_plot(stim.full_array[1,:,:], ax=AX[4][0])
    for i, theta2 in enumerate(np.linspace(0, .9*np.pi, 6)):
        stim.natural_images(image_number=i)
        ps.screen_plot(stim.full_array[1,:,:], ax=AX[4][i], with_scale_bar=np.invert(bool(i)))
        
    fig.savefig('docs/figs/static-stimuli.png')
    ps.show()
    
if (Runcase==10) or (Runcase=='model-doc') or (Runcase=='all'):
    """
    demo fig of SEM model
    """
    model = earlyVis_model()
    model.init_eye_movement('saccadic')
    plot_sim = plot(model=model)
    fig, _ = plot_sim.SEM_plot()
    fig.savefig('docs/figs/SEM-model.png')
    plot_sim.show()
    
if (Runcase==9) or (Runcase=='model-doc') or (Runcase=='all'):
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
    s[(t>1.2) & (t<1.25)] = -1
    
    AX[0].plot(t, s, 'k-')
    plot_sim.ge.set_plot(AX[0], ['left'], ylabel='  input\n  signal',
                         yticks=[-1,0,1], xlim=[t[0], t[-1]])
    r, a = model.temporal_filtering(t, s)
    AX[1].plot(t, a, '-', color=plot_sim.ge.purple, label='a(t)')
    AX[1].plot(t, r, '-', color=plot_sim.ge.blue, lw=2, label='r(t)')
    plot_sim.ge.annotate(AX[1], 'r(t)', (.6,0.), color=plot_sim.ge.blue)
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
if (Runcase==8) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model()
    model.draw_cell_RF_properties(model.Ncells, clustered_features=True)
    plot_sim = plot(model=model)
    icells = np.random.choice(np.arange(model.Ncells), 12)
    fig = plot_sim.plot_RF_properties(cell_list=icells)
    fig.savefig('docs/figs/RF.png')
    plot_sim.show()
if (Runcase==7) or (Runcase=='model-doc') or (Runcase=='all'):
    model = earlyVis_model()
    model.draw_cell_RF_properties(model.Ncells, clustered_features=True)
    plot_sim = plot(model=model)
    fig = plot_sim.show_cell_props_distrib()
    fig.savefig('docs/figs/cell-props.png')
    plot_sim.show()


if (Runcase==6) or (Runcase=='all'):
    model = earlyVis_model(from_file='data/drifting-grating.npz')
    plot_sim = plot(model=model)
    plot_sim.protocol_plot()
    plot_sim.show()
if (Runcase==5) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('drifting-grating', '')
    model.save_data('data/drifting-grating.npz')
if (Runcase==4) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('sparse-noise', 'saccadic')
    model.save_data('data/sparse-noise-saccadic.npz')
if (Runcase==3) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('dense-noise', 'saccadic')
    model.save_data('data/dense-noise.npz')
if (Runcase==2) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('sparse-noise', '')
    model.save_data('data/sparse-noise.npz')
if (Runcase==1) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('grating', 'saccadic')
    model.save_data('data/grating-saccadic.npz')
if (Runcase==0) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('grating', '')
    model.save_data('data/static-grating.npz')
if (Runcase==1) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('grating', 'saccadic')
    model.save_data('data/grating-saccadic.npz')
if (Runcase==0) or (Runcase=='all'):
    model = earlyVis_model()
    model.full_process('grating', '')
    model.save_data('data/static-grating.npz')
    
if (Runcase=='model-doc-plot'):
    
    model = earlyVis_model()
    np.random.seed(9)
    cell_list = np.random.choice(np.arange(model.Ncells), Nshow, replace=False)
    for stim in ['grating', 'drifting-grating', 'sparse-noise',
                 'dense-noise', 'center-surround', 'natural-image']:
        for em in ['', 'saccadic']:
            model = earlyVis_model(from_file='data/%s-%s.npz' % (stim,em))
            ps = plot(model=model)
            fig = ps.protocol_plot(cell_plot=cell_list)
            fig.savefig('docs/figs/response-%s-%s.png' % (stim,em))
            
if (Runcase=='model-doc-data'):
    for stim in ['grating', 'drifting-grating', 'sparse-noise',
                 'dense-noise', 'center-surround', 'natural-image']:
        for em in ['', 'saccadic']:
            model = earlyVis_model()
            model.full_process(stim, em)
            model.save_data('data/%s-%s.npz' % (stim,em))

if (Runcase=='model-seed-dep'):

    N_RF, N_STIM, N_SEM, N_POISSON = 3, 4, 3, 10
    for stim in ['grating', 'drifting-grating', 'sparse-noise',
                 'dense-noise', 'center-surround', 'natural-image']:
        print('-----------------------------------------------------')
        print('-----     %s                 -------------------' % stim)
        print('-----------------------------------------------------')
        for em in ['fixed', 'saccadic']:
            for RF_seed in np.arange(N_RF):
                for stim_seed in np.arange(N_STIM):
                    if (stim in ['grating', 'drifting-grating',
                                 'center-surround', 'natural-image']) and stim_seed>0:
                        pass
                    else:
                        for SEM_seed in np.arange(N_SEM):
                            if (em in ['fixed']) and SEM_seed>0:
                                pass
                            else:
                                model = earlyVis_model()
                                model.full_process(stim, em,
                                                   RF_seed=RF_seed,
                                                   em_seed=SEM_seed,
                                                   stim_seed=stim_seed)
                                model.save_data('data/%s-%s-RFseed-%i-StimSeed-%i-SEMseed-%i.npz' %\
                                                (stim,em,RF_seed,stim_seed,SEM_seed))

                                for poisson_seed in range(N_POISSON):

                                    model = earlyVis_model(from_file='data/%s-%s-RFseed-%i-StimSeed-%i-SEMseed-%i.npz' %\
                                                           (stim,em,RF_seed,stim_seed,SEM_seed))

                                    model.half_process2(seed=poisson_seed)
                                    model.save_data('data/%s-%s-RFseed-%i-StimSeed-%i-SEMseed-%i-PoissonSeed-%i.npz' %\
                                                    (stim,em,RF_seed,stim_seed,SEM_seed,poisson_seed))

    
