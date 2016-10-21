"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

import sys, os, time
sys.path.append('../../')
from ntwk_build.syn_and_connec_construct import build_populations,\
    build_up_recurrent_connections,\
    initialize_to_rest
from ntwk_build.syn_and_connec_library import get_connectivity_and_synapses_matrix, print_parameters
from ntwk_stim.waveform_library import double_gaussian, ramp_rise_then_constant
from ntwk_stim.connect_afferent_input import construct_feedforward_input
from cells.cell_library import initialize_AdExp_parameters, fill_NRN_params_with_values

def run_sim(args):
    ### SIMULATION PARAMETERS

    print('[initializing simulation ...]')
    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    exc_nrn_params = {'name':'exc. cell', 'N':int(args.Ntot*(1-args.gei))}
    for key in vars(args):
        k = key.split('_exc')
        if len(k)>1:
            exc_nrn_params[k[0]] = vars(args)[key]
    inh_nrn_params = {'name':'inh. cell', 'N':int(args.Ntot*args.gei)}
    for key in vars(args):
        k = key.split('_inh')
        if len(k)>1:
            inh_nrn_params[k[0]] = vars(args)[key]

    NTWK = [{'params':exc_nrn_params}, {'params':inh_nrn_params}]
    
    AFFERENCE_ARRAY = [{'Q':args.Qe, 'N':int(args.Ntot*(1-args.gei)), 'pconn':args.pconnec},
                       {'Q':args.Qe, 'N':int(args.Ntot*(1-args.gei)), 'pconn':args.pconnec}]
    rate_array = args.ext_drive+0.*t_array
    
    EXC_ACTS, INH_ACTS, SPK_TIMES, SPK_IDS = [], [], [], []

    M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK), verbose=args.verbose)
    if args.Qe!=0:
        M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
    if args.Qi!=0:
        M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi

    POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True, verbose=args.verbose)

    initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

    AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                         AFFERENCE_ARRAY,\
                                                         t_array,
                                                         rate_array,\
                                                         pop_for_conductance='A',
                                                         SEED=args.SEED)
    SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=args.SEED+1)

    EXC_SPIKES, INH_SPIKES = brian2.SpikeMonitor(POPS[0]), brian2.SpikeMonitor(POPS[1])
    trace_Vm_exc = brian2.StateMonitor(POPS[0], 'V', record=np.arange(5))
    trace_Vm_inh = brian2.StateMonitor(POPS[1], 'V', record=np.arange(5))
    trace_Ge = brian2.StateMonitor(POPS[0], 'GAA', record=10)
    trace_Gi = brian2.StateMonitor(POPS[0], 'GBA', record=10)

    net = brian2.Network(brian2.collect())
    # manually add the generated quantities
    net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, EXC_SPIKES, INH_SPIKES) 
    print('[running simulation ...]')
    net.run(args.tstop*brian2.ms)
    print('[simulation done -> saving output]')

    EXC_ACTS = POP_ACT[0].smooth_rate(window='flat',\
                                           width=args.smoothing*brian2.ms)/brian2.Hz
    INH_ACTS = POP_ACT[1].smooth_rate(window='flat',\
                                           width=args.smoothing*brian2.ms)/brian2.Hz

    np.savez(args.filename, args=args,
             t_array=t_array,
             exc_act = np.array(EXC_ACTS),
             inh_act = np.array(INH_ACTS),
             exc_spk = np.array(EXC_SPIKES.t),
             inh_spk = np.array(INH_SPIKES.t),
             exc_ids = np.array(EXC_SPIKES.i),
             inh_ids = np.array(INH_SPIKES.i),
             Vm_exc = [np.array(x.V/brian2.mV) for x in trace_Vm_exc],
             Vm_inh = [np.array(x.V/brian2.mV) for x in trace_Vm_inh],
             Ge = np.array(trace_Ge[10].GAA/brian2.nS),
             Gi = np.array(trace_Gi[10].GBA/brian2.nS),
             infos = print_parameters([exc_nrn_params, inh_nrn_params], vars(args)),
             plot=get_plotting_instructions())
    os.system('cp '+args.filename+ ' /tmp/'+time.strftime("%Y_%m_%d-%H:%M:%S")+'.npz')
    
    
def get_plotting_instructions():
    return """
args = data['args'].all()
from graphs.ntwk_dyn_plot import RASTER_PLOT, POP_ACT_PLOT
RASTER_PLOT([1e3*data['exc_spk'],1e3*data['inh_spk']], [data['exc_ids'],data['inh_ids']], MS=2)
POP_ACT_PLOT(data['t_array'], [data['exc_act'],data['inh_act']])
from graphs.my_graph import set_plot
fig = plt.figure(figsize=(5,3))
plt.subplots_adjust(left=.25, bottom=.25)
for i in range(len(data['Vm_exc'])):
    plt.plot(data['t_array'], data['Vm_exc'][i]-i*20., 'g-')
    for k in np.argwhere(data['exc_ids']==i).flatten():
       plt.plot(1e3*data['exc_spk'][k]*np.ones(2), [-50-i*20, -i*20], 'g-')
plt.plot([50,50], [-30,-10], 'k-', lw=4)
plt.annotate('20mV', (60,-7))
set_plot(plt.gca(), ['bottom'], ylabel='$V_m$ (mV)', xlabel='time (ms)', yticks=[])
fig4 = plt.figure(figsize=(5,3))
plt.subplots_adjust(left=.25, bottom=.25)
for i in range(len(data['Vm_inh'])):
    plt.plot(data['t_array'], data['Vm_inh'][i]-i*20., 'r-')
    for k in np.argwhere(data['inh_ids']==i).flatten():
       plt.plot(1e3*data['inh_spk'][k]*np.ones(2), [-54-i*20, -4-i*20], 'r-')
plt.plot([50,50], [-30,-10], 'k-', lw=4)
plt.annotate('20mV', (60,-10))
set_plot(plt.gca(), ['bottom'], ylabel='$V_m$ (mV)', xlabel='time (ms)', yticks=[])
fig7 = plt.figure(figsize=(5,3))
plt.subplots_adjust(left=.25, bottom=.25)
plt.plot(data['t_array'], data['Gi'], 'r-', label='inh.')
plt.plot(data['t_array'], data['Ge'], 'g-', label='exc.')
plt.legend(frameon=False, prop={'size':'x-small'})
set_plot(plt.gca(), ylabel='$G$ (nS)', xlabel='time (ms)')
fig8 = plt.figure(figsize=(5,3))
plt.gca().axis('off')
plt.annotate(data['infos'], (0, 0), fontsize=3)
"""


if __name__=='__main__':
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Investigates what is the network response of a single spike 
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    # simulation parameters
    parser.add_argument("--DT",help="simulation time step (ms)",type=float, default=0.1)
    parser.add_argument("--tstop",help="simulation duration (ms)",type=float, default=200.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=1)
    parser.add_argument("--SEED",help="seed for numerical sims", type=int, default=3)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=2.)
    # excitatory cells 
    parser.add_argument("--Cm_exc", type=float, default=200.)
    parser.add_argument("--Gl_exc", type=float, default=10.)
    parser.add_argument("--Vthre_exc", type=float, default=-50.)
    parser.add_argument("--Vreset_exc", type=float, default=-70.)
    parser.add_argument("--El_exc", type=float, default=-70.)
    parser.add_argument("--Trefrac_exc", type=float, default=5.)
    parser.add_argument("--a_exc", type=float, default=0.)
    parser.add_argument("--tauw_exc", type=float, default=500.)
    parser.add_argument("--b_exc", type=float, default=10.)
    parser.add_argument("--delta_v_exc", type=float, default=0.)
    # inhibitory cells 
    parser.add_argument("--Cm_inh", type=float, default=200.)
    parser.add_argument("--Gl_inh", type=float, default=10.)
    parser.add_argument("--Vthre_inh", type=float, default=-54.)
    parser.add_argument("--Vreset_inh", type=float, default=-70.)
    parser.add_argument("--El_inh", type=float, default=-70.)
    parser.add_argument("--Trefrac_inh", type=float, default=5.)
    parser.add_argument("--a_inh", type=float, default=0.)
    parser.add_argument("--tauw_inh", type=float, default=1e9)
    parser.add_argument("--b_inh", type=float, default=0.)
    parser.add_argument("--delta_v_inh", type=float, default=0.)
    # ntwk 
    parser.add_argument("--ext_drive", type=float, default=1.2)
    parser.add_argument("--Ntot", type=int, default=5000.)
    parser.add_argument("--pconnec", type=float, default=0.02)
    parser.add_argument("--gei", type=float, default=0.2)
    # exc synapses
    parser.add_argument("--Qe", type=float, default=3.)
    parser.add_argument("--Te", type=float, default=5.)
    parser.add_argument("--Ee", type=float, default=0.)
    parser.add_argument("--Qi", type=float, default=12.)
    parser.add_argument("--Ti", type=float, default=5.)
    parser.add_argument("--Ei", type=float, default=-80.)
    # miscellaneous
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-u", "--update_plot", help="update the instructions", action="store_true")
    parser.add_argument("-p", "--plot", help="plot the figures", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    args = parser.parse_args()

    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
        os.system('cp '+args.filename+ ' /tmp/'+time.strftime("%Y_%m_%d-%H:%M:%S")+'.npz')
    elif args.plot:
        import matplotlib.pylab as plt
        data = np.load(args.filename)
        exec(str(data['plot']))
        plt.show()
    else:
        run_sim(args)
