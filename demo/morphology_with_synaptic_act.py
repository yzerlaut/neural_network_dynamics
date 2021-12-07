import sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import nrn, ntwk

tstop = 600

Model = {
    'morpho_file':'nrn/morphologies/Jiang_et_al_2015/L23pyr-j150407a.CNG.swc',  
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 0.29, # [pS/um2] = 10*[S/m2] # FITTED --- Farinella et al. 0.5pS/um2 = 0.5*1e-12*1e12 S/m2, NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 0.91, # [uF/cm2] FITTED
    "Ri": 100., # [Ohm*cm]'
    "EL": -75, # mV
    ###################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    ###################################################
    'Ee':0,# [mV]
    'Ei':-80,# [mV]
    'qAMPA':2.5,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':2.5*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'qGABA':4.,# [nS] # Destexhe et al., 1998: "0.25 to 1.2 nS"
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseGABA':0.5,# [ms] Destexhe et al. 1998
    'tauDecayGABA':5,# [ms] Destexhe et al. 1998
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 100,# [ms] --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
}



if sys.argv[-1]=='plot':

    from analyz.IO.npz import load_dict
    from datavyz import ges as ge
    
    data = load_dict('data.npz')

    fig, AX = ge.figure(axes=(1,3), figsize=(2.,1.))
    ge.plot(data['t'], Y=[data['Vsoma'], data['Vdend']],
            LABELS=['soma', 'dend'], COLORS=['k', ge.blue], ax=AX[0],
            axes_args={'ylabel':'Vm (mV)', 'xlim':[0, data['t'][-1]]})
    ge.plot(data['t'], Y=[data['Casoma'], data['Cadend']],
            COLORS=['k', ge.blue], ax=AX[1],
            axes_args={'ylabel':'[Ca$^{2+}$] (nM)', 'xlabel':'time (ms)', 'xlim':[0, data['t'][-1]]})
    ge.scatter(data['Espike_times'], data['Espike_IDs'], color=ge.green,ax=AX[2],no_set=True,ms=3)
    ge.scatter(data['Ispike_times'], data['Espike_IDs'].max()*1.2+data['Ispike_IDs'], color=ge.red,
               ax=AX[2], no_set=True, ms=3)
    ge.set_plot(AX[2], [], xlim=[0, data['t'][-1]])
    fig.savefig('/home/yann/Desktop/trace.svg')
    ge.show()

else:
    
    nrn.defaultclock.dt = 0.025*nrn.ms

    # calcium dynamics following: HighVoltageActivationCalciumCurrent + LowThresholdCalciumCurrent
    # Equation_String = nrn.CalciumConcentrationDynamics(contributing_currents='0*mA/cm**2',
    Equation_String = nrn.CalciumConcentrationDynamics(contributing_currents='IHVACa+IT',
                                             name='CaDynamics').insert(nrn.Equation_String)
    
    # intrinsic currents
    CURRENTS = [nrn.PassiveCurrent(name='Pas'),
                nrn.PotassiumChannelCurrent(name='K'),
                nrn.SodiumChannelCurrent(name='Na'),
                nrn.HighVoltageActivationCalciumCurrent(name='HVACa'),
                nrn.LowThresholdCalciumCurrent(name='T'),
                nrn.MuscarinicPotassiumCurrent(name='Musc'),
                nrn.CalciumDependentPotassiumCurrent(name='KCa')]
    # CURRENTS = [nrn.PassiveCurrent(name='Pas'),
    #             nrn.HodgkinHuxleyCurrent(name='HH')]

    for current in CURRENTS:
        Equation_String = current.insert(Equation_String)




    ##########################################################
    # -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
    ##########################################################
    # cable theory:
    Equation_String +='''
    Is = gE * (({Ee}*mV) - v) + gI * (({Ei}*mV) - v) - Iinj : amp (point current)
    Iinj : amp 
    gE : siemens
    gI : siemens
    '''.format(**Model)

    # # synaptic dynamics:
    # -- excitation (NMDA-dependent)
    EXC_SYNAPSES_EQUATIONS ='''
    dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
    dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
    dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
    dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
    gAMPA = ({qAMPA}*nS)*(gDecayAMPA-gRiseAMPA) : siemens
    gNMDA = ({qNMDA}*nS)*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-v_post/({V0NMDA}*mV))) : siemens
    gE_post = gAMPA+gNMDA : siemens (summed)'''
    ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'

    # -- inhibition (NMDA-dependent)
    INH_SYNAPSES_EQUATIONS ='''
    dgRiseGABA/dt = -gRiseGABA/({tauRiseGABA}*ms) : 1 (clock-driven)
    dgDecayGABA/dt = -gDecayGABA/({tauDecayGABA}*ms) : 1 (clock-driven)
    gI_post = ({qGABA}*nS)*(gDecayGABA-gRiseGABA) : siemens (summed)'''
    ON_INH_EVENT = 'gRiseGABA += 1; gDecayGABA += 1'

    eqs = nrn.Equations(Equation_String)

    morpho = nrn.Morphology.from_swc_file(Model['morpho_file'])

    neuron = nrn.SpatialNeuron(morpho, model=eqs,
                               Cm=1*nrn.uF/nrn.cm**2, Ri=150*nrn.ohm*nrn.cm,
                               method='exponential_euler')

    # initial conditions:
    neuron.v = -75*nrn.mV
    neuron.InternalCalcium = 100*nrn.nM

    ## -- PASSIVE PROPS -- ##
    neuron.gbar_Pas = 1e-4*nrn.siemens/nrn.cm**2

    # ## -- SPIKE PROPS (Na & Kv) -- ##
    # # dendrites
    # neuron.gbarNa_HH = 40*1e-12*nrn.siemens/nrn.um**2
    # neuron.gbarK_HH = 30*1e-12*nrn.siemens/nrn.um**2
    # # soma
    # neuron.gbarNa_HH[0] = 1500*1e-12*nrn.siemens/nrn.um**2
    # neuron.gbarK_HH[0] = 200*1e-12*nrn.siemens/nrn.um**2
    
    # neuron.gbar_Na = 40*1e-12*nrn.siemens/nrn.um**2
    # neuron.gbar_K = 30*1e-12*nrn.siemens/nrn.um**2
    # # soma
    # neuron.gbar_Na[0] = 1500*1e-12*nrn.siemens/nrn.um**2
    # neuron.gbar_K[0] = 200*1e-12*nrn.siemens/nrn.um**2

    # ## -- HIGH-VOLTAGE-ACTIVATION CALCIUM CURRENT -- ##
    neuron.gbar_HVACa = 0.5*1e-12*nrn.siemens/nrn.um**2

    # ## -- CALCIUM-DEPENDENT POTASSIUM CURRENT -- ##
    neuron.gbar_KCa = 2.5*1e-12*nrn.siemens/nrn.um**2

    # ## -- T-CURRENT (Calcium) -- ##
    neuron.gbar_T = 0.0003*1e-12*nrn.siemens/nrn.um**2
    neuron.dend.gbar_T = 0.0006*1e-12*nrn.siemens/nrn.um**2

    # ## -- M-CURRENT (Potassium) -- ##
    neuron.gbar_Musc = 2.2*1e-12*nrn.siemens/nrn.um**2
    neuron.dend.gbar_Musc = 0.05*1e-12*nrn.siemens/nrn.um**2

    # # ## -- H-CURRENT (non-specific) -- ##
    # neuron.gbar_H = 0*1e-12*nrn.siemens/nrn.um**2 # set to zero !!

    for current in CURRENTS:
        current.init_sim(neuron)

    t = np.arange(int(tstop/nrn.defaultclock.dt*nrn.ms))*nrn.defaultclock.dt/nrn.ms

    soma_loc, dend_loc = 0, 1000
    mon = nrn.StateMonitor(neuron, ['v', 'InternalCalcium'], record=[soma_loc, dend_loc])

    SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)

    Nsyn_Exc, pre_to_iseg_Exc, Nsyn_per_seg_Exc = nrn.spread_synapses_on_morpho(SEGMENTS,
                                                                                0.5,
                                                                    density_factor=1./100./1e-12,
                                                                                verbose=True)

    Espike_IDs, Espike_times = ntwk.stim.spikes_from_time_varying_rate(t, 0*t+2.5, N=Nsyn_Exc)

    Estim, ES = nrn.process_and_connect_event_stimulation(neuron,
                                                          Espike_IDs, Espike_times,
                                                          pre_to_iseg_Exc,
                                                          EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                          ON_EXC_EVENT.format(**Model))

    Nsyn_Inh, pre_to_iseg_Inh, Nsyn_per_seg_Inh = nrn.spread_synapses_on_morpho(SEGMENTS,
                                                                                0.15,
                                                                        density_factor=1./100./1e-12,
                                                                                 verbose=True)
    Ispike_IDs, Ispike_times = ntwk.stim.spikes_from_time_varying_rate(t, 0*t+5., N=Nsyn_Inh)
    Istim, IS = nrn.process_and_connect_event_stimulation(neuron,
                                                           Ispike_IDs, Ispike_times,
                                                           pre_to_iseg_Inh,
                                                           INH_SYNAPSES_EQUATIONS.format(**Model),
                                                           ON_INH_EVENT.format(**Model))

    print('running sim [...]')
    nrn.run(280*nrn.ms)
    neuron.Iinj = 5e-3*nrn.pA
    nrn.run(150*nrn.ms)

    np.savez('data.npz', **dict(Espike_IDs=Espike_IDs, Espike_times=Espike_times,
                                Ispike_IDs=Ispike_IDs, Ispike_times=Ispike_times,
                                t = mon.t / nrn.ms,
                                Vsoma=np.array(mon[soma_loc].v/nrn.mV),
                                Vdend=np.array(mon[dend_loc].v/nrn.mV),
                                Casoma=np.array(mon[soma_loc].InternalCalcium/nrn.nM),
                                Cadend=np.array(mon[dend_loc].InternalCalcium/nrn.nM)))
