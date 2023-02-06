import os, time, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import nrn

##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################

# cable theory:
Equation_String = '''
Im = + ({gL}*siemens/meter**2) * (({EL}*mV) - v) : amp/meter**2
Is = gE * (({Ee}*mV) - v) : amp (point current)
gE : siemens'''

# synaptic dynamics:

# -- excitation (NMDA-dependent)
EXC_SYNAPSES_EQUATIONS = '''dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
                            gAMPA = ({qAMPA}*nS)*{nAMPA}*(gDecayAMPA-gRiseAMPA) : siemens
                            gNMDA = ({qNMDA}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-v_post/({V0NMDA}*mV))) : siemens
                            gE_post = gAMPA+gNMDA : siemens (summed)'''
ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'


Model = {
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 0.29, # [pS/um2] = 10*[S/m2] # FITTED --- Farinella et al. 0.5pS/um2 = 0.5*1e-12*1e12 S/m2, NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 0.91, # [uF/cm2] FITTED
    "Ri": 100., # [Ohm*cm]'
    "EL": -75, # [mV]
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'qAMPA':1.,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':1.*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'qGABA':1.,# [nS] # Destexhe et al., 1998: "0.25 to 1.2 nS"
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'tstop':600, # [ms]
    'dt':0.025,# [ms]
    'seed':1, #
    #################################################
    # ---------- MG-BLOCK PARAMS  ----------------- #
    #################################################
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
}    



def double_exp_normalization(T1, T2):
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

Model['nAMPA'] = double_exp_normalization(Model['tauRiseAMPA'],Model['tauDecayAMPA'])    
Model['nNMDA'] = double_exp_normalization(Model['tauRiseNMDA'],Model['tauDecayNMDA'])

###################################################
# ---------- SIMULATION PARAMS  ----------------- #
###################################################

def initialize_sim(Model,
                   method='current-clamp',
                   Vclamp=0.,
                   active=False,
                   Equation_String=Equation_String,
                   verbose=True,
                   tstop=400.):

    Model['tstop']=tstop
    
    # simulation params
    nrn.defaultclock.dt = Model['dt']*nrn.ms
    t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    
    np.random.seed(Model['seed'])
    
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(Nbranch=5)

    neuron = nrn.SpatialNeuron(morphology=BRT,
                               model=Equation_String.format(**Model),
                               method='euler',
                               Cm=Model['cm'] * nrn.uF / nrn.cm ** 2,
                               Ri=Model['Ri'] * nrn.ohm * nrn.cm)
    
    gL = Model['gL']*nrn.siemens/nrn.meter**2
    neuron.v = Model['EL']*nrn.mV # Vm initialized to E

    return t, neuron, BRT


def run(neuron, Model, BRT):

    # recording and running
    Ms = nrn.StateMonitor(neuron, ('v'), record=[0]) # soma
    Md = nrn.StateMonitor(neuron.root.LLLL, ('v'), record=[5]) # dendrite
    # S = nrn.StateMonitor(ES, ('X', 'gAMPA', 'gNMDA', 'bZn'), record=[0])
   
    spike_IDs = np.zeros(5)
    spike_times = (200+5*np.arange(5))
    synapses_loc = [neuron.root.LLLL[5] for i in range(5)]

    Estim, ES = nrn.process_and_connect_event_stimulation(neuron,
                                                          spike_IDs, spike_times,
                                                          synapses_loc,
                                                          EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                          ON_EXC_EVENT.format(**Model))

    # # Run simulation
    nrn.run(Model['tstop']*nrn.ms)

    output = {'t':np.array(Ms.t/nrn.ms),
              'Vm_soma':np.array(Ms.v/nrn.mV)[0,:],
              'Vm_dend':np.array(Md.v/nrn.mV)[0,:],
              'Model':Model}

    return output


def plot_signals(output, ge=None):
    fig, AX = plt.subplots(2, 1, figsize=(6,3))
    cond = output['t']>150
    AX[0].plot(output['t'][cond], output['Vm_dend'][cond], '-', color='k')
    AX[0].set_ylabel('dendritic $V_m$ (mV)')
    AX[1].plot(output['t'][cond], output['Vm_soma'][cond], '-', color='k')
    AX[1].set_ylabel('somatic $V_m$ (mV)')
    return fig

if __name__=='__main__':
    
    
    t, neuron, BRT = initialize_sim(Model)
    print(BRT.topology())

    # # Run simulation
    start = time.time()

    nrn.defaultclock.dt = 0.025*nrn.ms
    
    output = run(neuron, Model, BRT)
        
    print('Runtime: %.2f s' % (time.time()-start))

    plot_signals(output)
    plt.show()

