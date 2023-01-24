import sys, pathlib, os
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import nrn, ntwk

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'datavyz'))
from datavyz.nrn_morpho import nrnvyz

tstop = 60

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
    'qAMPA':12,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':12*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 100,# [ms] --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
}



nrn.defaultclock.dt = 0.025*nrn.ms

##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################

CURRENTS = [nrn.PassiveCurrent(name='Pas')]
Equation_String = nrn.Equation_String
for current in CURRENTS:
    Equation_String = current.insert(Equation_String)

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

eqs = nrn.Equations(Equation_String)

morpho = nrn.Morphology.from_swc_file(Model['morpho_file'])

neuron = nrn.SpatialNeuron(morpho, model=eqs,
                           Cm=1*nrn.uF/nrn.cm**2, Ri=150*nrn.ohm*nrn.cm,
                           method='exponential_euler')

# initial conditions:
neuron.v = -75*nrn.mV

## -- PASSIVE PROPS -- ##
neuron.gbar_Pas = 1e-4*nrn.siemens/nrn.cm**2

for current in CURRENTS:
    current.init_sim(neuron)

t = np.arange(int(tstop/nrn.defaultclock.dt*nrn.ms))*nrn.defaultclock.dt/nrn.ms

SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)

pre_to_iseg_Exc = [1800]
pre_to_iseg_Exc = [2250]
Espike_IDs, Espike_times = [0], [10]
pre_to_iseg_Exc = [400, 1000, 2250]
Espike_IDs, Espike_times = [0, 1, 2, 1], [10, 10, 10, 11]
Estim, ES = nrn.process_and_connect_event_stimulation(neuron,
                                                      Espike_IDs, Espike_times,
                                                      pre_to_iseg_Exc,
                                                      EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                      ON_EXC_EVENT.format(**Model))


print('running sim [...]')
nrn.run(tstop*nrn.ms)

nrnV = nrnvyz(SEGMENTS)

cond=SEGMENTS['comp_type']!='axon'

fig, ax = nrnV.plot_segments(cond=SEGMENTS['comp_type']!='axon', 
                             cmap=plt.cm.copper,
                             colors=neuron.v/nrn.mV)

nrnV.ge.bar_legend(ax,
              continuous=True,
              colorbar_inset=dict(rect=[0.02,0.1,0.03,0.5], facecolor=None),
              ticks=[-75, -60, -45],
              bounds=[-75, -40],
              colormap=plt.cm.copper)

plt.savefig('fig.svg')
plt.show()
