import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as nrn
from brian2 import *

from datavyz import ges as ge

Equation_String = nrn.Equation_String

defaultclock.dt = 0.01*ms

filename = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'single_cell_integration', 'morphologies', 'Jiang_et_al_2015', 'L5pyr-j140408b.CNG.swc')
morpho = Morphology.from_swc_file(filename)


iHH = nrn.HodgkinHuxleyCurrent()
# iT = LowThresholdCalciumCurrent()
# iHVACa = HighVoltageActivationCurrent()
# iK = nrn.PotassiumChannelCurrent()
# iNa = nrn.SodiumChannelCurrent()
iPas = nrn.PassiveCurrent()

# Equation_String = CalciumConcentrationDecay().insert(Equation_String)

# for current in [iPas, iNa, iK]:
for current in [iPas, iHH]:
    Equation_String = current.insert(Equation_String)

eqs = Equations(Equation_String)

# # Simplified three-compartment morphology
# morpho = Cylinder(x=[0, 30]*um, diameter=20*um)
# morpho.dend = Cylinder(x=[0, 20]*um, diameter=10*um)
# morpho.dend.distal = Cylinder(x=[0, 500]*um, diameter=3*um)

neuron = nrn.SpatialNeuron(morphology=morpho,
                           model=Equation_String,
                           Cm=1 * uF / cm ** 2,
                           Ri=100 * ohm * cm,
                           method='exponential_euler')

# initialize Vm
neuron.v = -75*mV
# 
for current in [iPas, iHH]:
    current.init_sim(neuron)

# for all    
neuron.gbar_Pas = 0.2*1e-12*siemens/um**2 # pS/um2

neuron.gHH_Na = 0.01*1e-12*siemens/um**2 # pS/um2
neuron.gHH_K = 0.01*1e-12*siemens/um**2 # pS/um2

# soma
neuron.gHH_Na[0] = 1500*1e-12*siemens/um**2 # pS/um2
neuron.gHH_K0[0] = 200*1e-12*siemens/um**2 # pS/um2

# Only the soma has Na/K channels
# neuron.gHH_Na = 100*msiemens/cm**2
# neuron.gHH_K = 100*msiemens/cm**2
# neuron.dend.gHH_Na = 0*msiemens/cm**2
# neuron.dend.gHH_K = 0*msiemens/cm**2
# neuron.dend.distal.gHH_Na = 0*msiemens/cm**2
# neuron.dend.distal.gHH_K = 0*msiemens/cm**2


isoma, idend = 0, 70
mon = StateMonitor(neuron, ['v'], record=[isoma, idend])

# neuron.P_Ca = 0*cm/second
# neuron.dend.distal.P_Ca = 0*cm/second

run(100*ms)
neuron.main.I_inj = 300*pA
run(200*ms)
neuron.main.I_inj = 0*pA
run(100*ms)
# WITH T-CURRENT
# neuron.P_Ca = 1.7e-5*cm/second
# neuron.dend.distal.P_Ca = 9.5e-5*cm/second
neuron.main.I_inj = 300*pA
run(200*ms)
neuron.main.I_inj = 0*pA
run(200*ms)

# ## Run the various variants of the model to reproduce Figure 12
from datavyz import ges as ge
fig, ax = ge.figure()
ax.plot(mon.t / ms, mon[isoma].v/mV, 'k', label='soma')
ax.plot(mon.t / ms, mon[idend].v/mV, 'blue', label='dend')
ge.legend(ax)
ge.show()
