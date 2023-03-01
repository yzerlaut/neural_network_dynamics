import sys, pathlib, os

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import nrn
from utils import plot_tools as pt


if '.swc' in sys.argv[-1]:
    filename = sys.argv[-1]
else:
    filename = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'nrn', 'morphologies', 'Jiang_et_al_2015', 'L5pyr-j140408b.CNG.swc')

morpho = nrn.Morphology.from_swc_file(filename)

COMP_LIST = nrn.morpho_analysis.get_compartment_list(morpho)
SEGMENT_LIST = nrn.morpho_analysis.compute_segments(morpho)

gL = 1e-4*nrn.siemens/nrn.cm**2
EL = -70*nrn.mV
Es = 0*nrn.mV

eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gs * (Es - v) : amp (point current)
gs : siemens
'''
neuron = nrn.SpatialNeuron(morphology=morpho, model=eqs,
                           Cm=1 * nrn.uF / nrn.cm ** 2,
                           Ri=100 * nrn.ohm * nrn.cm)
neuron.v = EL
stimulation = nrn.SpikeGeneratorGroup(2, np.arange(1), 5.*np.ones(1)*nrn.ms)
taus = 5.*nrn.ms
w = 10.*nrn.nS
S = nrn.Synapses(stimulation, neuron, model='''dg/dt = -g/taus : siemens
                                                gs_post = g : siemens (summed)''',
                  on_pre='g += w')
S.connect(i=0, j=len(neuron.v)-1)

# recording and running
M = nrn.StateMonitor(neuron, ('v'), record=[0, len(neuron.v)-1])
nrn.run(60.*nrn.ms)

fig, ax = pt.plt.subplots(1, figsize=(5,1.4))
ax.plot(np.array(M.t/nrn.ms), np.array(M.v/nrn.mV).T)
ax.set_ylabel('$V_m$ (mV)')
ax.set_xlabel('time (ms)')

# fig, ax = plot_nrn_shape(ge, COMP_LIST, axon_color='None')
# cond = SEGMENT_LIST['comp_type']!='axon' # only plot dendritic integration
# tsubsampling = 5
# ani = show_animated_time_varying_trace(np.array(M.t/nrn.ms)[::tsubsampling],
                                       # np.array(M.v/nrn.mV)[:, ::tsubsampling],
                                       # SEGMENT_LIST, fig, ax, ge,
                                       # picked_locations = [100, 50, 0, 1100, 300],
                                       # segment_condition=cond)
pt.plt.show()
