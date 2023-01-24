import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import ntwk, nrn

import numpy as np

from datavyz import graph_env
ge = graph_env('screen')
from datavyz.nrn_morpho import *

filename = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'nrn', 'morphologies', 'Jiang_et_al_2015', 'L5pyr-j140408b.CNG.swc')
morpho = ntwk.Morphology.from_swc_file(filename)

COMP_LIST = nrn.morpho_analysis.get_compartment_list(morpho)
SEGMENT_LIST = nrn.morpho_analysis.compute_segments(morpho)

gL = 1e-4*ntwk.siemens/ntwk.cm**2
EL = -70*ntwk.mV
Es = 0*ntwk.mV

eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gs * (Es - v) : amp (point current)
gs : siemens
'''
neuron = ntwk.SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * ntwk.uF / ntwk.cm ** 2, Ri=100 * ntwk.ohm * ntwk.cm)
neuron.v = EL
stimulation = ntwk.SpikeGeneratorGroup(2, np.arange(1), 5.*np.ones(1)*ntwk.ms)
taus = 5.*ntwk.ms
w = 10.*ntwk.nS
S = ntwk.Synapses(stimulation, neuron, model='''dg/dt = -g/taus : siemens
                                                gs_post = g : siemens (summed)''',
                  on_pre='g += w')
S.connect(i=0, j=100)

# recording and running
M = ntwk.StateMonitor(neuron, ('v'), record=np.arange(len(neuron.v)))
ntwk.run(60.*ntwk.ms)

fig, ax = plot_nrn_shape(ge, COMP_LIST, axon_color='None')
cond = SEGMENT_LIST['comp_type']!='axon' # only plot dendritic integration
tsubsampling = 5
ani = show_animated_time_varying_trace(np.array(M.t/ntwk.ms)[::tsubsampling],
                                       np.array(M.v/ntwk.mV)[:, ::tsubsampling],
                                       SEGMENT_LIST, fig, ax, ge,
                                       picked_locations = [100, 50, 0, 1100, 300],
                                       segment_condition=cond)
ge.show()
