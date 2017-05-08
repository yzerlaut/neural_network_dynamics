# =====================
# module for numerical network simulations and analysis
# relying on *brian2*
# =====================
from brian2 import *
## custom architecture
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# building the network
from neural_network_dynamics.ntwk_build.syn_and_connec_construct import *
from neural_network_dynamics.ntwk_build.syn_and_connec_library import *
# building the stimulation
from neural_network_dynamics.ntwk_stim.connect_afferent_input import *
from neural_network_dynamics.ntwk_stim.waveform_library import double_gaussian
# for saving
from neural_network_dynamics.recording.load_and_save import write_as_hdf5
# for plotting
from neural_network_dynamics.analysis.plot_sim import *
# for analysis of netwok activity
from neural_network_dynamics.analysis.macro_quantities import *
# plt.style.use('ggplot') # a nice plotting style
from graphs.my_graph import *
