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
# for this you need the data_analysis folder
# get it at: bitbucket.org/yzerlaut/data_analysis'
from neural_network_dynamics.recording.load_and_save import write_as_hdf5
from data_analysis.IO.hdf5 import load_dict_from_hdf5
# for plotting
from neural_network_dynamics.analysis.plot_sim import *
# for analysis of netwok activity
from neural_network_dynamics.analysis.population_quantities import *
from neural_network_dynamics.analysis.cellular_quantities import get_firing_rate as single_cell_firing_rate
from neural_network_dynamics.analysis.cellular_quantities import get_Vm_fluct_props
# for parameter scans
from neural_network_dynamics.scan.run import run_scan
from neural_network_dynamics.scan.get import get_scan
# theoretical module
import neural_network_dynamics.theory as theory
from theory import mean_field


#########################################################################
######## Some quick visualization functions
#########################################################################

def quick_look_at_Vm(NTWK):
    fig, AX = plt.subplots(1, len(NTWK['VMS']), figsize=(3*len(NTWK['VMS']),2.5))
    for i, pop_recording in enumerate(NTWK['VMS']):
        for j in pop_recording.record:
            AX[i].plot(pop_recording[j].t/ms, pop_recording[j].V/mV)
        set_plot(AX[i], xlabel='time (ms)', ylabel='Vm (mV)')
    show()
    



