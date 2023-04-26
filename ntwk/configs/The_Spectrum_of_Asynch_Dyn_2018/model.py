import numpy as np

import sys
sys.path.append('./')
import ntwk
from utils import params

paramsFile = 'ntwk/configs/The_Spectrum_of_Asynch_Dyn_2018/params.json'
Model = params.load(paramsFile)


if sys.argv[-1]=='plot':
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('data/CellRep2019_data.h5')
    print(data)
    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm(data, smooth_population_activity=10.)
    ntwk.show()

else:

    # ntwk.quick_ntwk_sim(Model)
    ntwk.quick_run.simulation(Model,
                              filename='data/CellRep2019_data.h5')
