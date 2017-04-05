import brian2
import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.IO.hdf5 import save_dict_to_hdf5, make_writable_dict

def write_as_hdf5(NTWK, filename='data.h5'):

    data = {'dt':NTWK['dt']*np.ones(1), 'tstop':NTWK['tstop']*np.ones(1)}
    
    # we write it per population
    for ii in range(len(NTWK['NEURONS'])):
        nrn = NTWK['NEURONS'][ii]
        data[str(ii)] = make_writable_dict({'name': nrn['name'] , 'N':nrn['N']})
        data[str(ii)+'_params'] = make_writable_dict(nrn['params'])
        name = NTWK['NEURONS'][ii]['name']
        
        if 'RASTER' in NTWK.keys():
            data['tRASTER_'+name] = NTWK['RASTER'][ii].t/brian2.ms
            data['iRASTER_'+name] = np.array(NTWK['RASTER'][ii].i, dtype=np.int)

        if 'POP_ACT' in NTWK.keys():
            data['POP_ACT_'+name] = NTWK['POP_ACT'][ii].rate/brian2.Hz

        if 'VMS' in NTWK.keys():
            data['VMS_'+name] = np.array([vv.V/brian2.mV for vv in NTWK['VMS'][ii]])

        if 'ISYNe' in NTWK.keys():
            data['ISYNe_'+name] = np.array([vv.Ie/brian2.pA for vv in NTWK['ISYNe'][ii]])
            
        if 'ISYNi' in NTWK.keys():
            data['ISYNi_'+name] = np.array([vv.Ii/brian2.pA for vv in NTWK['ISYNi'][ii]])
        
    save_dict_to_hdf5(data, filename)
    
