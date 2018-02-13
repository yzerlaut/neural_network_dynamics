import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from recording.hdf5 import save_dict_to_hdf5, make_writable_dict
import brian2, h5py, os

def write_as_hdf5(NTWK, filename='data.h5',
                  KEY_NOT_TO_RECORD=[]):

    data = {'dt':NTWK['dt']*np.ones(1), 'tstop':NTWK['tstop']*np.ones(1)}

    for key, val in NTWK['Model'].items():
        if (type(val)==int) or (type(val)==float):
            data[key] = np.ones(1)*val
        if (type(val)==np.ndarray):
            data[key] = val
            
    # we write it per population
    for ii in range(len(NTWK['NEURONS'])):
        nrn = NTWK['NEURONS'][ii]
        data[str(ii)] = make_writable_dict({'name': nrn['name'] , 'N':nrn['N']})
        data[str(ii)+'_params'] = make_writable_dict(nrn['params'])
        name = NTWK['NEURONS'][ii]['name']
        
        if ('RASTER' in NTWK) and ('RASTER' not in KEY_NOT_TO_RECORD):
            data['tRASTER_'+name] = NTWK['RASTER'][ii].t/brian2.ms
            data['iRASTER_'+name] = np.array(NTWK['RASTER'][ii].i, dtype=np.int)

        if ('POP_ACT' in NTWK) and ('POP_ACT' not in KEY_NOT_TO_RECORD):
            data['POP_ACT_'+name] = NTWK['POP_ACT'][ii].rate/brian2.Hz

        if ('VMS' in NTWK) and ('VMS' not in KEY_NOT_TO_RECORD):
            data['VMS_'+name] = np.array([vv.V/brian2.mV for vv in NTWK['VMS'][ii]])

        if ('ISYNe' in NTWK) and ('ISYNe' not in KEY_NOT_TO_RECORD):
            data['ISYNe_'+name] = np.array([vv.Ie/brian2.pA for vv in NTWK['ISYNe'][ii]])
            
        if ('ISYNi' in NTWK) and ('ISYNi' not in KEY_NOT_TO_RECORD):
            data['ISYNi_'+name] = np.array([vv.Ii/brian2.pA for vv in NTWK['ISYNi'][ii]])

        if ('GSYNe' in NTWK) and ('GSYNe' not in KEY_NOT_TO_RECORD):
            data['GSYNe_'+name] = np.array([vv.Ge/brian2.nS for vv in NTWK['GSYNe'][ii]])
            
        if ('GSYNi' in NTWK) and ('GSYNi' not in KEY_NOT_TO_RECORD):
            data['GSYNi_'+name] = np.array([vv.Gi/brian2.nS for vv in NTWK['GSYNi'][ii]])

    if ('iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK) and ('iRASTER_PRE_in_terms_of_Pre_Pop' not in KEY_NOT_TO_RECORD):
        data['iRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.int)
        data['tRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.float)

    if ('iRASTER_PRE' in NTWK) and ('iRASTER_PRE' not in KEY_NOT_TO_RECORD):
        for jj in range(len(NTWK['iRASTER_PRE'])):
            data['iRASTER_PRE'+str(jj)] = np.array(NTWK['iRASTER_PRE'][jj], dtype=np.int)
            data['tRASTER_PRE'+str(jj)] = np.array(NTWK['tRASTER_PRE'][jj]/brian2.ms, dtype=np.float)
            
    save_dict_to_hdf5(data, filename)
    
"""
taken from:
http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py?newreg=f582be64155a4c0f989a2aa05ee67efe
"""

def make_writable_dict(dic):
    dic2 = dic.copy()
    for key, value in dic.items():
        if (type(value)==float) or (type(value)==int):
            dic2[key] = np.ones(1)*value
        if type(value)==list:
            dic2[key] = np.array(value)
    return dic2

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, tuple):
            h5file[path + key] = np.array(item)
        elif isinstance(item, list):
            h5file[path + key] = np.array(item)
        elif isinstance(item, float):
            h5file[path + key] = np.array(item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
