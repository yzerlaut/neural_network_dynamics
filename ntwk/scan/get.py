import numpy as np
from itertools import product
import sys, pathlib
import zipfile
from analyz.IO.hdf5 import load_dict_from_hdf5

def get_scan(Model,
             filename=None,
             filenames_only=False,
             verbose=True):

    if filename is None:
        filename=str(Model['zip_filename'])
    zf = zipfile.ZipFile(filename, mode='r')
    
    data = zf.read(filename.replace('.zip', '_Model.npz'))
    with open(filename.replace('.zip', '_Model.npz'), 'wb') as f: f.write(data)
    Model = dict(np.load(filename.replace('.zip', '_Model.npz'), allow_pickle=True).items())

    if filenames_only:
        print('/!\ datafiles have to be unziped before /!\ ')
        return Model, dict(Model['PARAMS_SCAN'].all()), None
    else:
        DATA = []
        for fn in (Model['PARAMS_SCAN'].all()['FILENAMES']):
            if verbose:
                print('- including "%s" ' % fn)
            # data = zf.read(fn)
            # with open(fn, 'wb') as f: f.write(data)
            with open(fn, 'rb') as f:
                data = load_dict_from_hdf5(fn)
            DATA.append(data)
            
        PARAMS_SCAN = {} # array for read-out as np.array 
        for k, v in dict(Model['PARAMS_SCAN'].all()).items():
            PARAMS_SCAN[k] = np.array(v)
            
        return Model, PARAMS_SCAN, DATA

if __name__=='__main__':

    Model = {'data_folder': './', 'SEED':0, 'x':2, 'zip_filename':'data.zip'}
    Model, PARAMS_SCAN, DATA = get_scan(Model)
    print(DATA)
