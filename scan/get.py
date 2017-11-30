import numpy as np
from itertools import product
import sys, pathlib
import zipfile
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from recording.load_and_save import load_dict_from_hdf5

def get_scan(Model,
             filename=None):

    if filename is None:
        filename=str(Model['zip_filename'])
    zf = zipfile.ZipFile(filename, mode='r')
    
    data = zf.read(filename.replace('.zip', '_Model.npz'))
    with open(filename.replace('.zip', '_Model.npz'), 'wb') as f: f.write(data)
    Model = dict(np.load(filename.replace('.zip', '_Model.npz')).items())

    DATA = []
    print(Model['PARAMS_SCAN'])
    for fn in (Model['PARAMS_SCAN'].all()['FILENAMES']):
        print(fn)
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
        with open(fn, 'rb') as f: data = load_dict_from_hdf5(fn)
        DATA.append(data)
    return Model, dict(Model['PARAMS_SCAN'].all()), DATA

if __name__=='__main__':

    Model = {'data_folder': './', 'SEED':0, 'x':2, 'zip_filename':'data.zip'}
    Model, PARAMS_SCAN, DATA = get_scan(Model)
    print(DATA)
