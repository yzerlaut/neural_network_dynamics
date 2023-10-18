import sys, pathlib, os
import numpy as np

def make_new_swc(filename):

   # we cut the file  
    new_swc = ''       
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.split(' ')[1]!='3':
                new_swc += line
    with open("morphologies/%s/%s_no_axon.swc" % (ID,ID), 'w') as f:                                
        f.write(new_swc)
    


if __name__=='__main__':

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import nrn 

    if '.swc' in sys.argv[-1]:
        filename = sys.argv[-1]
    else:
       filename = os.path.join(str(pathlib.Path(__file__).resolve().parent),
                              'Jiang_et_al_2015',
                              'L5pyr-j140408b.CNG.swc')

    # full morphology
    morpho = nrn.Morphology.from_swc_file(filename)
    SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)

    for name, index, dist in zip(SEGMENTS['name'], SEGMENTS['index'],
                    SEGMENTS['distance_to_soma']):
        print(name, index, 1e6*dist)

    # print(np.unique(SEGMENTS['comp_type'][SEGMENTS['comp_type']=='dend']))

    # print(find_indices_with_conditions(SEGMENTS,\
    #                                    min_distance_to_soma=230e-6,
    #                                    comp_type='apic'))
    # print(SEGMENTS['start_x'][:3], SEGMENTS['end_x'][:3], SEGMENTS['y'][:3], SEGMENTS['z'][:3])
    # # print(SEGMENTS['x'][-3:], SEGMENTS['y'][-3:], SEGMENTS['z'][-3:])
    # # print(len(SEGMENTS['index']), len(np.unique(SEGMENTS['index'])))
    
    # COMP_LIST, INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
    #                             inclusion_condition='comp.type!="axon"')
    # print(dir(COMP_LIST[0]))
    # print(ntwk.morpho_analysis.list_compartment_types(COMP_LIST))
    # find_conditions(SEGMENTS,
    #                 comp_type = ['soma', 'dend'],
    #                 isoma = 0,
    #                 min_distance_to_soma=0.,
    #                 max_distance_to_soma=1e9)

