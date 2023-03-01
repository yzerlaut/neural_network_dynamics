import sys, pathlib, os
import numpy as np

def get_compartment_list(morpho,
                         inclusion_condition='True'):
    """
    condition should be of the form: 'comp.z>130', 'comp.type!="axon"'
    """
    
    COMP_LIST, COMP_LIST_WITH_ARB, N_seg, INCLUSION = [], [], [], []
    exec("COMP_LIST.append(morpho); COMP_LIST_WITH_ARB.append('soma'); comp = COMP_LIST[-1]; N_seg.append(len(comp.x)); INCLUSION.append("+inclusion_condition+")")

    TOPOL = str(morpho.topology())
    TT = TOPOL.split('\n')
    condition, comp = True, None
    for index, t in enumerate(TT[1:-1]):
        exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; N_seg.append(len(comp.x)); INCLUSION.append("+inclusion_condition+")")
        COMP_LIST_WITH_ARB.append(t.split('|  .')[1])
        
    NSEG_TOT = np.cumsum(N_seg)
    COMPARTMENT_LIST, COMPARTMENT_NAMES, NSEG_INDICES = [], [], []
    for c, cwa, i, nseg_tot, nseg in zip(COMP_LIST, COMP_LIST_WITH_ARB, INCLUSION, NSEG_TOT, N_seg):
        if i:
            COMPARTMENT_LIST.append(c)
            COMPARTMENT_NAMES.append(cwa)
            NSEG_INDICES.append(nseg_tot+np.arange(nseg))
    # return COMPARTMENT_LIST, np.arange(len(COMP_LIST))[np.array(INCLUSION, dtype=bool)]
    return COMPARTMENT_LIST, COMPARTMENT_NAMES, NSEG_INDICES



def compute_segments(morpho,
                     soma_comp=None,
                     without_axon=False):
    """

    """

    SEGMENTS = {} # a dictionary storing segment informations
    COMP_LIST, COMP_NAMES, SEG_INDICES = get_compartment_list(morpho)

    SEGMENTS['name'] = np.concatenate([[n for i in range(len(c.x))] for c, n in zip(COMP_LIST, COMP_NAMES)])
    SEGMENTS['x'] = np.concatenate([c.x for c in COMP_LIST])
    SEGMENTS['y'] = np.concatenate([c.y for c in COMP_LIST])
    SEGMENTS['z'] = np.concatenate([c.z for c in COMP_LIST])
    SEGMENTS['start_x'] = np.concatenate([c.start_x for c in COMP_LIST])
    SEGMENTS['start_y'] = np.concatenate([c.start_y for c in COMP_LIST])
    SEGMENTS['start_z'] = np.concatenate([c.start_z for c in COMP_LIST])
    SEGMENTS['end_x'] = np.concatenate([c.end_x for c in COMP_LIST])
    SEGMENTS['end_y'] = np.concatenate([c.end_y for c in COMP_LIST])
    SEGMENTS['end_z'] = np.concatenate([c.end_z for c in COMP_LIST])
    SEGMENTS['diameter'] = np.concatenate([c.diameter for c in COMP_LIST])
    SEGMENTS['length'] = np.concatenate([c.length for c in COMP_LIST])
    SEGMENTS['comp_type'] = np.concatenate([\
                        [c.type for i in range(len(c.x))] for c in COMP_LIST])
    SEGMENTS['area'] = np.concatenate([c.area for c in COMP_LIST])
    SEGMENTS['index'] = np.concatenate([c.indices[:] for c in COMP_LIST])
    SEGMENTS['distance_to_soma'] = np.concatenate([c.distance for c in COMP_LIST])
    
    return SEGMENTS

def distance(x, y, z):
    return np.sqrt(x**2+y**2+z**2)

def find_conditions(SEGMENTS,
                    comp_type = None,
                    isoma = 0,
                    min_distance_to_soma=0.,
                    max_distance_to_soma=1e9):


    if type(comp_type) is list:
        condition = np.zeros(len(SEGMENTS['x']), dtype=bool)
        for c in comp_type:
            condition = condition | (SEGMENTS['comp_type']==c)
    elif comp_type is not None:
        condition = SEGMENTS['comp_type']==comp_type
    else:
        condition = np.ones(len(SEGMENTS['x']), dtype=bool)

    # distances:
    dx = SEGMENTS['x']-SEGMENTS['x'][isoma]
    dy = SEGMENTS['y']-SEGMENTS['y'][isoma]
    dz = SEGMENTS['z']-SEGMENTS['z'][isoma]
    
    condition = condition & (distance(dx, dy, dz)>=min_distance_to_soma)
    condition = condition & (distance(dx, dy, dz)<=max_distance_to_soma)
    
    return condition


if __name__=='__main__':

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    import nrn 

    filename = os.path.join(str(pathlib.Path(__file__).resolve().parent),
                            'morphologies',
                            'Jiang_et_al_2015',
                            'L5pyr-j140408b.CNG.swc')

    filename = sys.argv[-1]

    morpho = nrn.Morphology.from_swc_file(filename)
    
    SEGMENTS = compute_segments(morpho)

    print(SEGMENTS['name'])
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

