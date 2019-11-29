import sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk # my custom layer on top of Brian2


def list_compartment_types(COMP_LIST):
    type_list = []
    for cc in COMP_LIST:
        if cc.type not in type_list:
            type_list.append(cc.type)
    return type_list
    

def get_compartment_list(morpho,
                         inclusion_condition='True'):
    """
    condition should be of the form: 'comp.z>130', 'comp.type!="axon"'
    """
    
    COMP_LIST, N_seg, INCLUSION = [], [], []
    exec("COMP_LIST.append(morpho); comp = COMP_LIST[-1]; N_seg.append(len(comp.x)); INCLUSION.append("+inclusion_condition+")")

    TOPOL = str(morpho.topology())
    TT = TOPOL.split('\n')
    condition, comp = True, None
    for index, t in enumerate(TT[1:-1]):
        exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; N_seg.append(len(comp.x)); INCLUSION.append("+inclusion_condition+")")

    NSEG_TOT = np.cumsum(N_seg)
    COMPARTMENT_LIST, NSEG_INDICES = [], []
    for c, i, nseg_tot, nseg in zip(COMP_LIST, INCLUSION, NSEG_TOT, N_seg):
        if i:
            COMPARTMENT_LIST.append(c)
            NSEG_INDICES.append(nseg_tot+np.arange(nseg))
    # return COMPARTMENT_LIST, np.arange(len(COMP_LIST))[np.array(INCLUSION, dtype=bool)]
    return COMPARTMENT_LIST, NSEG_INDICES


def compute_segments(morpho,
                     soma_comp=None,
                     polar_angle=0, azimuth_angle=np.pi/2., 
                     without_axon=False):
    """
    BROKEN / USELESS function
    """

    SEGMENTS = {} # a dictionary storing segment informations
    COMP_LIST, SEG_INDICES = get_compartment_list(morpho)
    if soma_comp is None:
        somaS, _ = ntwk.morpho_analysis.get_compartment_list(morpho,
                            inclusion_condition='comp.type=="soma"')
        if len(somaS)==1:
            soma = somaS[0]
        else:
            print('/!\ several compartments for soma, took:', soma)
            soma = somaS[0]

            
    [x0, y0, z0] = soma.x, soma.y, soma.z
    SEGMENTS['x'] = np.concatenate([c.x for c in COMP_LIST])
    SEGMENTS['y'] = np.concatenate([c.y for c in COMP_LIST])
    SEGMENTS['z'] = np.concatenate([c.z for c in COMP_LIST])
    SEGMENTS['comp_type'] = np.concatenate([[c.type for i in range(len(c.x))] for c in COMP_LIST])
    SEGMENTS['area'] = np.concatenate([c.area for c in COMP_LIST])
    SEGMENTS['index'] = np.arange(len(SEGMENTS['x']))
    
    return SEGMENTS

if __name__=='__main__':

    filename = os.path.join(str(pathlib.Path(__file__).resolve().parent),
                            'morphologies',
                            'Jiang_et_al_2015',
                            'L5pyr-j140408b.CNG.swc')
    morpho = ntwk.Morphology.from_swc_file(filename)
    SEGMENTS = compute_segments(morpho)
    
    print(len(SEGMENTS['index']), len(np.unique(SEGMENTS['index'])))
    
    # COMP_LIST, INDICES = ntwk.morpho_analysis.get_compartment_list(morpho)
    # print(ntwk.morpho_analysis.list_compartment_types(COMP_LIST))
    # COMP_LIST, INDICES = ntwk.morpho_analysis.get_compartment_list(morpho,
    #                             inclusion_condition='comp.type!="axon"')
    # print(ntwk.morpho_analysis.list_compartment_types(COMP_LIST))
