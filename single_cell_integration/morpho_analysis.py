import sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import main as ntwk # my custom layer on top of Brian2


def list_compartment_types(COMP_LIST):
    type_list = []
    for cc in COMP_LIST:
        if cc.type not in type_list:
            type_list.append(cc.type)
    return type_list
    

def get_compartment_list(morpho,
                         inclusion_condition='True',
                         without_axon=False):
    """
    condition should be of the form: 'comp.z>130'
    """
    
    COMP_LIST, INCLUSION = [], []
    exec("COMP_LIST.append(morpho); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+")")
    
    TOPOL = str(morpho.topology())
    TT = TOPOL.split('\n')
    condition, comp, ii = True, None, 0
    for index, t in enumerate(TT[1:-1]):

        # exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+"); print("+inclusion_condition+")")
        exec("COMP_LIST.append(morpho."+t.split(' .')[-1]+"); comp = COMP_LIST[-1]; INCLUSION.append("+inclusion_condition+")")

        if without_axon and (len(t.split('axon'))>1):
            INCLUSION[-1] = False
    COMPARTMENT_LIST = []
    for c,i in zip(COMP_LIST, INCLUSION):
        if i:
            COMPARTMENT_LIST.append(c)
    return COMPARTMENT_LIST, np.arange(len(COMP_LIST))[np.array(INCLUSION, dtype=bool)]

