from brian2 import *
import itertools
import numpy as np


def build_morpho(branch_legnth=100,
                 diameter_soma=10,
                 diameter_root=2,
                 Nbranch = 10,
                 Nperbranch=10):

    
    morpho = Soma(diameter=diameter_soma*um, 
                  x=[0]*um, 
                  y=[0]*um)
    morpho.Nbranch = Nbranch

    morpho.root = Cylinder(diameter=diameter_root*um, 
                           n=Nperbranch,
                           x=np.linspace(0, branch_length, 2)*um, 
                           y=np.zeros(2)*um)


    for b in range(2, Nbranch+1):

        comps = get_comp_list_at_level(b-1, morpho)

        for c in comps:

            setattr(c, 'L', Cylinder(diameter=diameter_root*(2/3)**(b-1)*um,
                                     n=Nperbranch,
                                     x=(branch_length*b+np.linspace(0, branch_length, 2))*um,
                                     y=c.y[0]-np.ones(2)*um))
            setattr(c, 'R', Cylinder(diameter=diameter_root*(2/3)**(b-1)*um,
                                     n=Nperbranch,
                                     x=(branch_length*b+np.linspace(0, branch_length, 2))*um,
                                     y=c.y[0]+np.ones(2)*um))

    return morpho



def get_comp_list_at_level(i, morpho):

    if i==1:

        return [morpho.root]

    elif i>1 and i<=morpho.Nbranch:

        comps = []
        for X in itertools.product(*[('L','R') for k in range(i-1)]):
            comps.append(getattr(morpho.root, ''.join(X)))
        return comps

    else:

        print(' level "%i" does not correspong to the BRT model' % i)




if __name__=='__main__':

    BRT = build_morpho(Nbranch=5)
    print(BRT.topology())

