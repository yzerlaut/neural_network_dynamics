from brian2 import *
import itertools
import numpy as np


def build_morpho(branch_length=100,
                 soma_radius=5,
                 root_diameter=2,
                 Nbranch = 10,
                 Nperbranch=10):

    
    morpho = Soma(diameter=2*soma_radius*um, 
                  x=[0]*um, 
                  y=[0]*um)
    morpho.Nbranch = Nbranch

    morpho.root = Cylinder(diameter=root_diameter*um, 
                           n=Nperbranch,
                           x=np.linspace(0, branch_length, 2)*um, 
                           y=np.zeros(2)*um)


    for b in range(2, Nbranch+1):

        comps = get_comp_list_at_level(b-1, morpho)

        for c in comps:

            setattr(c, 'L', Cylinder(diameter=root_diameter*(2/3)**(b-1)*um,
                                     n=Nperbranch,
                                     x=(branch_length*b+np.linspace(0, branch_length, 2))*um,
                                     y=c.y[0]-np.ones(2)*um))
            setattr(c, 'R', Cylinder(diameter=root_diameter*(2/3)**(b-1)*um,
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

