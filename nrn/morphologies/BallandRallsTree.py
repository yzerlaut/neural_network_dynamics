from brian2 import *
import itertools
import numpy as np


def build_morpho(branch_length=100,
                 soma_radius=5,
                 root_diameter=2,
                 Nbranch = 10,
                 diameter_exponent=2./3., # Rall's branching rule by default
                 Nperbranch=10,
                 angle=np.pi/8.,
                 random_angle=np.pi/15.,
                 seed=0):

    
    np.random.seed(seed)

    morpho = Soma(diameter=2*soma_radius*um, 
                  x=[0]*um, 
                  y=[0]*um)
    morpho.Nbranch = Nbranch

    morpho.root = Cylinder(diameter=root_diameter*um, 
                           n=Nperbranch,
                           x=np.cos(np.pi/4.)*branch_length*np.arange(2)*um,
                           y=np.sin(np.pi/4.)*branch_length*np.arange(2)*um)
    new_Angles = [np.pi/4.]

    for b in range(2, Nbranch+1):

        Angles = new_Angles
        new_Angles = []

        comps = get_comp_list_at_level(b-1, morpho)

        for i, c in enumerate(comps):

            # left angle
            new_Angles.append(Angles[i]+angle+np.random.randn()*random_angle)
            setattr(c, 'L', Cylinder(diameter=root_diameter*(diameter_exponent)**(b-1)*um,
                                     n=Nperbranch,
                                     x=np.cos(new_Angles[-1])*branch_length*np.arange(2)*um,
                                     y=np.sin(new_Angles[-1])*branch_length*np.arange(2)*um))

            # right angle
            new_Angles.append(Angles[i]-angle+np.random.randn()*random_angle)
            setattr(c, 'R', Cylinder(diameter=root_diameter*(diameter_exponent)**(b-1)*um,
                                     n=Nperbranch,
                                     x=np.cos(new_Angles[-1])*branch_length*np.arange(2)*um,
                                     y=np.sin(new_Angles[-1])*branch_length*np.arange(2)*um))

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

    sys.path.append('../neural_network_dynamics/')
    import nrn
    from nrn.plot import nrnvyz
    from utils import plot_tools as pt

    BRT = build_morpho(Nbranch=5, 
                       soma_radius=5,
                       angle=.4*np.pi/4., random_angle=np.pi/20., seed=1)

    vyz = nrnvyz(nrn.morpho_analysis.compute_segments(BRT))
    vyz.plot_segments()
    pt.plt.show()

