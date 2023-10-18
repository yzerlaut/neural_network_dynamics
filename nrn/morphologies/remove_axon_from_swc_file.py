import sys, pathlib, os, tempfile
import numpy as np

def make_new_swc(filename):

   # we cut the file by preserving the parent mapping 
    new_swc = ''       
    mapping = {'old':[], 'new':[]}

    with open(filename, 'r') as f:

        for line in f.readlines():
            index, comp, x, y, z, diam, parent = line.split(' ')
            parent = parent.replace('\n', '')

            if comp!='2':
                mapping['new'].append(len(mapping['new']))
                mapping['old'].append(int(index))

                if parent=='-1':
                    new_parent = '-1'
                else:
                    # print(mapping['old'], parent)
                    try:
                        new_parent = mapping['new'][np.flatnonzero(np.array(mapping['old'])==int(parent))[0]]
                    except BaseException:
                        print('\nPb with line:') 
                        print(line)
                        new_parent = mapping['new'][-2]
                        print('  -->  no dendritic parent found, putting it on parent:', new_parent)

                new_swc += '%s %s %s %s %s %s %s\n' % (mapping['new'][-1], comp, x, y, z, diam, new_parent)

    print('\n --> [ok] new swc file without axon succesfully written as:\n',
         filename.replace('.swc', '_no_axon.swc'))

    with open(filename.replace('.swc', '_no_axon.swc'), 'w') as f:                                
        f.write(new_swc)

if __name__=='__main__':


    if '.swc' in sys.argv[-1]:
        filename = sys.argv[-1]
    else:
       filename = os.path.join(str(pathlib.Path(__file__).resolve().parent),
                              'Jiang_et_al_2015',
                              'L5pyr-j140408b.CNG.swc')

    # now without the axon
    make_new_swc(filename)

    try:
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
        import nrn 
        import utils.plot_tools as pt
        import matplotlib.pylab as plt

        # full morphology
        morpho = nrn.Morphology.from_swc_file(filename.replace('.swc', '_no_axon.swc'))
        SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)

        vis = pt.nrnvyz(SEGMENTS)

        fig, AX = plt.subplots(1, 3, figsize=(7,2))
        # dendrites and soma
        vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'), ax=AX[0], color='tab:red')
        AX[0].annotate('soma+dendrites', (0,0), xycoords='axes fraction', color='tab:red')
        # axon only
        vis.plot_segments(cond=(SEGMENTS['comp_type']=='axon'), ax=AX[1], color='tab:blue')
        AX[1].annotate('axon only', (0,0), xycoords='axes fraction', color='tab:blue')
        # both dendrites and axon
        vis.plot_segments(cond=(SEGMENTS['comp_type']=='axon'), ax=AX[2], color='tab:blue')
        AX[2].annotate('axon', (0,0), xycoords='axes fraction', color='tab:blue')
        vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'), ax=AX[2], color='tab:red')
        AX[2].annotate('soma+dendrites', (0,0), xycoords='axes fraction', color='tab:red')
        plt.show()

    except BaseException as be:
        print('plotting tools not found')



