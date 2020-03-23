"""

A gabor filter is paramterized by:
- $\theta$ the gabor filter orientation , let's write \vec{u} the unitary vector 
- $f$ the gabor filter frequency
- $\beta$ the gabor filter excentricity
- $\Phi$ the gabor filter size


And it is defined by:

$$
G(\vec{x}) = \cos\Big( 2 \pi f (\vec{x}-\vec{x_0}) \cdot \vec{u} + \Phi \Big) \,
         e^{\frac{\Big((\vec{x}-\vec{x_0}) \cdot \vec{u}\Big)^2 +
                   \beta \Big((\vec{x}-\vec{x_0}) \cdot \vec{v}\Big)^2}{2 \, \alpha^2}}
$$

"""
import numpy as np
from datavyz.main import graph_env


def gabor(x, y,
          x0=0, y0=0,
          freq = 1.,
          size=0.5,
          beta = 1,
          theta=0.,
          psi=0):
    """
    Gabor filter
    """
    
    # Centering and Rotation
    x_theta = (x-x0) * np.cos(theta) + (y-y0) * np.sin(theta)
    y_theta = -(x-x0) * np.sin(theta) + (y-y0) * np.cos(theta)
    
    # Computing cosinus and gaussian product
    gb = np.exp(-.5 * (x_theta**2 + beta * y_theta**2) / size**2) *\
        np.cos(2 * np.pi * freq * x_theta + psi)
    
    return gb



def gabor_plot(graph, params, ax,
               title='', acb=None, bar_legend=None,
               Nx=4, Ny=3, factor=2.5):

    x, y = np.meshgrid(np.linspace(-1, 1)*factor*0.5,
                       np.linspace(-1, 1)*factor*0.5, indexing='ij')

    gb = gabor(x, y,
               x0=params['x0'],
               y0=params['y0'],
               freq=params['freq'],
               size=params['size'],
               beta=params['beta'],
               theta=params['theta'],
               psi=params['psi'])
    
    graph.twoD_plot(x.flatten(), y.flatten(), gb.flatten(),
                    vmin=-1, vmax=1, colormap=graph.binary_r,
                    ax=ax, acb=acb, bar_legend=bar_legend)
    graph.annotate(ax, title, (0.01,1), va='top', color='w')
    
if __name__=='__main__':

    import matplotlib.pylab as plt

    params = {'size':0.5,
              'x0':0, 'y0':0,
              'freq':1., 'theta':0,
              'beta':1.,'psi':0.}
    
    import string, itertools
    ge = graph_env('manuscript')

    fig, AX = ge.figure(axes=(6, 4), figsize=(.72,.86),
                        hspace=.12, wspace=.25,
                        bottom=.7, left=.8, top=.3)

    acb = plt.axes([.35, .86, .02, .12])

    for ax in AX[0][1:]:
        ax.axis('off')

    # starting config plot
    gabor_plot(ge, params, AX[0][0],
               Nx=len(AX[0]), Ny=len(AX), acb=acb,
               bar_legend={'ticks':[-1,0,1], 'labelpad':4, 'label':'filter value'})
    ge.annotate(AX[0][2], r"""$\vec{{ x_0 }}$=({x0}, {y0}) deg.
$f$= {freq} cycle/deg., 
$\theta$= {theta} Rd,    $\Psi$= {psi} Rd
$\sigma$= {size} deg.   $\beta$= {beta}""".format(**params), (0.1, .95), va='top')

    new_parameters = params.copy()

    # size and position
    new_parameters['size'] = 0.3
    gabor_plot(ge, new_parameters, AX[2][0], title='$\\sigma$=0.3')
    new_parameters['size'], new_parameters['x0'], new_parameters['y0'] = 0.3, .7, .7
    gabor_plot(ge, new_parameters, AX[2][1],
               title=r"""
+
$\vec{{x_0}}$=
({x0}, {y0})$^o$
""".format(**new_parameters))
    new_parameters['size'], new_parameters['theta'] = 0.3, np.pi/3.
    gabor_plot(ge, new_parameters, AX[2][2],
               title=r"""

+
$\theta$=$\pi$/3
""".format(**new_parameters))
    new_parameters['freq'] = 2
    gabor_plot(ge, new_parameters, AX[2][3],
               title=r"""

+
$f$={freq}
""".format(**new_parameters))
    new_parameters = params.copy()
    
    # freq
    new_parameters['size'] = 0.2
    gabor_plot(ge, new_parameters, AX[3][0], title='$\\sigma$=0.2')
    new_parameters['size'] = 0.7
    gabor_plot(ge, new_parameters, AX[3][1], title='$\\sigma$=0.7')
    new_parameters['size'], new_parameters['freq'] = 0.3, 2
    gabor_plot(ge, new_parameters, AX[3][2], title='$\\sigma$=0.3,$f$=2')
    new_parameters['size'], new_parameters['freq'] = 0.6, 2
    gabor_plot(ge, new_parameters, AX[3][3], title='$\\sigma$=0.6,$f$=2')
    new_parameters = params.copy()
    
    # psi
    new_parameters['psi'] = np.pi
    gabor_plot(ge, new_parameters, AX[5][0], title='$\\psi$=$\pi$')
    new_parameters['psi'] = np.pi/4.
    gabor_plot(ge, new_parameters, AX[5][1], title='$\\psi$=$\pi$/4')
    new_parameters['psi'] = np.pi/2.
    gabor_plot(ge, new_parameters, AX[5][2], title='$\\psi$=$\pi$/2')
    new_parameters['psi'] = 7*np.pi/4.
    gabor_plot(ge, new_parameters, AX[5][3], title='$\\psi$=7$\pi$/4')
    new_parameters = params.copy()

    # beta
    new_parameters['beta'] = 0.5
    gabor_plot(ge, new_parameters, AX[4][0], title='$\\beta$=0.5')
    new_parameters['beta'] = 1
    gabor_plot(ge, new_parameters, AX[4][1], title='$\\beta$=1')
    new_parameters['beta'] = 2
    gabor_plot(ge, new_parameters, AX[4][2], title='$\\beta$=2')
    new_parameters['beta'] = 4
    gabor_plot(ge, new_parameters, AX[4][3], title='$\\beta$=4')
    new_parameters = params.copy()
    
    
    # theta
    new_parameters['theta'] = np.pi/8.
    gabor_plot(ge, new_parameters, AX[1][0], title='$\\theta$=$\pi$/8')
    new_parameters['theta'] = np.pi/4.
    gabor_plot(ge, new_parameters, AX[1][1], title='$\\theta$=$\pi$/4')
    new_parameters['theta'] = np.pi/2
    gabor_plot(ge, new_parameters, AX[1][2], title='$\\theta$=$\pi$/2')
    new_parameters['theta'] = 5*np.pi/6.
    gabor_plot(ge, new_parameters, AX[1][3], title='$\\theta$=5$\pi$/6')
    # gabor_plot(ge, params, AX[3][0], title='$\\theta$=0')
    
    Nx, Ny= len(AX[0]), len(AX)
    for i, ax in enumerate(np.array(AX).flatten()):
        if (i%Nx==0) and (int(i/Nx)==(Ny-1)):
            ge.set_plot(ax, xticks=[-2*params['size'],0,2*params['size']], yticks=[-2*params['size'],0,2*params['size']], xlabel='x (deg.)', ylabel='  y (deg.)', ylabelpad=-1)
        elif (i%Nx==0):
            ge.set_plot(ax, ['left'], xticks=[-2*params['size'],0,2*params['size']], yticks=[-2*params['size'],0,2*params['size']], ylabel='  y (deg.)', ylabelpad=-1)
        elif (int(i/Nx)==(Ny-1)):
            ge.set_plot(ax, ['bottom'], xticks=[-2*params['size'],0,2*params['size']], yticks=[-2*params['size'],0,2*params['size']], xlabel='x (deg.)')
        else:
            ge.set_plot(ax, [], xticks=[-2*params['size'],0,2*params['size']], xticks_labels=[], yticks=[-2*params['size'],0,2*params['size']], yticks_labels=[])


    fig.savefig('docs/figs/gabor.png')
    ge.show()
