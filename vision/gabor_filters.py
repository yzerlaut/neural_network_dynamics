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


if __name__=='__main__':

    import string, itertools
    ge = graph_env('manuscript')

    fig, AX = ge.figure(axes=(3,3), figsize=(.75,.9), hspace=.9, wspace=.9,
                        bottom=.3, left=.4)

    x, y = np.meshgrid(np.linspace(0, 4), np.linspace(0, 4))

    # standard gabor
    z = gabor(x, y, x0=2 ,y0=2)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[0][0])
    ge.annotate(AX[0][0], 'see legend', (0.5, 1.05), ha='center')
    # rotated
    z = gabor(x, y, x0=2,y0=2,theta=np.pi/4.)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[0][1])
    ge.annotate(AX[0][1], r'$\theta$=$\pi$/4', (0.5, 1.05), ha='center')
    # shifted location
    z = gabor(x, y, x0=3 ,y0=3)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[0][2])
    ge.annotate(AX[0][2], r'$\vec{x_0}$=(3,3)', (0.5, 1.05), ha='center')
    # phase shifted
    z = gabor(x, y, x0=2 ,y0=2, psi=np.pi)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[1][0])
    ge.annotate(AX[1][0], r'$\Psi$=$\pi$', (0.5, 1.05), ha='center')
    # frequency
    z = gabor(x, y, x0=2 ,y0=2, freq=0.5)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[1][1])
    ge.annotate(AX[1][1], r'f=1/2', (0.5, 1.05), ha='center')
    # frequency
    z = gabor(x, y, x0=2 ,y0=2, freq=2.)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[1][2])
    ge.annotate(AX[1][2], r'f=.2', (0.5, 1.05), ha='center')
    # increased excentricity
    z = gabor(x, y, x0=2 ,y0=2, beta=2)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[2][0])
    ge.annotate(AX[2][0], r'$\beta$=2', (0.5, 1.05), ha='center')
    # shrkined
    z = gabor(x, y, x0=2 ,y0=2, size=0.25)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[2][1])
    ge.annotate(AX[2][1], r'$\sigma$=0.25', (0.5, 1.05), ha='center')
    # expanded
    z = gabor(x, y, x0=2 ,y0=2, size=1., freq=0.5)
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten()+0.5, vmin=0, vmax=1, colormap=ge.binary,
                 ax=AX[2][2])
    ge.annotate(AX[2][2], r'$\sigma$=1,f=1/2', (0.5, 1.05), ha='center')
    
    for l, ax in zip(list(string.ascii_lowercase), itertools.chain(*AX)):
        ge.top_left_letter(ax, l+'  ')
        ax.set_xticks([0,2,4])
        ax.set_yticks([0,2])
        
    fig.savefig('fig.png')

