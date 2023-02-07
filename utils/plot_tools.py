import os, pathlib
import numpy as np
import matplotlib as mpl
from matplotlib.pylab import *
from matplotlib.pylab import Circle, setp

style.use(os.path.join(pathlib.Path(__file__).resolve().parents[0],
              'matplotlib_style.py'))

def figure(axes=1,
           figsize=(1.5,1.3),
           wspace=0.5, hspace=0.5,
           left=0.3, right=0.95, bottom=0.3, top=0.85,
           keep_shape=False):

    if axes==1:
        nx, ny = 1, 1
        fig, AX = subplots(axes, figsize=figsize)
        if keep_shape:
            AX = [[AX]]

    elif type(axes) in [int]:

        nx, ny = 1, axes
        fig, AX = subplots(1, axes, figsize=figsize)
        if keep_shape:
            AX = [AX]

    elif type(axes) in [tuple, list]:

        nx, ny = axes[1], axes[0]
        fig, AX = subplots(axes[1], axes[0],
                               figsize=(figsize[0]*axes[1],
                                        figsize[1]*axes[0]))

        if keep_shape and (axes[0]==1) and (axes[0]==1):
            AX = [[AX]]

        elif keep_shape and ((axes[0]==1) or (axes[1]==1)):
            AX = [AX]

    else:
        print(axes, ' --> shape not recognized ')

    fig.subplots_adjust(left=left/nx,
                        right=1-(1-right)/nx,
                        top=1-(1-top)/ny,
                        bottom=bottom/ny,
                        wspace=wspace, hspace=hspace)

    return fig, AX

def inset(stuff,
          rect=[.5,.5,.5,.4],
          facecolor='w'):
    """
    creates an inset inside "stuff" (either a figure or an axis)
    """


    if type(stuff)==mpl.figure.Figure: # if figure, no choice
        subax = stuff.add_axes(rect,
                               facecolor=facecolor)
    else:
        fig = mpl.pyplot.gcf()
        box = stuff.get_position()
        width = box.width
        height = box.height
        inax_position  = stuff.transAxes.transform([rect[0], rect[1]])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        subax = fig.add_axes([x,y,width,height],facecolor=facecolor)

    return subax

def plot(x, y, sy=None,
        ax=None,
        color='k',
        lw=1,
        alpha=0.3):

    if ax is None:
        fig, ax = figure()
    else:
        fig = None

    ax.plot(x, y, lw=lw, color=color)
    if sy is not None:
        ax.fill_between(x, 
                np.array(y)-np.array(sy),
                np.array(y)+np.array(sy),
                lw=0, color=color,
                alpha=alpha)
        

def draw_bar_scales(ax,
                    Xbar=0., Xbar_label='', Xbar_fraction=0.1, Xbar_label_format='%.1f',
                    Ybar=0., Ybar_label='', Ybar_fraction=0.1, Ybar_label_format='%.1f',
                    loc='top-left',
                    orientation=None,
                    xyLoc=None, 
                    Xbar_label2='',Ybar_label2='',
                    color='k', xcolor='k', ycolor='k', ycolor2='grey',
                    fontsize=8, size='normal',
                    shift_factor=20., lw=1,
                    remove_axis=False):
    """
    USE:

    fig, ax = figure()
    ax.plot(np.random.randn(10), np.random.randn(10), 'o')
    draw_bar_scales(ax, (0,0), 1, '1s', 2, '2s', orientation='right-bottom', Ybar_label2='12s')
    set_plot(ax)    
    """

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    if Xbar==0:
        Xbar = (xlim[1]-xlim[0])*Xbar_fraction
        Xbar_label = Xbar_label_format % Xbar
        print('X-bar label automatically set to: ', Xbar_label, ' Using the format', Xbar_label_format, ' --> adjust it and add units through the format !')
    if Ybar==0:
        Ybar = (ylim[1]-ylim[0])*Ybar_fraction
        Ybar_label = Ybar_label_format % Ybar
        print('Y-bar label automatically set to: ', Ybar_label, ' Using the format', Ybar_label_format, ' --> adjust it and add units through the format !')

    if type(loc) is tuple:
        xyLoc = xlim[0]+loc[0]*(xlim[1]-xlim[0]), ylim[0]+loc[1]*(ylim[1]-ylim[0])
        
    if (loc in ['top-right', 'right-top']) or (orientation in ['left-bottom','bottom-left']):

        if xyLoc is None:
            xyLoc = (xlim[1]-0.05*(xlim[1]-xlim[0]), ax.get_ylim()[1]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]-np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]-np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=xcolor, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=ycolor, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate('\n'+Ybar_label2, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor),
                        color=ycolor2, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
            
    elif (loc in ['top-left', 'left-top']) or (orientation in ['right-bottom','bottom-right']):
        
        if xyLoc is None:
            xyLoc = (xlim[0]+0.05*(xlim[1]-xlim[0]), ax.get_ylim()[1]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]+np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]-np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=xcolor, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=ycolor, va='top', ha='right',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate('\n'+Ybar_label2, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor),
                        color=ycolor2, va='top', ha='right',fontsize=fontsize, annotation_clip=False)

    elif (loc in ['bottom-right', 'right-bottom']) or (orientation in ['left-top','top-left']):
        
        if xyLoc is None:
            xyLoc = (xlim[1]-0.05*(xlim[1]-xlim[0]), ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]-np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]+np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=xcolor, va='top', ha='right',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=ycolor, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate(Ybar_label2+'\n',
                        (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor),
                        color=ycolor2, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)

    elif (loc in ['bottom-left', 'left-bottom']) or (orientation in ['right-top','top-right']):
        
        if xyLoc is None:
            xyLoc = (xlim[0]+0.05*(xlim[1]-xlim[0]), ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]+np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]+np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=xcolor, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=ycolor, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate(Ybar_label2+'\n', (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor),
                        color=ycolor2, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
    else:
        print("""
        orientation not recognized, it should be one of
        - right-top, top-right
        - left-top, top-left
        - right-bottom, bottom-right
        - left-bottom, bottom-left
        """)
        
    if remove_axis:
        ax.axis('off')


def get_linear_colormap(color1='blue', color2='red'):
    return mpl.colors.LinearSegmentedColormap.from_list('mycolors',[color1, color2])

if __name__=='__main__':

    fig, AX = figure((2,2))
    for Ax in AX:
        for ax in Ax:
            ax.plot(*np.random.randn(2, 10), 'o')
            ax.set_title('test')
    fig.supxlabel('x-label (unit)')
    fig.supylabel('y-label (unit)')

    fig, ax = figure()
    for i in range(5):
        ax.plot(*np.random.randn(2, 20), 'o')
    ax.set_xlabel('x-label (unit)')
    ax.set_ylabel('y-label (unit)')
    ax.set_title('title')

    show()



