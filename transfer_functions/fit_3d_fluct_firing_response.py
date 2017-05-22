import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.optimize import curve_fit, leastsq
import scipy.special as sp_spec
import statsmodels.api as sm

## NORMALIZING COEFFICIENTS
# needs to be global here, because used both in the function
# and its derivatives
muV0, DmuV0 = -60e-3,10e-3
sV0, DsV0 =4e-3, 6e-3
TvN0, DTvN0 = 0.5, 1.

### CORRECTIONS FOR THE EFFECTIVE THRESHOLD

default_correction = {
    'V0':True,
    'muV_lin':True,
    'sV_lin':True,
    'Tv_lin':True,
    'muG_log':False,
    'muV_square':False,
    'sV_square':False,
    'Tv_square':False,
    'muV_sV_square':False,
    'muV_Tv_square':False,
    'sV_Tv_square':False
}

def final_threshold_func(coeff, muV, sV, TvN, muGn, El,\
                         correction = default_correction):
    full_coeff = np.zeros(len(correction))
    full_coeff[0] = coeff[0] # one threshold by default
    i = 1 # now, we set the others coeff in the order
    if correction['muV_lin']: full_coeff[1] = coeff[i];i+=1
    if correction['sV_lin']: full_coeff[2] = coeff[i];i+=1
    if correction['Tv_lin']: full_coeff[3] = coeff[i];i+=1
    if correction['muG_log']: full_coeff[4] = coeff[i];i+=1
    if correction['muV_square']: full_coeff[5] = coeff[i];i+=1
    if correction['sV_square']: full_coeff[6] = coeff[i];i+=1
    if correction['Tv_square']: full_coeff[7] = coeff[i];i+=1
    if correction['muV_sV_square']: full_coeff[8] = coeff[i];i+=1
    if correction['muV_Tv_square']: full_coeff[9] = coeff[i];i+=1
    if correction['sV_Tv_square']: full_coeff[10] = coeff[i];i+=1

    if not i==len(coeff):
        print('==================================================')
        print('mismatch between coeff number and correction type')
        print('==================================================')

        
    output = full_coeff[0]+\
      full_coeff[1]*(muV-muV0)/DmuV0+\
      full_coeff[2]*(sV-sV0)/DsV0+\
      full_coeff[3]*(TvN-TvN0)/DTvN0+\
      full_coeff[4]*np.log(muGn+1e-12)+\
      full_coeff[5]*((muV-muV0)/DmuV0)**2+\
      full_coeff[6]*((sV-sV0)/DsV0)**2+\
      full_coeff[7]*((TvN-TvN0)/DTvN0)**2+\
      full_coeff[8]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
      full_coeff[9]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
      full_coeff[10]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0

    return output

### FUNCTION, INVERSE FUNCTION

def erfc_func(mu, sigma, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*\
      sp_spec.erfc((Vthre-mu)/np.sqrt(2)/sigma)

def effective_Vthre(Y, mVm, sVm, TvN, Gl, Cm):
    Vthre_eff = mVm+np.sqrt(2)*sVm*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff



#######################################################
####### PRINTING COEFFICIENTS

def print_reduce_parameters(P, with_return=True, correction=default_correction):
    # first we reduce the parameters
    P = 1e3*np.array(P)
    final_string = ''
    keys = ['0', '\mu_V', '\sigma_V', '\tau_V']
    for p, key in zip(P, keys):
        final_string += '$P_{'+key+'}$ ='+str(round(p,1))+'mV, '
    if with_return:
        return final_string
    else:
        print(final_string)
    



#######################################################
####### FITTING PROCEDURE
#######################################################

def linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=1e5, xtol=1e-18,
            correction=default_correction,\
            print_things=True):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)

    # initial guess !!! mean and linear regressions !
    i = 0
    for val in iter(correction.values()):
        if val: i+=1
    P = np.zeros(i)
    P[0] = -45e-3 # just threshold in the right range

    def Res(p):
        threshold2 = final_threshold_func(p, muV2, sV2, TvN2, muGn2, El,\
                                          correction=correction)
        to_minimize = (vthre-threshold2)**2
        return np.mean(to_minimize)/len(threshold2)

    # bnds = ((-90e-3, -10e-3), (None,None), (None,None), (None,None), (None,None),\
    #         (None,None),(-2e-3, 3e-3), (-2e-3, 3e-3))
    # plsq = minimize(Res,P, method='SLSQP', bounds=bnds, tol=xtol,\
    #         options={'maxiter':maxiter})

    plsq = minimize(Res,P, tol=xtol, options={'maxiter':maxiter})
            
    P = plsq.x
    if print_things:
        print(plsq)
    return P

def fitting_Vthre_then_Fout(Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
        maxiter=1e5, ftol=1e-15,\
        correction=default_correction, print_things=True,\
        return_chi2=False):

    P = linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=maxiter, xtol=ftol,\
            correction=correction,\
            print_things=print_things)

    def Res(p, muV, sV, TvN, muGn, Fout):
        return (Fout-erfc_func(muV, sV, TvN,\
           final_threshold_func(p, muV, sV, TvN, muGn, El,\
                                correction=correction), Gl, Cm))
                                
    if return_chi2:
        P,cov,infodict,mesg,ier = leastsq(
            Res,P, args=(muV, sV, TvN, muGn, Fout),\
            full_output=True)
        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((Fout-Fout.mean())**2).sum()
        rsquared=1-(ss_err/ss_tot)
        return P, rsquared
    else:
        P = leastsq(Res, P, args=(muV, sV, TvN, muGn, Fout))[0]
        return P
    
def make_3d_and_2d_figs(P, Fout, s_Fout, muV, sV, Tv_ratio,\
                        muGn, Gl, Cm, El, cell_id, vthre_lim=None,\
                        FONTSIZE=18):
    
    font = {'size'   : FONTSIZE}
    mpl.rc('font', **font)

    Tv_ratio = np.round(1000.*Tv_ratio)/10
    sV, muV = np.round(sV), np.round(muV)
    Tv_levels = np.unique(Tv_ratio)

    muV_levels = np.unique(muV)
    DISCRET_muV = len(muV_levels)

    # 3d view - Fout
    fig1 = plt.figure(figsize=(5,3))
    plt.subplots_adjust(left=.3, bottom=.3)
    ax = plt.subplot(111, projection='3d')
    ax.set_title(cell_id)
    ax.view_init(elev=20., azim=210.)
    plt.xlabel('\n\n $\mu_V$ (mV)')
    plt.ylabel('\n\n $\sigma_V$ (mV)')
    ax.set_zlabel('\n\n $\\nu_\mathrm{out}$ (Hz)')

    # the colorbar to index the autocorrelation
    fig3 = plt.figure(figsize=(1.7,4.))
    plt.subplots_adjust(right=.25)
    Tv_levels = np.unique(Tv_ratio)
    # levels no more than 5
    mymap = mpl.colors.LinearSegmentedColormap.from_list(\
                        'mycolors',['red','blue'])
    bounds= np.linspace(Tv_levels.min()-10, Tv_levels.max()+10, len(Tv_levels)+1)
    norm = mpl.colors.BoundaryNorm(bounds, mymap.N)
    cb = mpl.colorbar.ColorbarBase(plt.subplot(111), cmap=mymap, norm=norm,
                                    orientation='vertical')
    cb.set_ticks(np.round(Tv_levels)) 
    cb.set_label('$\\tau_V / \\tau_\mathrm{m}^0$ (%)', fontsize=16)
    
    for TvN in Tv_levels:

        i_repet_Tv = np.where(Tv_ratio==TvN)[0]
        # setting color
        if len(Tv_levels)>1:
            r = np.min([1,(TvN-bounds.min())/(bounds.max()-bounds.min())])
        else:
            r=1

        muV2, sV2 = muV[i_repet_Tv], sV[i_repet_Tv]
        Fout2, s_Fout2 = Fout[i_repet_Tv], s_Fout[i_repet_Tv]
    
        for muV3 in np.unique(muV2):
            
            i_repet_muV = np.where(muV2==muV3)[0]
            i_muV = np.where(muV3==muV_levels)[0]

            sV3 = sV2[i_repet_muV]
            Fout3, s_Fout3 = Fout2[i_repet_muV], s_Fout2[i_repet_muV]

            ax.plot(muV3*np.ones(len(sV3)), sV3, Fout3,\
                     'D', color=mymap(r,1), ms=6, lw=0)
            
            sv_th = np.linspace(0, sV3.max())
            muGn3 =np.ones(len(sv_th))
            Vthre_th = final_threshold_func(P,\
                  1e-3*muV3, 1e-3*sv_th, TvN/100., muGn3, El)
            Fout_th = erfc_func(1e-3*muV3, 1e-3*sv_th,\
                                    TvN/100., Vthre_th, Gl, Cm)
            ax.plot(muV3*np.ones(len(sv_th)), sv_th,\
                    Fout_th, color=mymap(r,1), alpha=.7, lw=3)
                
            for ii in range(len(Fout3)): # then errobar manually
                    ax.plot([muV3, muV3], [sV3[ii], sV3[ii]],\
                        [Fout3[ii]+s_Fout3[ii], Fout3[ii]-s_Fout3[ii]],\
                        marker='_', color=mymap(r,1))
                    
    ax.set_zlim([0., Fout.max()])
    ax.xaxis.set_major_locator( MaxNLocator(nbins = 4,prune='both') )
    ax.yaxis.set_major_locator( MaxNLocator(nbins = 4) )
    ax.zaxis.set_major_locator( MaxNLocator(nbins = 4,prune='lower'))
    
    fig1.tight_layout()

    ax.set_zlim([0., max([1,Fout.max()])])

    ax.xaxis.set_major_locator( MaxNLocator(nbins = 4,prune='both') )
    ax.yaxis.set_major_locator( MaxNLocator(nbins = 4) )
    ax.zaxis.set_major_locator( MaxNLocator(nbins = 4,prune='lower') )

    return fig1

if __name__=='__main__':
    # for spiking properties, what model ?? see models.py
    import argparse
    parser=argparse.ArgumentParser(description=
     """ 
     Stimulate a reconstructed cell with a shotnoise and study Vm dynamics
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("NEURON",\
        help="Choose a cell (e.g. 'cell1') or a model of neuron (e.g. 'LIF')", default='LIF')
    parser.add_argument("--NO_PLOT", help="do not plot", action="store_true")

    args = parser.parse_args()
    
    data = np.load('data/'+args.NEURON+'.npz')

    ##### FITTING OF THE PHENOMENOLOGICAL THRESHOLD #####
    # two-steps procedure, see template_and_fitting.py
    # need SI units !!!
    P = fitting_Vthre_then_Fout(data['Fout'], 1e-3*data['muV'],\
                                1e-3*data['sV'], data['TvN'],\
                                data['muGn'], data['Gl'], data['Cm'],
                                data['El'], print_things=args.NO_PLOT)

    if args.NO_PLOT:
        np.save('data/'+args.NEURON+'_coeff.npy', P)
    else:
        ##### PLOTTING #####
        # see plotting_tools.py
        # need non SI units (electrophy units) !!!
        FIG = make_3d_and_2d_figs(P,\
                data['Fout'], data['s_Fout'], data['muV'],\
                data['sV'], data['TvN'], data['muGn'],\
                data['Gl'], data['Cm'], data['El'], args.NEURON)
        plt.show()
