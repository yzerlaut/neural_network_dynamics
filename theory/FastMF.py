import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scipy.interpolate import RegularGridInterpolator
import itertools

from cells.cell_library import built_up_neuron_params
from theory.tf import build_up_afferent_synaptic_input
from theory.mean_field import input_output

from ntwk_stim.waveform_library import *

class FastMeanField:

    def __init__(self, Model, REC_POPS, AFF_POPS,
                 tstop=None, dt=5e-3):

        self.REC_POPS = REC_POPS
        self.AFF_POPS = AFF_POPS
        self.Model = Model
        
        # initialize time axis
        self.dt = dt
        if tstop is None:
            if Model['tstop']>100:
                print('very large value of tstop, suspecting MilliSecond instead of switching. Override by expliciting the "tstop" arg.')
                self.tstop = 1e-3*Model['tstop']
            else:
                self.tstop = Model['tstop']
        else:
            self.tstop = tstop
        self.t = np.arange(int(self.tstop/self.dt))*self.dt
        
        # initialize connectivity matrix
        self.initialize_Effective_Connectivity_Matrix(Model)
        self.compute_exc_inh_matrix_factors(Model)
        
        # initialize afferent input
        self.FAFF = np.zeros((len(self.t),len(self.AFF_POPS)))
        for ipop, pop in enumerate(AFF_POPS):
            if '%s_IncreasingStep_size'%pop in Model:
                print('Adding Increasing Step Waveform to:', pop)
                self.FAFF[:,ipop] = IncreasingSteps(self.t, pop, Model, translate_to_SI=True)
            else:
                print('Setting Constant Level to:', pop)
                self.FAFF[:,ipop] = 0*self.t+Model['F_%s'%pop]

        # intrinsic currents
        self.I_INTRINSINC = np.zeros((len(self.t),len(self.REC_POPS)))
        for ipop, pop in enumerate(self.REC_POPS):
            # only support for time-phase-locked oscillatory current so far
            if '%s_Ioscill_freq'%pop in Model:
                print('Adding intrinsic oscillation to:', pop)
                self.I_INTRINSINC[:,ipop] = Intrinsic_Oscill(self.t, pop, Model, translate_to_SI=True)

        # matrix
        self.compute_exc_inh_matrix_factors(Model)

        self.TF_func = None # to be initialized !
        
    # initialize Effective Connectivity
    def initialize_Effective_Connectivity_Matrix(self, Model):
        self.ecMatrix = np.zeros((len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS)))
        for i, ii in enumerate(self.REC_POPS+self.AFF_POPS):
            for j, jj in enumerate(self.REC_POPS):
                self.ecMatrix[i,j] = Model['p_%s_%s' % (ii,jj)]*Model['N_%s' % ii]


    def compute_exc_inh_matrix_factors(self, Model):
        self.CexcF = np.zeros((len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS)))
        self.CinhF = np.zeros((len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS)))
        for i, ii in enumerate(self.REC_POPS+self.AFF_POPS):
            for j, jj in enumerate(self.REC_POPS):
                if len(ii.split('Exc'))>1:
                    self.CexcF[i,j] = 1.
                elif len(ii.split('Inh'))>1:
                    self.CinhF[i,j] = 1.
                else:
                    raise NameError('/!\ %s is not classified as Exc or Inh' % ii+jj)

                
    def compute_exc_inh_matrices(self, ecMatrix):
        # just separates excitation and inhibition
        return np.multiply(ecMatrix, self.CexcF), np.multiply(ecMatrix, self.CinhF)

    
    def build_TF_func(self, Ngrid=20,
                      coeffs_location='data/COEFFS_pyrExc.npy',
                      pop=None,
                      Exc_lim=[0.01,1000], Inh_lim=[0.01, 1000], sampling='log',
                      EXC_VALUE_THRESHOLD=10.):
        """
        """

        print('Initializing simulation [...]')
        if pop is None:
            pop = self.REC_POPS[0]
            
        # taking just one Exc and One Inh pop for the scan !!
        Exc_pop = [rec for rec in self.REC_POPS if len(rec.split('Exc'))>1][0]
        Inh_pop = [rec for rec in self.REC_POPS if len(rec.split('Inh'))>1][0]
        
        # building artificial simulation situation (with just one exc and one inh)
        AFF_POPS = [Exc_pop, Inh_pop]
        Model2 = self.Model.copy()
        Model2['N_%s'%Exc_pop], Model2['N_%s'%Inh_pop] = 10, 10
        Model2['p_%s_%s'%(Exc_pop, pop)], Model2['p_%s_%s'%(Inh_pop, pop)] = 0.1, 0.1
        
        nrn_params = built_up_neuron_params(Model2, pop)
        syn_input = build_up_afferent_synaptic_input(Model2,
                                                          AFF_POPS, pop)
        Model2['COEFFS'] = np.load(coeffs_location)
        if sampling=='log':
            Freq_Exc = np.logspace(*np.log10(Exc_lim), Ngrid+1)
            Freq_Inh = np.logspace(*np.log10(Inh_lim), Ngrid)
        else:
            Freq_Exc = np.linspace(*Exc_lim, Ngrid+1)
            Freq_Inh = np.linspace(*Inh_lim, Ngrid)

        Ioscill = np.linspace(0, 20*10, int(Ngrid/2))
        output_freq = np.zeros((len(Freq_Exc), len(Freq_Inh), len(Ioscill)))

        print('Performing grid simulation [...]')
        for i, j, k in itertools.product(range(len(Freq_Exc)), range(len(Freq_Inh)),
                                         range(len(Ioscill))):
            if Freq_Exc[i]<EXC_VALUE_THRESHOLD:
                output_freq[i,j,k] = 0
            else:
                output_freq[i,j,k] = input_output(nrn_params, syn_input,
                                                  {'F_%s'%Exc_pop:Freq_Exc[i], 'F_%s'%Inh_pop:Freq_Inh[j]},
                                                  Model2['COEFFS'],
                                                  current_input=Ioscill[k])
        print('Building interpolation [...]')
        self.TF_func = RegularGridInterpolator([Freq_Exc*\
                                                Model2['p_%s_%s'%(Exc_pop, pop)]*Model2['N_%s'%Exc_pop],
                                                Freq_Inh*\
                                                Model2['p_%s_%s'%(Inh_pop, pop)]*Model2['N_%s'%Inh_pop],
                                                Ioscill],
                                               output_freq,
                                               method='linear',
                                               fill_value=None, bounds_error=False)
        print('--> Done !')

    def rise_factor(self, X, t, Cexc, Cinh):
        return self.TF_func(np.array([np.dot(np.concatenate([X, self.FAFF[int(t/self.dt),:]]), Cexc),
                              np.dot(np.concatenate([X, self.FAFF[int(t/self.dt),:]]), Cinh),
                              self.I_INTRINSINC[int(t/self.dt),:]]).T)

    
    def dX_dt(self, X, t, Cexc, Cinh, tau=5e-3):
        return (self.rise_factor(X,t,Cexc,Cinh)-X)/tau

        
    def run_single_connectivity_sim(self, ecMatrix):
        
        X = np.zeros((len(self.t), len(self.REC_POPS)))
        
        if self.TF_func is None:
            raise NameError('/!\ Need to run the "build_TF_func" protocol before')
        else:
            Cexc, Cinh = self.compute_exc_inh_matrices(ecMatrix)
            # simple forward Euler iteration
            for it, tt in enumerate(self.t[:-1]):
                X[it+1,:] = X[it,:]+self.dt*self.dX_dt(X[it,:], tt, Cexc, Cinh)
                
        return X
