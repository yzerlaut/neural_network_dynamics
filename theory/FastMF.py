import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import time, copy

from scipy.interpolate import RegularGridInterpolator
import itertools

from cells.cell_library import built_up_neuron_params
from theory.tf import build_up_afferent_synaptic_input
from theory.mean_field import input_output
from theory.Vm_statistics import getting_statistical_properties
from ntwk_stim.waveform_library import *

class FastMeanField:

    def __init__(self, Model,
                 tstop=None, dt=10e-3, tau=50e-3):

        self.REC_POPS = list(Model['REC_POPS'])
        self.AFF_POPS = list(Model['AFF_POPS'])
        self.Model = Model
        
        # initialize time axis
        self.dt, self.tau = dt, tau
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
        self.FAFF = np.zeros((len(self.AFF_POPS),len(self.t)))
        for ipop, pop in enumerate(self.AFF_POPS):
            if '%s_IncreasingStep_size'%pop in Model:
                print('Adding Increasing Step Waveform to:', pop)
                self.FAFF[ipop,:] = IncreasingSteps(self.t, pop, Model, translate_to_SI=True)
            else:
                print('Setting Constant Level to:', pop)
                self.FAFF[ipop,:] = 0*self.t+Model['F_%s'%pop]

        # intrinsic currents
        self.I_INTRINSINC = np.zeros((len(self.REC_POPS),len(self.t)))
        for ipop, pop in enumerate(self.REC_POPS):
            # only support for time-phase-locked oscillatory current so far
            if '%s_Ioscill_freq'%pop in Model:
                print('Adding intrinsic oscillation to:', pop)
                self.I_INTRINSINC[ipop,:] = Intrinsic_Oscill(self.t, pop, Model, translate_to_SI=True)

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


    def build_TF_inputs(self,
                        Exc_lim, Inh_lim, Iinj_lim, sampling, Ngrid):
        
        if sampling=='log':
            Freq_Exc = np.logspace(*np.log10(Exc_lim), Ngrid+1)
            Freq_Inh = np.logspace(*np.log10(Inh_lim), Ngrid)
        else:
            Freq_Exc = np.linspace(*Exc_lim, Ngrid+1)
            Freq_Inh = np.linspace(*Inh_lim, Ngrid)

        if Iinj_lim is not None:
            Iinj = np.linspace(*Iinj_lim, int(Ngrid/2)) # doesn't need as much resolution
        else:
            Iinj = np.array([0.])

        return Freq_Exc, Freq_Inh, Iinj

    
    def simulate_TF_func(self, Ngrid=20,
                         coeffs_location='data/COEFFS_pyrExc.npy',
                         tf_sim_file='tf_sim_points.npz',
                         with_Vm_functions=False,
                         pop=None,
                         Exc_lim=[0.01,1000],
                         Inh_lim=[0.01, 1000],
                         Iinj_lim=None,
                         sampling='log',
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
        Model2 = copy.deepcopy(self.Model)
        Model2['N_%s'%Exc_pop], Model2['N_%s'%Inh_pop] = 10, 10
        Model2['p_%s_%s'%(Exc_pop, pop)], Model2['p_%s_%s'%(Inh_pop, pop)] = 0.1, 0.1
        
        nrn_params = built_up_neuron_params(Model2, pop)
        syn_input = build_up_afferent_synaptic_input(Model2,
                                                     AFF_POPS, pop)
        Model2['COEFFS'] = np.load(coeffs_location)

        Freq_Exc, Freq_Inh, Iinj = self.build_TF_inputs(Exc_lim, Inh_lim, Iinj_lim, sampling, Ngrid)

        output_freq = np.zeros((len(Freq_Exc), len(Freq_Inh), len(Iinj)))
        
        if with_Vm_functions:
            mean_Vm, std_Vm, gamma_Vm = 0*output_freq, 0*output_freq, 0*output_freq

        print('Performing grid simulation [...]')
        for i, j, k in itertools.product(range(len(Freq_Exc)),
                                         range(len(Freq_Inh)),
                                         range(len(Iinj))):
            if Freq_Exc[i]<EXC_VALUE_THRESHOLD:
                output_freq[i,j,k] = 0
            else:
                output_freq[i,j,k] = input_output(nrn_params,
                                                  syn_input,
                                                  {'F_%s'%Exc_pop:Freq_Exc[i], 'F_%s'%Inh_pop:Freq_Inh[j]},
                                                  Model2['COEFFS'],
                                                  current_input=Iinj[k])
            if with_Vm_functions:
                mean_Vm[i,j,k], std_Vm[i,j,k], _, _ =  getting_statistical_properties(nrn_params, syn_input,
                                                                                      {'F_%s'%Exc_pop:Freq_Exc[i], 'F_%s'%Inh_pop:Freq_Inh[j]},
                                                                                      current_input=Iinj[k])

        np.savez(tf_sim_file, **{'output_freq':output_freq,
                                 'mean_Vm':mean_Vm, 'std_Vm':std_Vm,
                                 'sampling':sampling,
                                 'Ngrid':Ngrid,
                                 'Exc_lim':Exc_lim,
                                 'Inh_lim':Inh_lim,
                                 'Iinj_lim':Iinj_lim})
        
    def build_TF_func(self,
                      tf_sim_file='tf_sim_points.npz'):
        
        print('Building interpolation [...]')
        sim = np.load(tf_sim_file)

        for key, val in sim.items():
            setattr(self, key, val)
            
        self.Freq_Exc, self.Freq_Inh, self.Iinj = self.build_TF_inputs(self.Exc_lim,
                                                                       self.Inh_lim,
                                                                       self.Iinj_lim,
                                                                       self.sampling, self.Ngrid)
        
    def full_MF_func(self, X, t, Cexc, Cinh):
            
            I = np.digitize(np.clip(np.dot(np.concatenate([X, self.FAFF[:,int(t/self.dt)]]), Cexc),
                                    *self.Exc_lim),
                            self.Freq_Exc, right=True)
            J = np.digitize(np.clip(np.dot(np.concatenate([X, self.FAFF[:,int(t/self.dt)]]), Cinh),
                                    *self.Inh_lim),
                            self.Freq_Inh, right=True)
            K = np.digitize(np.clip(self.I_INTRINSINC[:,int(t/self.dt)],
                                    *self.Iinj_lim),
                                    self.Iinj, right=True)
            
            return self.output_freq[I,J,K], self.mean_Vm[I,J,K]
        
        
    def run_single_connectivity_sim(self, ecMatrix, verbose=False):
        
        X = np.zeros((len(self.REC_POPS),len(self.t)))
        Vm = np.zeros((len(self.REC_POPS),len(self.t)))

        if verbose:
            start_time=1e3*time.time()
            print('running ODE integration [...]')
        if self.full_MF_func is None:
            raise NameError('/!\ Need to run the "build_TF_func" protocol before')
        else:
            Cexc, Cinh = self.compute_exc_inh_matrices(ecMatrix)
            _, Vm[:,0] = self.full_MF_func(X[:,0], 0, Cexc, Cinh)
            # simple forward Euler iteration
            for it, tt in enumerate(self.t[:-1]):
                RF, Vm[:,it+1] = self.full_MF_func(X[:,it], tt, Cexc, Cinh)
                X[:,it+1] = X[:,it]+self.dt*(RF-X[:,it])/self.tau
        if verbose:
            print('--- ODE integration took %.1f milliseconds ' % (1e3*time.time()-start_time))
                
        return X, Vm

    
if __name__=='__main__':

    import sys
    sys.path.append('../configs/Network_Modulation_2020')
    from model import Model
    
    
    mf = FastMeanField(Model, tstop=6., dt=1e-2, tau=5e-2)

    if sys.argv[-1]=='sim':
        print('building the TF sim. (based on the COEFFS)')
        mf.simulate_TF_func(200,
                            coeffs_location='../configs/Network_Modulation_2020/COEFFS_pyrExc.npy',
                            Iinj_lim=[0, 200.], # in pA
                            Exc_lim=[0.01,1000],
                            Inh_lim=[0.01, 1000],
                            with_Vm_functions=True,
                            sampling='log')

        
    mf.build_TF_func(tf_sim_file='tf_sim_points.npz')
    X, Vm = mf.run_single_connectivity_sim(mf.ecMatrix, verbose=True)

    from datavyz import ges as ge
    fig, AX = ge.figure(figsize=(3,1), axes=(1,5))
    COLORS=[ge.g, ge.b, ge.r, ge.purple]
    for i, label in enumerate(Model['REC_POPS']):
        AX[-1].plot(1e3*mf.t, 1e-2+X[i,:], lw=4, color=COLORS[i], alpha=.5)
        AX[i].plot(1e3*mf.t, 1e3*Vm[i,:], 'k-')
        AX[i].set_ylim([-72,-45])
        
    
    ge.show()
    # # benchmark
