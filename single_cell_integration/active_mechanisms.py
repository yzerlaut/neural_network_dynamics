from brian2 import *
from brian2.units.constants import (zero_celsius, faraday_constant as F,
                                    gas_constant as R)

T = 34*kelvin + zero_celsius # needs to be a global variable for now, 34 degC
gamma = F/(R*T)  # R=gas constant, F=Faraday constant

Equation_String= '''
Im = + 0*amp/meter**2 : amp/meter**2
I_inj : amp (point current)'''

#########################################
########### Membrane currents ###########
#########################################

class MembraneCurrent:
    """
    parent class for all membrane currents
    """
    def __init__(self, name, params):
        
        self.name=name

        if params is None:
            self.params = self.default_params()
        else:
            self.params = params
            
        self.params['name'] = self.name
        self.compute_code(self.params) 
        
        
    def compute_code(self, params):
        # we format the string equation with the parameters
        self.code = self.equations.format(**params)

        
    def insert(self, eqs):
        # returns the updated equation string
        return eqs.replace('Im = ', 'Im = - I%s ' % self.name)+self.code


    def init_sim(self, neuron):
        # if model has HH-like variable ("m","n","h","l") -> we initialize to inf values if possible, else 0
        for l in ['m', 'n', 'h', 'l']:
            if hasattr(neuron, l+self.name):
                try:
                    setattr(neuron, l+self.name, l+self.name+'_inf')
                except KeyError:
                    setattr(neuron, l+self.name, 0)
                print(l+self.name, 'was set to :', getattr(neuron, l+self.name))


class PassiveCurrent(MembraneCurrent):
    """
    basic leak current

    parameters El, gl
    """
    def __init__(self, name='Pas',
                 params=None):
        
        self.equations = """
        I{name} = gbar_{name}*1e-12*siemens/um**2*(v-{El}*mV): amp/meter**2"""
        
        super().__init__(name, params)
    
    def default_params(self):
        return dict(El=-75) # mV

    
    
class HodgkinHuxleyCurrent(MembraneCurrent):
    """
    HH-type currents for spike initiation

    parameters:
    E_Na
    E_K
    VT shift of membrane potential (Traub convention)
    tadj_HH  # temperature adjustment for Na & K (original recordings at 36 degC)
    """
    
    def __init__(self, name='HH',
                 params=None):

        self.equations = """
        I{name} = I{name}_Na + I{name}_K : amp/meter**2
        I{name}_Na = g{name}_Na * m{name}**3 * h{name} * (v-{E_Na}*mV) : amp/meter**2
        I{name}_K = g{name}_K * n{name}**4 * (v-{E_K}*mV) : amp/meter**2
        g{name}_Na : siemens/meter**2
        g{name}_K : siemens/meter**2
        v2 = v - {VT}*mV : volt  # shifted membrane potential (Traub convention)
        dm{name}/dt = (0.32*(mV**-1)*(13.*mV-v2)/
          (exp((13.*mV-v2)/(4.*mV))-1.)*(1-m{name})-0.28*(mV**-1)*(v2-40.*mV)/
          (exp((v2-40.*mV)/(5.*mV))-1.)*m{name}) / ms * {tadj_HH}: 1
        dn{name}/dt = (0.032*(mV**-1)*(15.*mV-v2)/
          (exp((15.*mV-v2)/(5.*mV))-1.)*(1.-n{name})-.5*exp((10.*mV-v2)/(40.*mV))*n{name}) / ms * {tadj_HH}: 1
        dh{name}/dt = (0.128*exp((17.*mV-v2)/(18.*mV))*(1.-h{name})-4./(1+exp((40.*mV-v2)/(5.*mV)))*h{name}) / ms * {tadj_HH}: 1"""
        
        super().__init__(name, params)
        
    def default_params(self):
        return dict(VT=-52,#mV
                    E_Na=50,#mV
                    E_K=-100,#mV
                    tadj_HH=3.0**((34-36)/10.0))


        
class LowThresholdCalciumCurrent(MembraneCurrent):
    """
    Low-threshold Calcium current (I_T)  -- nonlinear function of voltage
    """
    def __init__(self, name='LTCa',
                 params=None):
        
        self.equations = """
        I{name} = P_Ca * m{name}**2*h{name} * G_Ca : amp/meter**2
        P_Ca : meter/second  # maximum Permeability to Calcium
        G_Ca = {Z_Ca}**2*F*v*gamma*({Ca_i}*nM - {Ca_o}*mM*exp(-{Z_Ca}*gamma*v))/(1 - exp(-{Z_Ca}*gamma*v)) : coulomb/meter**3
        m{name}_inf = 1/(1 + exp(-(v/mV + 56)/6.2)) : 1
        h{name}_inf = 1/(1 + exp((v/mV + 80)/4)) : 1
        tau_m{name} = (0.612 + 1.0/(exp(-(v/mV + 131)/16.7) + exp((v/mV + 15.8)/18.2))) * ms / {tadj_mLTCa}: second
        tau_h{name} = (int(v<-81*mV) * exp((v/mV + 466)/66.6) +
               int(v>=-81*mV) * (28 + exp(-(v/mV + 21)/10.5))) * ms / {tadj_hLTCa}: second
        dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
        dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
        """

        super().__init__(name, params)
        
    def default_params(self):
        return dict(Z_Ca=2, # Valence of Calcium ions
                    Ca_i= 240, # nM, intracellular Calcium concentration
                    Ca_o = 2, # mM, extracellular Calcium concentration
                    tadj_mLTCa = 2.5**((34-24)/10.0),
                    tadj_hLTCa = 2.5**((34-24)/10.0))



class HighVoltageActivationCurrent(MembraneCurrent):
    """
    Genealogy (NEURON comment):

    - adapted from 
    26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
    "ca.mod" Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal
    Uses fixed eca instead of GHK eqn
    -
    HVA Ca current
    Author: Zach Mainen, Salk Institute, 1994, zach@salk.edu
    -
    Based on Reuveni, Friedman, Amitai and Gutnick (1993) J. Neurosci. 13: 4609-4621.
    -----------------------------------------------------------------------------------
    
    """
    def __init__(self, name='HVACa', params=None):

        self.equations = """
        I{name} = g{name} * (v - {E_Ca}*mV) : amp/meter**2
	g{name} = tadj{name} * m{name}*m{name}*h{name} * {gbar}*1e-12*siemens/um**2 : siemens/meter**2
        tadj{name} = {q10}**(({celsius} - {temp})/10): 1
	a_m{name} = 0.055*(-27 - v/mV)/(exp((-27-v/mV)/3.8) - 1) : 1
	b_m{name} = 0.94*exp((-75-v/mV)/17) : 1
	tau_m{name} = 1/tadj{name}/(a_m{name}+b_m{name})*ms : second
	m{name}_inf = a_m{name}/(a_m{name}+b_m{name}) : 1
	a_h{name} = 0.000457*exp((-13-v/mV)/50) : 1
	b_h{name} = 0.0065/(exp((-v/mV-15)/28) + 1) : 1
	tau_h{name} = 1/tadj{name}/(a_h{name}+b_h{name})*ms : second
	h{name}_inf = a_h{name}/(a_h{name}+b_h{name}) : 1
        dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
        dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
        """
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_Ca = 140, # mV
            gbar = 0., #(pS/um2)
	    vshift = 0, # mV	: voltage shift (affects all)
	    cao = 2, # (mM) : external ca concentration
	    cai = 240, # (nM)	: internal ca concentration
	    temp = 23, # (degC)		: original temp 
	    celsius = 34, # (degC)	    : true temp 
	    q10  = 2.3, #			: temperature sensitivity
	    vmin = -120, # (mV)
	    vmax = 100)


class PotassiumChannelCurrent(MembraneCurrent):
    """
    Genealogy (NEURON comment):

    - adapted from 
    26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
    "kv.mod" Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal
    - Potassium channel, Hodgkin-Huxley style kinetics
    Kinetic rates based roughly on Sah et al. and Hamill et al. (1991)
    Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
    -----------------------------------------------------------------------------------
    
    """
    def __init__(self, name='K', params=None):

        self.equations = """
        I{name} = g{name} * (v - {E_K}*mV) : amp/meter**2
        gbar_{name} : siemens/meter**2
        g{name} = tadj{name}*n{name} * gbar_{name} : siemens/meter**2
        a{name} = {Ra} * (v/mV - {tha}) / (1 - exp(-(v/mV - {tha})/{qa})) : 1
        b{name} = -{Rb} * (v/mV - {tha}) / (1 - exp((v/mV - {tha})/{qa})) : 1
        tadj{name} = {q10}**(({celsius} - {temp})/10): 1
	tau_n{name} = 1/tadj{name}/(a{name}+b{name})*ms : second 
	n{name}_inf = a{name}/(a{name}+b{name}) : 1
        dn{name}/dt = -(n{name} - n{name}_inf)/tau_n{name} : 1
        """
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_K=-90, # mV
            tha = 25, # mV
            qa = 9, # mV
	    temp = 23, # (degC)		: original temp 
	    celsius = 34, # (degC)	    : true temp 
            q10 = 2.3, 
            Ra = 0.02, # kHz=1/ms
            Rb = 0.002) # kHz

"""
FUNCTION trap0(v,th,a,q) {
	if (fabs((v-th)/q) > 1e-6) {
	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
	        trap0 = a * q
 	}
}	
"""
def trap0(v,th,a,q):
    if sum(abs((v-th)/q) < 1e-6)==0: # meaning that they are all passing the above criteria
        return a*(v-th)/(1-exp(-(v - th)/q))
    else:
        return a*q
trap0 = Function(trap0, arg_units=[1, 1, 1, 1], return_unit=1)

    
class SodiumChannelCurrent(MembraneCurrent):
    """
    Sodium channel, Hodgkin-Huxley style kinetics.  

    Genealogy (NEURON comment):

    -----------------------------------------------------------------------------------
    26 Ago 2002 Modification of original channel to allow variable time
    step and to correct an initialization error.
    "na.mod" Done by Michael Hines(michael.hines@yale.e) and Ruggero
    Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational
    Neuroscience. Obidos, Portugal
    11 Jan 2007
    Glitch in trap where (v/th) was where (v-th)/q is. (thanks Ronald
    van Elburg!)

    Kinetics were fit to data from Huguenard et al. (1988) and Hamill et
    al. (1991)

    qi is not well constrained by the data, since there are no points
    between -80 and -55.  So this was fixed at 5 while the thi1,thi2,Rg,Rd
    were optimized using a simplex least square proc

    voltage dependencies are shifted approximately from the best
    fit to give higher threshold
    Author: Zach Mainen, Salk Institute, 1994, zach@salk.edu    

    ---------------------------

    """
    def __init__(self, name='Na', params=None):

        self.equations = """
        I{name} = g{name} * (v + {vshift}*mV - {E_Na}*mV) : amp/meter**2
        gbar_{name} : siemens/meter**2
        g{name} = tadj{name}*m{name}**3 *h{name} * gbar_{name} : siemens/meter**2
        tadj{name} = {q10}**(({celsius} - {temp})/10): 1
	a_m{name} = {Ra}/ms*{qa}/exprel(-(v/mV-{tha})/{qa}): 1/second
	b_m{name} = {Rb}/ms*{qa}/exprel((v/mV-{tha})/{qa}): 1/second
	tau_m{name} = 1/tadj{name}/(a_m{name}+b_m{name}) : second
	m{name}_inf = a_m{name}/(a_m{name}+b_m{name}) : 1
	a_h{name} = {Rd}/ms*{qi}/exprel(-(v/mV-{thi1})/{qi}): 1/second
	b_h{name} = {Rg}/ms*{qi}/exprel((v/mV-{thi2})/{qi}): 1/second
	tau_h{name} = 1/tadj{name}/(a_h{name}+b_h{name}) : second
	h{name}_inf = 1/(1+exp((v/mV-{thinf})/{qinf})) : 1
        dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
        dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
        """
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_Na=60, # mV
	    vshift = -5,#	(mV)		: voltage shift (affects all)
	    tha  = -35,#	(mV)		: v 1/2 for act		(-42)
	    qa   = 9, #	(mV)		: act slope		
	    Ra   = 0.182, #	(/ms)		: open (v)		
	    Rb   = 0.124, #	(/ms)		: close (v)		
	    thi1  = -50, #	(mV)		: v 1/2 for inact 	
	    thi2  = -75, #	(mV)		: v 1/2 for inact 	
	    qi   = 5, #	(mV)	        : inact tau slope
	    thinf  = -65, #	(mV)		: inact inf slope	
	    qinf  = 6.2, #	(mV)		: inact inf slope
	    Rg   = 0.0091, #	(/ms)		: inact (v)	
	    Rd   = 0.024, #	(/ms)		: inact recov (v) 
	    temp = 23, #	(degC)		: original temp 
	    celsius = 34, # (degC)	    : true temp 
	    q10  = 2.3)

class CalciumDependentPotassiumCurrent(MembraneCurrent):
    """
    Calcium-dependent potassium channel

    Genealogy (NEURON comment):

    - 
    26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
    "kca.mod" Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal
    -
    Based on
    Pennefather (1990) -- sympathetic ganglion cells
    taken from
    Reuveni et al (1993) -- neocortical cells
    -
    Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
    -----------------------------------------------------------------------------------
    
    """
    def __init__(self, name='KCa', params=None):

        self.equations = """
        I{name} = g{name} * (v + {vshift}*mV - {E_K}*mV) : amp/meter**2
        g{name} = tadj{name}*m{name}**3 *h{name} * {gbar}*1e-12*siemens/um**2 : siemens/meter**2

        a{name} = {Ra} * (v/mV - {tha}) / (1 - exp(-(v/mV - {tha})/{qa})) : 1
        b{name} = {Rb} : 1

        tadj{name} = {q10}**(({celsius} - {temp})/10): 1
	tau_n{name} = 1/tadj{name}/(a{name}+b{name})*ms : second 
	n{name}_inf = a{name}/(a{name}+b{name}) : 1
        dn{name}/dt = -(n{name} - n{name}_inf)/tau_n{name} : 1
        """
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_K=-90, # mV
	    gbar = 0., #   	(pS/um2)	: 0.03 mho/cm2
	    caix = 1, #	
	    Ra   = 0.01, #	(/ms)		: max act rate  
	    Rb   = 0.02, #	(/ms)		: max deact rate 
	    temp = 23, #	(degC)		: original temp 	
	    celsius = 34, # (degC)	    : true temp 
	    q10  = 2.3)
    
class HyperpolarizationActivatedCationCurrent(MembraneCurrent):
    """
    hyperpolarization-activated cation current (Ih)

    I-h channel from Magee 1998 for distal dendrites
    """
    def __init__(self, name='h', params=None):

        self.equations = """
        I{name} = l{name} * (v - {E_hdb}*mV) * {gbar}*1e-12*siemens/um**2 : amp/meter**2

        a{name} = {Ra} * (v/mV - {tha}) / (1 - exp(-(v/mV - {tha})/{qa})) : 1
        b{name} = {Rb} : 1

	tau_l{name} = 1/(a{name}+b{name})*ms : second 
	l{name}_inf = a{name}/(a{name}+b{name}) : 1
        dl{name}/dt = -(l{name} - l{name}_inf)/tau_l{name} : 1
        """
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_hdb=0,#  (mV)        
            gbar=0.,# pS/um2, (mho/cm2)
            vhalfl=-81, #   (mV)
            kl=-8, #
            vhalft=-75, #   (mV)
            a0t=0.011, #      (/ms)
            zetat=2.2, #    (1)
            gmt=.4, #   (1)
            q10=4.5, #
            qtl=1, #
	    temp = 23)


################################################
########### Concentration Mechanisms ###########
################################################

def lin_rectified(x):
    if x>=0:
        return x
    else:
        return 0
    # return (sign(x)+1)/2.*x
lin_rectified = Function(lin_rectified,
                         arg_units=[molar/second],
                         return_unit=molar/second)

class CalciumConcentrationDecay:
    """
    /!\ Needs to be coupled with a Calcium current (e.g. the HighVoltageActivationCurrent)


    Genealogy (NEURON comment):

    ---------------------------------------------------------------------------------------
    :26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
    : Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal
 
    TITLE decay of internal calcium concentration
    :
    : Internal calcium concentration due to calcium currents and pump.
    : Differential equations.
    :
    : Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)
    :     Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
    : A Michaelis-Menten approximation is assumed, which reduces the complexity
    : of the system to 2 parameters: 
    :       kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
    :	kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE
    : The values of these parameters are chosen assuming a high affinity of 
    : the pump to calcium and a low transport capacity (cfr. Blaustein, 
    : TINS, 11: 438, 1988, and references therein).  
    :
    : Units checked using "modlunit" -> factor 10000 needed in ca entry
    :
    : VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering)
    :
    : This mechanism was published in:  Destexhe, A. Babloyantz, A. and 
    : Sejnowski, TJ.  Ionic mechanisms for intrinsic slow oscillations in
    : thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993)
    : Written by Alain Destexhe, Salk Institute, Nov 12, 1992


    -- linear rectification because cannot pump inward (see above function)
    """
    def __init__(self,
                 name='CaDecay', # not a current, so not needed
                 contributing_current='IHVACa',
                 params=None):
        self.name=name
        if params is None:
            self.params = self.default_params()
        else:
            self.params = params
        self.params['name'] = self.name
        self.params['contributing_current'] = contributing_current
            
        # self.equations ="""
	# dCa/dt = lin_rectified(-{contributing_current}/2/F/{depth}/um)+({cainf}*mM-Ca)/{taur}/ms : mM"""
        self.equations ="""
	dCa/dt = -{contributing_current}/2/F/{depth}/um+({cainf}*mM-Ca)/{taur}/ms : mM"""
        self.code = self.equations.format(**self.params)
        
    def insert(self, eqs):
        return eqs+self.code

    def init_sim(self, neuron):
        neuron.Ca = params['cainf']*mM
                
    def default_params(self):
        return dict(
	    depth = .1, #	(um)		: depth of shell
	    taur = 200, #	(ms)		: rate of calcium removal
	    cainf = 100e-6, # (mM)
	    cai = 240) # nM
                
    
if __name__=='__main__':

    defaultclock.dt = 0.01*ms

    iHH = HodgkinHuxleyCurrent()
    # iT = LowThresholdCalciumCurrent()
    # iHVACa = HighVoltageActivationCurrent()
    # iK = PotassiumChannelCurrent()
    # iNa = SodiumChannelCurrent()
    iPas = PassiveCurrent()

    # Equation_String = CalciumConcentrationDecay().insert(Equation_String)
    
    # for current in [iT, iHH, iHVACa, iPas, iK]:
    for current in [iHH, iPas]:
        Equation_String = current.insert(Equation_String)
    
    eqs = Equations(Equation_String)
    
    # Simplified three-compartment morphology
    morpho = Cylinder(x=[0, 30]*um, diameter=20*um)
    morpho.dend = Cylinder(x=[0, 20]*um, diameter=10*um)
    morpho.dend.distal = Cylinder(x=[0, 500]*um, diameter=3*um)
    neuron = SpatialNeuron(morpho, eqs, Cm=0.88*uF/cm**2, Ri=173*ohm*cm,
                           method='exponential_euler')

    # for current in [iT, iHH, iHVACa, iK]:
    for current in [iHH, iPas]:
        current.init_sim(neuron)
        
    neuron.v = -75*mV
    
    # Only the soma has Na/K channels
    neuron.gHH_Na = 100*msiemens/cm**2
    neuron.gHH_K = 100*msiemens/cm**2
    neuron.dend.gHH_Na = 0*msiemens/cm**2
    neuron.dend.gHH_K = 0*msiemens/cm**2
    neuron.dend.distal.gHH_Na = 0*msiemens/cm**2
    neuron.dend.distal.gHH_K = 0*msiemens/cm**2

    
    mon = StateMonitor(neuron, ['v'], record=[0, 2])

    # neuron.P_Ca = 0*cm/second
    # neuron.dend.distal.P_Ca = 0*cm/second
    
    run(100*ms)
    neuron.main.I_inj = 300*pA
    run(200*ms)
    neuron.main.I_inj = 0*pA
    run(100*ms)
    # WITH T-CURRENT
    # neuron.P_Ca = 1.7e-5*cm/second
    # neuron.dend.distal.P_Ca = 9.5e-5*cm/second
    neuron.main.I_inj = 300*pA
    run(200*ms)
    neuron.main.I_inj = 0*pA
    run(200*ms)

    # ## Run the various variants of the model to reproduce Figure 12
    from datavyz import ges as ge
    fig, ax = ge.figure()
    ax.plot(mon.t / ms, mon[0].v/mV, 'k')
    ax.plot(mon.t / ms, mon[2].v/mV, 'blue')
    ge.show()
