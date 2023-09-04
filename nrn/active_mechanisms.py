"""
tries to implement a NEURON-like way to incorporate mechanisms into morphologically detailed cells
"""
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

        # initialize params to class default
        self.params = self.default_params()
        if params is not None:
            for key in params:
                self.params[key] = params[key] # passed values overrides default
            
        self.params['name'] = self.name
        
        self.compute_code(self.params) # build the equations with the params
        
        
    def compute_code(self, params):
        # we format the string equation with the parameters
        self.code = self.equations.format(**params)

        
    def insert(self, eqs):
        # returns the updated equation string
        return eqs.replace('Im = ', 'Im = - I%s ' % self.name)+self.code


    def init_sim(self, neuron, verbose=True):
        # if model has HH-like variable ("m","n","h","l") -> we initialize to inf values if possible, else 0
        for l in ['m', 'n', 'h', 'l']:
            if hasattr(neuron, l+self.name):
                try:
                    setattr(neuron, l+self.name, l+self.name+'_inf')
                except KeyError:
                    setattr(neuron, l+self.name, 0)
                if verbose:
                    print(l+self.name, 'was set to :', getattr(neuron, l+self.name))


class PassiveCurrent(MembraneCurrent):
    """
    basic leak current

    parameters El, gl
    """
    def __init__(self, name='Pas', params=None):
        
        self.equations = """
I{name} = gbar_{name}*(v-{El}*mV): amp/meter**2
gbar_{name} : siemens/meter**2
"""
        
        super().__init__(name, params)
    
    def default_params(self):
        return dict(El=-75.) # mV

    
    
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
I{name}_Na = gbarNa_{name} * m{name}**3 * h{name} * (v-{E_Na}*mV) : amp/meter**2
I{name}_K = gbarK_{name} * n{name}**4 * (v-{E_K}*mV) : amp/meter**2
gbarNa_{name} : siemens/meter**2
gbarK_{name} : siemens/meter**2
v2 = v - {VT}*mV : volt  # shifted membrane potential (Traub convention)
dm{name}/dt = (0.32*(mV**-1)*(13.*mV-v2)/
  (exp((13.*mV-v2)/(4.*mV))-1.)*(1-m{name})-0.28*(mV**-1)*(v2-40.*mV)/
  (exp((v2-40.*mV)/(5.*mV))-1.)*m{name}) / ms * {tadj}: 1
dn{name}/dt = (0.032*(mV**-1)*(15.*mV-v2)/
  (exp((15.*mV-v2)/(5.*mV))-1.)*(1.-n{name})-.5*exp((10.*mV-v2)/(40.*mV))*n{name}) / ms * {tadj}: 1
dh{name}/dt = (0.128*exp((17.*mV-v2)/(18.*mV))*(1.-h{name})-4./(1+exp((40.*mV-v2)/(5.*mV)))*h{name}) / ms * {tadj}: 1
"""
        
        super().__init__(name, params)
        
    def default_params(self):
        return dict(VT=-52.,#mV
                    E_Na=50.,#mV
                    E_K=-100.,#mV
                    tadj=3.0**((34-36)/10.0))


        
class LowThresholdCalciumCurrent(MembraneCurrent):
    """
    Low-threshold Calcium current (iT)

    Genealogy (NEURON comment):

    T-type Ca channel 
    ca.mod to lead to thalamic ca current inspired by destexhe and huguenrd
    Uses fixed eca instead of GHK eqn
    changed from (AS Oct0899)
    changed for use with Ri18  (B.Kampa 2005)
    """
    def __init__(self, name='T',
                 params=None):
        
        self.equations = """
I{name} = g{name} * (v - {E_Ca}*mV): amp/meter**2
g{name} = gbar_{name} * m{name}**2 * h{name} : siemens/meter**2
gbar_{name} : siemens/meter**2
m{name}_inf = 1.0 / ( 1 + exp(-(v/mV+{v12m})/{vwm}) ) : 1 
h{name}_inf = 1.0 / ( 1 + exp((v/mV+{v12h})/{vwh}) ) : 1 
tau_m{name} = ( {am} + 1.0 / ( exp((v/mV+{vm1})/{wm1}) + exp(-(v/mV+{vm2})/{wm2}) ) ) * ms : second
tau_h{name} = ( {ah} + 1.0 / ( exp((v/mV+{vh1})/{wh1}) + exp(-(v/mV+{vh2})/{wh2}) ) ) * ms : second
dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
"""

        super().__init__(name, params)
        
    def default_params(self):
        return dict(E_Ca = 140., # mV
	            vshift = 0., #	(mV)		: voltage shift (affects all)
	            cao  = 2.5, #	(mM)	        : external ca concentration
	            v12m=50., #         	(mV)
	            v12h=78., #         	(mV)
	            vwm =7.4, #         	(mV)
	            vwh=5.0, #         	(mV)
	            am=3., #         	(mV)
	            ah=85., #         	(mV)
	            vm1=25., #         	(mV)
	            vm2=100., #         	(mV)
	            vh1=46., #         	(mV)
	            vh2=405., #         	(mV)
	            wm1=20., #         	(mV)
	            wm2=15., #         	(mV)
	            wh1=4., #         	(mV)
	            wh2=50., #         	(mV)
                    tadj = 2.5**((34-24)/10.0))



class HighVoltageActivationCalciumCurrent(MembraneCurrent):
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
g{name} = gbar_{name} * {tadj} * m{name}*m{name}*h{name} : siemens/meter**2
gbar_{name} : siemens/meter**2
# --> activation: m
a_m{name} = 0.055*(-27 - v/mV)/expm1((-27-v/mV)/3.8) : 1
b_m{name} = 0.94*exp((-75. - v/mV)/17.) : 1
h{name}_inf = a_h{name}/(a_h{name}+b_h{name}) : 1
tau_m{name} = 1/{tadj}/(a_m{name}+b_m{name})*second : second
dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
# --> inactivation: h
a_h{name} = 0.000457*exp((-13-v/mV)/50.) : 1
b_h{name} = 0.0065/(exp((-v/mV-15)/28.) + 1) : 1
m{name}_inf = a_m{name}/(a_m{name}+b_m{name}) : 1
tau_h{name} = 1/{tadj}/(a_h{name}+b_h{name})*second : second
dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
"""
        
        super().__init__(name, params)


    def default_params(self):
        return dict(
            E_Ca = 140., # mV
	    vshift = 0., # mV	: voltage shift (affects all)
	    cao = 2., # (mM) : external ca concentration
	    cai = 240., # (nM)	: internal ca concentration
	    vmin = -120., # (mV)
	    vmax = 100,
	    tadj = 2.3**((34-23)/10))


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
g{name} = gbar_{name} * {tadj} * n{name} : siemens/meter**2
gbar_{name} : siemens/meter**2
a{name} = {Ra} * (v/mV - {tha}) / (1 - exp(-(v/mV - {tha})/{qa})) : 1
b{name} = -{Rb} * (v/mV - {tha}) / (1 - exp((v/mV - {tha})/{qa})) : 1
tau_n{name} = 1/{tadj}/(a{name}+b{name})*ms : second 
n{name}_inf = a{name}/(a{name}+b{name}) : 1
dn{name}/dt = -(n{name} - n{name}_inf)/tau_n{name} : 1
"""
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_K=-90., # mV
            tha = 25., # mV
            qa = 9., # mV
            tadj = 2.3**((34-23)/10),
            Ra = 0.02, # kHz=1/ms
            Rb = 0.002) # kHz

    
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
I{name} = g{name} * (v - {E_Na}*mV) : amp/meter**2
gbar_{name} : siemens/meter**2
g{name} = gbar_{name} * {tadj} * m{name}**3 *h{name} : siemens/meter**2
a_m{name} = {Ra}/ms*{qa}/exprel(-(clip(v/mV+{vshift},-120,100)-{tha})/{qa}): 1/second
b_m{name} = {Rb}/ms*{qa}/exprel(-(-clip(v/mV+{vshift},-120,100)+{tha})/{qa}): 1/second
tau_m{name} = 1/{tadj}/(a_m{name}+b_m{name}) : second
m{name}_inf = a_m{name}/(a_m{name}+b_m{name}) : 1
a_h{name} = {Rd}/ms*{qi}/exprel(-(clip(v/mV+{vshift},-120,100)-{thi1})/{qi}): 1/second
b_h{name} = {Rg}/ms*{qi}/exprel(-(-clip(v/mV+{vshift},-120,100)+{thi2})/{qi}): 1/second
tau_h{name} = 1/{tadj}/(a_h{name}+b_h{name}) : second
h{name}_inf = 1/(1+exp((clip(v/mV+{vshift}, -120, 100)-{thinf})/{qinf})) : 1
dm{name}/dt = -(m{name} - m{name}_inf)/tau_m{name} : 1
dh{name}/dt = -(h{name} - h{name}_inf)/tau_h{name} : 1
"""
	# h{name}_inf = a_h{name}/(a_h{name}+b_h{name}) : 1
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_Na=60., # mV
	    vshift = -10,#	(mV)		: voltage shift (affects all)
	    tha  = -35,#	(mV)		: v 1/2 for act		(-42)
	    qa   = 9., #	(mV)		: act slope		
	    Ra   = 0.182, #	(/ms)		: open (v)		
	    Rb   = 0.124, #	(/ms)		: close (v)		
	    thi1  = -50., #	(mV)		: v 1/2 for inact 	
	    thi2  = -75., #	(mV)		: v 1/2 for inact 	
	    qi   = 5., #	(mV)	        : inact tau slope
	    thinf  = -65., #	(mV)		: inact inf slope	
	    qinf  = 6.2, #	(mV)		: inact inf slope
	    Rg   = 0.0091, #	(/ms)		: inact (v)	
	    Rd   = 0.024, #	(/ms)		: inact recov (v) 
            tadj = 2.3**((34-23)/10))

    
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
I{name} = g{name} * (v - {E_K}*mV) : amp/meter**2
g{name} = gbar_{name} * {tadj}* n{name} : siemens/meter**2
gbar_{name} : siemens/meter**2
a{name} = {Ra} * (InternalCalcium/({InternalCalcium0}*uM))**{ExpCai} : 1
b{name} = {Rb} : 1
tau_n{name} = 1/{tadj}/(a{name}+b{name})*ms : second 
n{name}_inf = a{name}/(a{name}+b{name}) : 1
dn{name}/dt = -(n{name} - n{name}_inf)/tau_n{name} : 1
"""
        
        super().__init__(name, params)

    def insert(self, eqs):
        """
        we override the default "insert()" to check for the internal [Ca2+] variable
        """
        if (len(eqs.split('InternalCalcium : mmolar'))<2) and (len(eqs.split('dInternalCalcium'))<2):
            # not present
            raise BaseException("""
            -------------------------------------------------------------------
            /!\ No mechanism to describe the Calcium Concentration dynamics inserted !! /!\

            Need to include one, e.g. check the "CalciumConcentrationDynamics" class
            -------------------------------------------------------------------
            """)
        else:
            return super().insert(eqs) # then we can insert as usual
            
    def default_params(self):
        return dict(
            E_K=-90., # mV
            InternalCalcium0 = 1., # uM
	    ExpCai = 1., # exponent for InternalCalcium term
	    Ra   = 0.01, #	(/ms)		: max act rate  
	    Rb   = 0.02, #	(/ms)		: max deact rate 
	    tadj = 2.3**((34-23)/10))

    
class HyperpolarizationActivatedCationCurrent(MembraneCurrent):
    """
    hyperpolarization-activated cation current (Ih)

    I-h channel from Magee 1998 for distal dendrites
    """
    def __init__(self, name='H', params=None):

        self.equations = """
I{name} = g{name} * (v - {E_hdb}*mV) : amp/meter**2
g{name} = gbar_{name} * l{name} : siemens/meter**2
gbar_{name} : : siemens/meter**2
a{name} = 1/(1+ exp(0.0378*{zetat}*(v/mV-{vhalft}))) : 1
tau_l{name} = exp(0.0378*{zetat}*{gmt}*(v/mV-{vhalft})) / ({qtl}*{qt}*{a0t}*(1+a{name})) *ms : second 
l{name}_inf = 1/(1+a{name}) : 1
dl{name}/dt = -(l{name} - l{name}_inf)/tau_l{name} : 1
"""
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_hdb=0,#  (mV)        
            vhalfl=-81., #   (mV)
            kl=-8., #
            vhalft=-75., #   (mV)
            a0t=0.011, #      (/ms)
            zetat=2.2, #    (1)
            gmt=.4, #   (1)
            qtl=1., #
            qt=4.5**((34-33)/10))


class MuscarinicPotassiumCurrent(MembraneCurrent):
    """
    Im current

    26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
       Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal

    "km.mod"
    Potassium channel, Hodgkin-Huxley style kinetics
    Based on I-M (muscarinic K channel)
    Slow, noninactivating

    Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
    """
    def __init__(self, name='Musc', params=None):

        self.equations = """
I{name} = g{name} * (v - {E_K}*mV) : amp/meter**2
g{name} = gbar_{name} * n{name} : siemens/meter**2
gbar_{name} : siemens/meter**2
a{name} = {Ra} * (v/mV - {tha}) / (1 - exp(-(v/mV - {tha})/{qa})) : 1
b{name} = {Rb} : 1
tau_n{name} = 1/{tadj}/(a{name}+b{name})*ms : second 
n{name}_inf = a{name}/(a{name}+b{name}) : 1
dn{name}/dt = -(n{name} - n{name}_inf)/tau_n{name} : 1
"""
        
        super().__init__(name, params)

        
    def default_params(self):
        return dict(
            E_K = -90., # (mV)
	    tha  = -30., #	(mV)		: v 1/2 for inf
	    qa   = 9., #	(mV)		: inf slope		
	    Ra   = 0.001, #	(/ms)		: max act rate  (slow)
	    Rb   = 0.001, #	(/ms)		: max deact rate  (slow)
	    tadj = 2.3**((34-23)/10))
    
################################################
########### Concentration Mechanisms ###########
################################################

class CalciumConcentrationDynamics:
    """
    /!\ Needs to be coupled with a Calcium current (e.g. the HighVoltageActivationCalciumCurrent)


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

    ====================================
    Y. Zerlaut (2020): hard to understand the unit factor
    so reversed engineered from the NEURON code using:
    1 mA/cm**2/2/F/um = 51821.34786953 * nmolar / msecond
    """
    def __init__(self,
                 name='CaDynamics', # not a current, so not really needed (do not need to be identified)
                 contributing_currents='IHVACa',
                 params=None):
        self.name=name
        if params is None:
            self.params = self.default_params()
        else:
            self.params = params
        self.params['name'] = self.name
        self.params['contributing_currents'] = contributing_currents


        # drive_channel = -0.05182136/{depth}*({contributing_currents})/mA*cm**2*nmolar/ms : mmolar/second
        self.equations ="""
drive_channel = -0.5182*({contributing_currents})/mA*cm**2*mmolar/ms : mmolar/second
dInternalCalcium/dt = drive_channel+({cainf}*mmolar-InternalCalcium)/{taur}/ms : mmolar
"""
        self.code = self.equations.format(**self.params)
        
    def insert(self, eqs):
        return eqs+self.code

    def init_sim(self, neuron):
        neuron.InternalCalcium = params['cainf']*mmolar
                
    def default_params(self):
        return dict(
	    depth = .1, #	(um)		: depth of shell
	    taur = 200., #	(ms)		: rate of calcium removal
	    cainf = 100e-6, # (mM)
	    cai = 240) # nM
                
    
if __name__=='__main__':

    defaultclock.dt = 0.025*ms

    # Starting from an empty equation string:
    Equation_String= '''
Im = + 0*amp/meter**2 : amp/meter**2
I_inj : amp (point current)
'''

    # calcium dynamics following: HighVoltageActivationCalciumCurrent + LowThresholdCalciumCurrent
    Equation_String = CalciumConcentrationDynamics(contributing_currents='IT+IHVACa',
                                             name='CaDynamics').insert(Equation_String)

    # intrinsic currents
    CURRENTS = [PassiveCurrent(name='Pas'),
                PotassiumChannelCurrent(name='K'),
                SodiumChannelCurrent(name='Na'),
                HighVoltageActivationCalciumCurrent(name='HVACa'),
                LowThresholdCalciumCurrent(name='T'),
                MuscarinicPotassiumCurrent(name='Musc'),
                CalciumDependentPotassiumCurrent(name='KCa')]
    
    for current in CURRENTS:
        Equation_String = current.insert(Equation_String)
    
    print(Equation_String)

    eqs = Equations(Equation_String)
    
    # Simplified three-compartment morphology
    morpho = Cylinder(x=[0, 30]*um, diameter=20*um)
    morpho.dend = Cylinder(x=[0, 20]*um, diameter=10*um)
    morpho.dend.distal = Cylinder(x=[0, 500]*um, diameter=3*um)
    neuron = SpatialNeuron(morpho, eqs, Cm=1*uF/cm**2, Ri=150*ohm*cm,
                           method='exponential_euler')

    # initial conditions:
    neuron.v = -75*mV
    neuron.InternalCalcium = 100*nM

    for current in CURRENTS:
        current.init_sim(neuron)

    ## -- PASSIVE PROPS -- ##
    neuron.gbar_Pas = 1e-4*siemens/cm**2

    ## -- SPIKE PROPS (Na & Kv) -- ##
    # soma
    neuron.gbar_Na = 1500*1e-12*siemens/um**2
    neuron.gbar_K = 200*1e-12*siemens/um**2
    # dendrites
    neuron.dend.gbar_Na = 40*1e-12*siemens/um**2
    neuron.dend.gbar_K = 30*1e-12*siemens/um**2
    neuron.dend.distal.gbar_Na = 40*1e-12*siemens/um**2
    neuron.dend.distal.gbar_K = 30*1e-12*siemens/um**2

    ## -- HIGH-VOLTAGE-ACTIVATION CALCIUM CURRENT -- ##
    neuron.gbar_HVACa = 0.5*1e-12*siemens/um**2

    ## -- CALCIUM-DEPENDENT POTASSIUM CURRENT -- ##
    neuron.gbar_KCa = 2.5*1e-12*siemens/um**2

    ## -- T-CURRENT (Calcium) -- ##
    neuron.gbar_T = 0.0003*1e-12*siemens/um**2
    neuron.dend.gbar_T = 0.0006*1e-12*siemens/um**2
    neuron.dend.distal.gbar_T = 0.0006*1e-12*siemens/um**2

    ## -- M-CURRENT (Potassium) -- ##
    neuron.gbar_Musc = 2.2*1e-12*siemens/um**2
    neuron.dend.gbar_Musc = 0.05*1e-12*siemens/um**2
    neuron.dend.distal.gbar_Musc = 0.05*1e-12*siemens/um**2

    # ## -- H-CURRENT (non-specific) -- ##
    # neuron.gbar_H = 0*1e-12*siemens/um**2 # set to zero !!

    soma_loc, dend_loc = 0, 2
    mon = StateMonitor(neuron, ['v', 'I_inj', 'InternalCalcium'], record=[soma_loc, dend_loc])

    run(100*ms)
    neuron.main.I_inj = 300*pA
    run(200*ms)
    neuron.main.I_inj = 0*pA
    run(100*ms)
    neuron.dend.I_inj = 300*pA
    run(200*ms)
    neuron.dend.I_inj = 0*pA
    run(200*ms)

    # # # ## Run the various variants of the model to reproduce Figure 12
    import matplotlib.pylab as plt
    fig, AX = plt.subplots(3,1, figsize=(12,4))

    AX[0].plot(mon.t / ms, mon[soma_loc].v/mV, color='blue', label='soma')
    AX[0].plot(mon.t / ms, mon[dend_loc].v/mV, color='red', label='dend')
    AX[0].set_ylabel('Vm (mV)')
    AX[0].legend()

    AX[1].plot(mon.t / ms, mon[soma_loc].InternalCalcium/nM, color='blue', label='soma')
    AX[1].plot(mon.t / ms, mon[dend_loc].InternalCalcium/nM, color='red', label='dend')
    AX[1].set_ylabel('[Ca$^{2+}$] (nM)')

    AX[2].plot(mon.t / ms, mon[soma_loc].I_inj/pA, color='blue', label='soma')
    AX[2].plot(mon.t / ms, mon[dend_loc].I_inj/pA, color='red', label='dend')
    AX[2].set_ylabel('Iinj (pA)')
    AX[2].set_xlabel('time (ms)')

    plt.show()
