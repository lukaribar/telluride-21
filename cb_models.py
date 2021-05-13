from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
import numpy as np
from numpy import exp
from copy import deepcopy

class HHKinetics(ABC):
    """
    HH-type (alpha-beta) kinetics abstract class.
    Has to be implemented by an activation or inactivation subclass.
    """
    @abstractmethod
    def alpha(self,V):
        pass

    @abstractmethod
    def beta(self,V):
        pass

    def inf(self,V):
        return self.alpha(V) / (self.alpha(V) + self.beta(V))

    def tau(self, V):
        return 1 / (self.alpha(V) + self.beta(V))

    def vfield(self,x,V,Vpos=[]):
        if Vpos == []:   
            # Intrinsic kinetics vector field
            return self.alpha(V)*(1 - x) - self.beta(V)*x
        else:
            # Synaptic kinetics vector field
            return self.alpha(V)*(1 - x) - self.beta(Vpos)*x

class NeuroDynRate:
    """
    NeuroDyn-type alpha or beta functions (kinetic rates)
    """
    def __init__(self,Ib,kappa,Vt,Vb,sign):
        self.Ib = Ib
        self.perturbations = np.ones(np.size(Ib))
        self.kappa = kappa
        self.Vt = Vt
        self.Vb = Vb
        self.sign = sign

    def I_rate(self,V):
        I=0
        Ib = self.Ib * self.perturbations
        for i in range(np.size(self.Ib)):
            I += Ib[i] / (1 + np.exp(self.sign * self.kappa * (self.Vb[i] - V)  / self.Vt))
        return I     
    
    def perturb(self, sigma=0.15):
        self.perturbations = (1 + sigma*np.random.randn(7))
        
class NeuroDynActivation(HHKinetics):
    """
    NeuroDyn-type activation gating variable kinetics.
    """
    def __init__(self,Ib,kappa,C,Vt,Vb):
        self.C = C
        self.Vt = Vt    
        self.alpharate = NeuroDynRate(Ib[0],kappa,Vt,Vb,1)
        self.betarate = NeuroDynRate(Ib[1],kappa,Vt,Vb,-1) 
    
    def alpha(self,V):
        return self.alpharate.I_rate(V) / (self.C * self.Vt)

    def beta(self,V):
        return self.betarate.I_rate(V) / (self.C * self.Vt)
    
class NeuroDynInactivation(HHKinetics):
    """
    NeuroDyn-type inactivation gating variable kinetics.
    """
    def __init__(self,Ib,kappa,C,Vt,Vb):
        self.C = C
        self.Vt = Vt
        self.alpharate = NeuroDynRate(Ib[0],kappa,Vt,Vb,-1) 
        self.betarate = NeuroDynRate(Ib[1],kappa,Vt,Vb,1) 
    
    def alpha(self,V):
        return self.alpharate.I_rate(V) / (self.C * self.Vt)

    def beta(self,V):
        return self.betarate.I_rate(V) / (self.C * self.Vt)

#self.m = HHActivation(25*scl_v, 0.1*scl_t/scl_v, 10*scl_v, 0*scl_v, 4*scl_t, 18*scl_v)
class HHActivation(HHKinetics):
    """
    HH-type (alpha-beta) activation gating variable kinetics.
    """
    def __init__(self, aVh, aA, aK, bVh, bA, bK, SI_units=False):
        self.aVh = aVh
        self.aA = aA
        self.aK = aK
        self.bVh = bVh
        self.bA = bA
        self.bK = bK
        
        # Convert paramaters to SI units
        if (SI_units):
            self.aVh *= 1e-3
            self.aA *= 1e6
            self.aK *= 1e-3
            self.bVh *= 1e-3
            self.bA *= 1e3
            self.bK *= 1e-3

    def alpha(self,V):
        A = self.aA
        K = self.aK
        Vh = self.aVh
        a =	np.zeros(np.size(V))
        V = np.array(V)
        a[V!=Vh] = A * (Vh - V[V!=Vh]) / (exp((Vh - V[V!=Vh]) / K) - 1)
        a[V==Vh] = A*K
        return a

    def beta(self,V):
        return self.bA * exp((self.bVh - V) / self.bK)

#self.h = HHInactivation(0*scl_v, 0.07*scl_t, 20*scl_v, 30*scl_v, 1*scl_t, 10*scl_v)
class HHInactivation(HHKinetics):
    """
    HH-type (alpha-beta) inactivation gating variable kinetics. 
    """
    def __init__(self, aVh, aA, aK, bVh, bA, bK, SI_units=False):
        self.aVh = aVh
        self.aA = aA
        self.aK = aK
        self.bVh = bVh
        self.bA = bA
        self.bK = bK
        
        # Convert paramaters to SI units
        if (SI_units):
            self.aVh *= 1e-3
            self.aA *= 1e3
            self.aK *= 1e-3
            self.bVh *= 1e-3
            self.bA *= 1e3
            self.bK *= 1e-3

    def alpha(self,V):
        return self.aA * exp((self.aVh - V)/ self.aK)

    def beta(self,V):
        return self.bA / (exp((self.bVh - V) / self.bK) + 1)

# Develop this in case we decide to work with very general models:
# class OhmicElement:
#     """
#     Single ohmic current element consisting of multiple gates:
#         Iout = g_max * x1 * x2 * ... * xn * (V - E_rev)
#         *args: [x1,x2,...,xn] = gates
#     """
#     def __init__(self, g_max, E_rev = 0, gates = [], expos = []):
#         self.g_max = g_max
#         self.E_rev = E_rev
#         self.gates = gates
#         self.expos = expos

#     # Add a gating variable to the conductance element
#     def add_gate(self, gates):
#         self.gates.append(gates)
#         return

#     def kinetics(self,V,X):
#         dx = np.array([])
#         for n in range(np.size(self.gates)):
#             dx.append(self.gates[n].vfield(X[n],V))
#         return dx
    
#     def I(self,V,X):
#         i_out = self.g_max * (V - self.E_rev)
#         for n in range(np.size(X))

class NeuronalModel(ABC):
    """
    Abstract class for neuronal models.
    """
    @abstractmethod
    def vfield(self, x, I):
        pass

    def simulate(self, trange, x0, Iapp, mode="continuous"):
        # Note: Iapp should be a function of t, e.g., Iapp = lambda t : I0
        if mode == "continuous":
            def odesys(t, x):
                return self.vfield(x, Iapp(t))
            return solve_ivp(odesys, trange, x0)
        else:
            #... code forward-Euler integration
            return

class NeuroDynModel(NeuronalModel):
    """
    NeuroDyn model
    """
    def __init__(self, dg=np.array([400, 160, 12]), dErev=np.array([450, -250, -150]),
                 dIb=[], V_ref=0.9, I_voltage = 270e-9, I_master = 200e-9,
                 I_ref = 100e-9):
        self.V_ref = V_ref              # Unit V
        self.I_voltage = I_voltage      # Unit A
        self.I_master = I_master              # Unit A
        self.I_ref = I_ref              # Unit A

        self.vHigh = self.V_ref + I_voltage*1.85*1e6
        self.vLow = self.V_ref - I_voltage*1.85*1e6
        
        # Membrane & gate capacitances
        self.C_m = 4e-12        # Unit F
        self.C_gate = 5e-12     # Unit F
        
        # Scaling parameters (e.g. parameters that set the voltage scale, time scale..)
        self.kappa = 0.7
        self.Vt = 26e-3     # Unit Volt
        self.Res = 1.63e6   # Unit Ohm
        
        # Digital parameters
        self.dg = dg
        self.dErev = dErev
                
        # Convert digital to physical
        self.gna,self.gk,self.gl = self.convert_conductance(dg)
        self.Ena,self.Ek,self.El = self.convert_potential(dErev)
        
        # Gating variable coefficients
        self.p = 3
        self.q = 1
        self.r = 4
        
        Vb = self.get_Vb()
        self.Vb = Vb
        
        # Default to nominal NeuroDyn activation parameters
        if (dIb == []):
            dIb_m = np.array([[0, 0, 120, 400, 800, 1023, 1023],
                     [1023, 1023, 1023, 1023, 0, 0, 0]])
            dIb_h = np.array([[237, 5, 7, 6, 0, 0, 0],
                    [0, 0, 0, 0, 41, 25, 8]])
            dIb_n = np.array([[0, 0, 0, 0, 80, 40, 250],
                    [4, 0, 0, 10, 0, 0, 4]])
            dIb = [dIb_m, dIb_h, dIb_n]
        self.dIb = dIb            
        
        Ib_m = self.convert_current(dIb[0])
        Ib_h = self.convert_current(dIb[1])
        Ib_n = self.convert_current(dIb[2])
        
        self.m = NeuroDynActivation(Ib_m,self.kappa,self.C_gate,self.Vt,Vb)
        self.h = NeuroDynInactivation(Ib_h,self.kappa,self.C_gate,self.Vt,Vb)
        self.n = NeuroDynActivation(Ib_n,self.kappa,self.C_gate,self.Vt,Vb)
            
    def convert_current(self, dI):
        # Factor for converting digital to physical I
        I_factor = self.I_master / 1024
        return dI * I_factor
    
    def convert_conductance(self, dg):
        # Factor for converting digital to physical g
        g_factor = (self.kappa / self.Vt) * (self.I_master / 1024)
        return dg * g_factor
        
    def convert_potential(self, dErev):
        # Factor for converting digital to physical Erev
        E_factor = (self.I_voltage / 1024) * self.Res
        return dErev * E_factor + self.V_ref

    def get_pars(self):
        params = {
            'kappa': self.kappa,
            'C': self.C_gate,
            'C_ND': self.C_m,
            'Vt': self.Vt,
            'Res': self.Res
        }
        return params
    
    def get_Vb(self):
        # Bias voltages for the 7-point spline regression
        Vb = np.zeros(7) # Define the 7 bias voltages
        I_factor = (self.vHigh - self.vLow) / 700e-3
        Vb[0] = self.vLow + (I_factor * 50e-3)
        for i in range(1, 7):
            Vb[i] = Vb[i-1] + (I_factor * 100e-3)
        return Vb
    
    def i_int(self,V, m, h, n):
        return (self.gna*(m**self.p)*(h**self.q)*(V - self.Ena) +
                self.gk*(n**self.r)*(V - self.Ek) + self.gl*(V - self.El))

    def iNa_ss(self,V):
        return self.gna*(self.m.inf(V)**self.p)*(self.h.inf(V)**self.q)*(V - self.Ena)

    def iK_ss(self,V):
        return self.gk*(self.n.inf(V)**self.r)*(V - self.Ek)    

    def iL_ss(self,V):
        return self.gl*(V - self.El)
    
    def vfield(self, x, I):
        V, m, h, n = x
        dV = (-self.i_int(V, m, h, n) + I)/self.C_m
        dm = self.m.vfield(m,V)
        dh = self.h.vfield(h,V)
        dn = self.n.vfield(n,V)
        return [dV, dm, dh, dn]

    def perturb(self,sigma=0.15):
        # Pertrub exponents
        self.p = 3 + 0.2*np.random.randn()
        self.q = 1 + 0.1*np.random.randn()
        self.r = 4 + 0.2*np.random.randn()
        
        # For each alpha/beta, perturb Itaus
        for x in [self.m, self.h, self.n]:
            x.alpharate.perturb(sigma)
            x.beta.perturb(sigma)
            
        # Perturb maximal conductances
        self.gna = self.convert_conductance(self.dg[0]*(1 + sigma*np.random.randn()))
        self.gk = self.convert_conductance(self.dg[1]*(1 + sigma*np.random.randn()))
        self.gl = self.convert_conductance(self.dg[2]*(1 + sigma*np.random.randn()))
        
        # Perturb reversal potentials
        self.Ena = self.convert_potential(self.dErev[0]*(1 + sigma*np.random.randn()))
        self.Ek = self.convert_potential(self.dErev[1]*(1 + sigma*np.random.randn()))
        self.El = self.convert_potential(self.dErev[2]*(1 + sigma*np.random.randn()))
        
        # Perturb voltage offsets?
        # Would add ~15mV sigma to each 'bias' voltage

class HHModel(NeuronalModel):
    """
        Hodgkin-Huxley model 
    """    
    def __init__(self, gna=120, gk=36, gl=0.3, Ena=120, Ek =-12, El =10.6,
                 gates=[], scl_v=1, scl_t=1, SI_units=False):
        self.C = 1
        self.gna = gna*scl_t
        self.gk = gk*scl_t
        self.gl = gl*scl_t
        self.Ena = Ena*scl_v
        self.Ek = Ek*scl_v
        self.El = El*scl_v
        
        self.scl_v = scl_v
        self.scl_t = scl_t
        self.SI_units = SI_units
        
        # Convert to SI units
        if (SI_units):
            self.C *= 1e-6
            self.gna *= 1e-3
            self.gk *= 1e-3
            self.gl *= 1e-3
            self.Ena *= 1e-3
            self.Ek *= 1e-3
            self.El *= 1e-3
        
        if not gates:
            # Default to nominal HH kinetics
            self.m = HHActivation(25*scl_v, 0.1*scl_t/scl_v, 10*scl_v, 0*scl_v,
                                  4*scl_t, 18*scl_v, SI_units)
            self.h = HHInactivation(0*scl_v, 0.07*scl_t, 20*scl_v, 30*scl_v,
                                    1*scl_t, 10*scl_v, SI_units)
            self.n = HHActivation(10*scl_v, 0.01*scl_t/scl_v, 10*scl_v,
                                  0*scl_v, 0.125*scl_t, 80*scl_v, SI_units)
        else:
            # We should perhaps scale the gates passed by the user as well
            self.m = gates[0]
            self.h = gates[1]
            self.n = gates[2]
        
        # Gating variable coefficients
        self.p = 3
        self.q = 1
        self.r = 4
        
        # Save the nominal parameters
        self.nominal = deepcopy(self)

    def i_int(self,V, m, h, n):
        return (self.gna*(m**self.p)*(h**self.q)*(V - self.Ena) +
                self.gk*(n**self.r)*(V - self.Ek) + self.gl*(V - self.El))

    def iNa_ss(self,V):
        return self.gna*(self.m.inf(V)**self.p)*(self.h.inf(V)**self.q)*(V - self.Ena)

    def iK_ss(self,V):
        return self.gk*(self.n.inf(V)**self.r)*(V - self.Ek)    

    def iL_ss(self,V):
        return self.gl*(V - self.El)

    def vfield(self, x, I):
        V, m, h, n = x
        
        # Do conversion here for external input?
        #if (self.SI_units):
        #    I *= 1e-6
        # IMPORTANT: External input should not be scaled here?
        dV = (-self.i_int(V, m, h, n) + I*self.scl_v*self.scl_t)/self.C
        dm = self.m.vfield(m,V)
        dh = self.h.vfield(h,V)
        dn = self.n.vfield(n,V)
        return [dV, dm, dh, dn]
        
    def perturb(self, sigma=0.15):
        nom = self.nominal
        
        # Pertrub exponents
        self.p = nom.p + 0.2*np.random.randn()
        self.q = nom.q + 0.1*np.random.randn()
        self.r = nom.r + 0.2*np.random.randn()
        
        # Perturb maximal conductances
        self.gna = nom.gna * (1 + sigma*np.random.randn())
        self.gk = nom.gk * (1 + sigma*np.random.randn())
        self.gl = nom.gl * (1 + sigma*np.random.randn())
        
        # Perturb reversal potential
        self.Ena = nom.Ena * (1 + sigma*np.random.randn())
        self.Ek = nom.Ek * (1 + sigma*np.random.randn())
        self.El = nom.El * (1 + sigma*np.random.randn())
        
        # Perturb alpha/beta rates
        gates = [self.m, self.h, self.n]
        nom_gates = [nom.m, nom.h, nom.n]
        for x, x_nom in zip(gates, nom_gates):
            x.aA = x_nom.aA * (1 + sigma*np.random.randn())
            x.bA = x_nom.bA * (1 + sigma*np.random.randn())

##### NETWORK-RELATED CLASSES #####

class NeuroDynAMPA(NeuroDynActivation):
    """
    AMPA Synapse in the neurodyn chip.
    Physiological values taken from Ermentrout et al. 2010, p. 161
    """
    def __init__(self,gsyn=1,Esyn=0,kappa=0.7,C=5e-12,Vt=26e-3,I_master=33e-9):
        self.gsyn = gsyn
        self.Esyn = Esyn
        # Physiological constants
        Tmax, ar, ad, Kp, V_T = 0.001, 1.1, 0.19, 0.005, 0.002
        dIb = [[Tmax*ar*C*Vt, 0, 0, 0, 0, 0, 0],
               [0, Tmax*ad*C*Vt, 0, 0, 0, 0, 0]]
        Vb = [V_T,-10,0,0,0,0,0] 
        # IMPORTANT: WE CAN'T REALLY USE THE SIGMOIDS THIS WAY.
        # WE NEED TO FIT THE 7 SIGMOIDS TO THE AMPA SIGMOID
        super().__init__(dIb,kappa,C,Kp*kappa,Vb) 

class AMPA(HHKinetics):
    """
    AMPA gating variable kinetics  
    Physiological values taken from Ermentrout et al. 2010, p. 161
    Note: Voltage values need to be shifted +65mV
    """
    def __init__(self,Tmax=1,Kp=5,V_T=2+65,ar=1.1,ad=0.19):
        self.Tmax = Tmax
        self.Kp = Kp
        self.V_T = V_T
        self.ar = ar        
        self.ad = ad

    def alpha(self,V):
        return self.ar * self.Tmax / (1+np.exp(-(V-self.V_T)/self.Kp))

    def beta(self,V):
        return self.ad

# Could be derived from general conductance class if we code it?
class Synapse:
    """
    Arbitrary synapse class
    Arguments:
        gsyn: maximal conductance
        Esyn: synapse reversal potential
        r: synapse activation kinetics
    """
    def __init__(self, gsyn, Esyn, r):
        self.gsyn = gsyn
        self.Esyn = Esyn
        self.r = r # HHKinetics class
        
    def Iout(self, r, Vpost):
        return self.gsyn * r * (Vpost - self.Esyn)

class AMPASynapse(Synapse):
    """
    AMPA synapse with parameters taken from Ermentrout et al. 2010, p. 161
    """
    def __init__(self, gsyn):
        super().__init__(gsyn, 65, AMPA())
        
class NeuronalNetwork:
    """
    Neuronal network class (biophysical or neuromorphic)
    Arguments:
        gapAdj : a gap junction adjacency matrix containing conductance values
        synAdj : a synapse adjacency matrix containing 1's and 0's
        syns : a matrix containing a list of synapse objects in each entry corresponding
            to a 1 in synAdj
    """
    def __init__(self,neurons,gapAdj=[],synAdj=[],syns=[]):
        self.neurons = neurons
        self.gapAdj = gapAdj
        self.synAdj = synAdj
        self.syns = syns

    def vfield(self, x, I):
        dx = []
        dx_syn = []
        
        idx_syn = len(self.neurons)*4 # synapse states start after neural states
        
        # Iterate through all neurons
        # Note: need to take into account number of states if not const 4
        for i, neuron_i in enumerate(self.neurons):
            i_syn = 0
            i_gap = 0
            
            Vpost = x[i*4]
            for j, _ in enumerate(self.neurons):
                Vpre = x[j*4]
                if (self.synAdj[i][j]):
                    for syn in self.syns[i][j]:
                        r = x[idx_syn] # activation of the synapse
                        i_syn += syn.Iout(r, Vpost)
                        dx_syn.append(syn.r.vfield(r, Vpre, Vpost))
                        idx_syn += 1
                
                if (self.gapAdj != []):
                    i_gap += self.gapAdj[i][j] * (Vpost - Vpre)
                
            Iext = I[i] - i_syn - i_gap
            dx.extend(neuron_i.vfield(x[4*i:4*(i+1)], Iext))
            
        dx.extend(dx_syn)
        return dx
        
    def simulate(self, trange, x0, Iapp, mode="continuous"):
        # Note: Iapp should be a function of t, e.g., Iapp = lambda t : I0
        if mode == "continuous":
            def odesys(t, x):
                return self.vfield(x, Iapp(t))
            return solve_ivp(odesys, trange, x0)
        else:
            #... code forward-Euler integration
            return
# class NeuroDynCascade(NeuronalNetwork):
#     def __init__(self):
#         neurons = [NeuroDynModel(),NeuroDynModel()]
#         gapAdj = []
#         synAdj = np.array([[0,1],[0,0]])
#         synList = [[[],[NeuroDynAMPA()]],[],[]]