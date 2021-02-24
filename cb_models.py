from abc import ABC, abstractmethod  # for abstract classes
from scipy.integrate import solve_ivp
import numpy as np
from numpy import exp

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
    def __init__(self,dIb,kappa,Vt,Vb,I_tau,sign):
        self.dIb = dIb
        self.kappa = kappa  #global?
        self.Vt = Vt  #global?
        self.Vb = Vb  #global?
        self.I_tau = np.array([I_tau]*7)
        self.sign = sign

    def I_rate(self,V):
        I=0
        Ib = self.dIb*self.I_tau/1024
        for i in range(np.size(Ib)):
            I += Ib[i] / (1 + np.exp(self.sign * self.kappa * (self.Vb[i] - V)  / self.Vt))
        return I     

class NeuroDynActivation(HHKinetics):
    """
    NeuroDyn-type activation gating variable kinetics.
    """
    def __init__(self,dIb,kappa,C,Vt,Vb,I_tau):
        self.C = C
        self.Vt = Vt
        self.alpharate = NeuroDynRate(dIb[0],kappa,Vt,Vb,I_tau,1)
        self.betarate = NeuroDynRate(dIb[1],kappa,Vt,Vb,I_tau,-1) 
    
    def alpha(self,V):
        return self.alpharate.I_rate(V) / (self.C * self.Vt)

    def beta(self,V):
        return self.betarate.I_rate(V) / (self.C * self.Vt)
    
class NeuroDynInactivation(HHKinetics):
    """
    NeuroDyn-type activation gating variable kinetics.
    """
    def __init__(self,dIb,kappa,C,Vt,Vb,I_tau):
        self.C = C
        self.Vt = Vt
        self.alpharate = NeuroDynRate(dIb[0],kappa,Vt,Vb,I_tau,-1) 
        self.betarate = NeuroDynRate(dIb[1],kappa,Vt,Vb,I_tau,1) 
    
    def alpha(self,V):
        return self.alpharate.I_rate(V) / (self.C * self.Vt)

    def beta(self,V):
        return self.betarate.I_rate(V) / (self.C * self.Vt)

class HHActivation(HHKinetics):
    """
    HH-type (alpha-beta) activation gating variable kinetics.
    """
    def __init__(self, aVh, aA, aK, bVh, bA, bK):
        self.aVh = aVh
        self.aA = aA
        self.aK = aK
        self.bVh = bVh
        self.bA = bA
        self.bK = bK

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

class HHInactivation(HHKinetics):
    """
    HH-type (alpha-beta) inactivation gating variable kinetics. 
    """
    def __init__(self, aVh, aA, aK, bVh, bA, bK):
        self.aVh = aVh
        self.aA = aA
        self.aK = aK
        self.bVh = bVh
        self.bA = bA
        self.bK = bK

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

class NeuroDynModel:
    """
    NeuroDyn model
    """
    
    def __init__(self, dg=[400, 160, 12], dErev=[450, -250, -150], gates=[]):
        self.V_ref = 0              # Unit V , 1 volt
        self.I_tau = 33e-9          # Unit A
        self.I_voltage = 230e-9     # Unit A
        self.I_ref = 15e-9          # Unit A
        
        # Membrane & gate capacitances
        self.C_m = 4e-12 # Unit F
        self.C_gate = 5e-12 # Unit F
        
        # Scaling parameters (e.g. parameters that set the voltage scale, time scale..)
        self.kappa = 0.7
        self.Vt = 26e-3 # Unit Volt
        self.Res = 1.63e6 # Unit Ohm
        
        # Digital parameters
        self.dg = dg
        self.dErev = dErev
                
        # Convert digital to physical
        self.gna = self.convert_conductance(dg[0])
        self.gk = self.convert_conductance(dg[1])
        self.gl = self.convert_conductance(dg[2])
        self.Ena = self.convert_potential(dErev[0])
        self.Ek = self.convert_potential(dErev[1])
        self.El = self.convert_potential(dErev[2])
        
        # Gating variable coefficients
        self.p = 3
        self.q = 1
        self.r = 4
        
        if not gates:
            # Default to nominal NeuroDyn activation parameters
            Vb = self.get_default_Vb()
            dIb_m = [[0, 0, 120, 400, 800, 1023, 1023],
                     [1023, 1023, 1023, 1023, 0, 0, 0]]
            dIb_h = [[237, 5, 7, 6, 0, 0, 0],
                    [0, 0, 0, 0, 41, 25, 8]]
            dIb_n = [[0, 0, 0, 0, 80, 40, 250],
                    [4, 0, 0, 10, 0, 0, 4]]
            self.m = NeuroDynActivation(dIb_m,self.kappa,self.C_gate,self.Vt,Vb,self.I_tau)
            self.h = NeuroDynInactivation(dIb_h,self.kappa,self.C_gate,self.Vt,Vb,self.I_tau)
            self.n = NeuroDynActivation(dIb_n,self.kappa,self.C_gate,self.Vt,Vb,self.I_tau)
        else:
            self.m = gates[0]
            self.h = gates[1]
            self.n = gates[2]
            
    def convert_conductance(self, dg):
        # Factor for converting digital to physical g
        g_factor = (self.kappa / self.Vt) * (self.I_tau / 1024)
        return dg * g_factor
        
        
    def convert_potential(self, dErev):
        # Factor for converting digital to physical Erev
        E_factor = (self.I_voltage / 1024) * self.Res
        return dErev * E_factor + self.V_ref
    
    def get_default_Vb(self):
         # Bias voltages for the 7-point spline regression
        Vb = np.zeros(7) # Define the 7 bias voltages
        vHigh = self.V_ref + 0.426
        vLow = self.V_ref - 0.434
        I_factor = (vHigh - vLow) / 700e3
        Vb[0] = vLow + (I_factor * 50e3)
        for i in range(1, 7):
            Vb[i] = Vb[i-1] + (I_factor * 100e3)
            
        return Vb
    
    def i_int(self,V, m, h, n):
        return self.gna*(m**self.p)*(h**self.q)*(V - self.Ena) + self.gk*(n**self.r)*(V - self.Ek) + self.gl*(V - self.El)

    def iNa_ss(self,V):
        return self.gna*self.m.inf(V)**3*self.h.inf(V)*(V - self.Ena)

    def iK_ss(self,V):
        return self.gk*self.n.inf(V)**4*(V - self.Ek)    

    def iL_ss(self,V):
        return self.gl*(V - self.El)
    
    def vfield(self, x, I):
        V, m, h, n = x
        dV = (-self.i_int(V, m, h, n) + I)/self.C_m
        dm = self.m.vfield(m,V)
        dh = self.h.vfield(h,V)
        dn = self.n.vfield(n,V)
        return [dV, dm, dh, dn]
    
    def simulate(self, trange, x0, Iapp, mode="continuous"):
        # Note: Iapp should be a function of t, e.g., Iapp = lambda t : I0
        if mode == "continuous":
            def odesys(t, x):
                return self.vfield(x, Iapp(t))
            return solve_ivp(odesys, trange, x0)
        else:
            #... code forward-Euler integration
            return

    def perturb(self,sigma=0.15):
        
        # Pertrub exponents
        self.p = 3 + 0.2*np.random.randn()
        self.q = 1 + 0.1*np.random.randn()
        self.r = 4 + 0.2*np.random.randn()
        
        # For each alpha/beta, perturb Itaus
        for x in [self.m, self.h, self.n]:
            x.alpharate.I_tau = self.I_tau * (1 + sigma*np.random.randn(7))
            x.betarate.I_tau = self.I_tau * (1 + sigma*np.random.randn(7))
            
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

class HHModel:
    """
        Hodgkin-Huxley model 
    """
    
#    class Ina:
#        def __init__(self, gna, Ena):
#        self.gna = gna
#        self.Ena = Ena
#        self.p = 3
#        self.q = 1
#        
#        def out(V, m, h):
#            return self.gna*(m**p)*(h**q)*(V - self.Ena)
#        
#    class Ik:
#        def __init__(self, gk, Ek):
#            self.gk = gk
#            self.Ek = Ek
#            self.p = 4
#            
#        def out(V, n):
#             return self.gk*(n**p)*(V - self.Ek)
#         
#    class Il:
#        def __init__(self. gl, El):
#            self.gl = gl
#            self.El = El
#        
#        def out(V):
#            return self.gl*(V - self.El)
    
    # Default to nominal HH Nernst potentials and maximal conductances
    def __init__(self, gna = 120, gk = 36, gl = 0.3, Ena = 120, Ek = -12, El = 10.6, gates=[]):
        self.gna = gna
        self.gk = gk
        self.gl = gl
        self.Ena = Ena
        self.Ek = Ek
        self.El = El        
        if not gates:
            # Default to nominal HH kinetics
            self.m = HHActivation(25, 0.1, 10, 0, 4, 18)
            self.h = HHInactivation(0, 0.07, 20, 30, 1, 10)
            self.n = HHActivation(10, 0.01, 10, 0, 0.125, 80)
        else:
            self.m = gates[0]
            self.h = gates[1]
            self.n = gates[2]

    def i_int(self,V, m, h, n):
        return self.gna*m**3*h*(V - self.Ena) + self.gk*n**4*(V - self.Ek) + self.gl*(V - self.El)

    def iNa_ss(self,V):
        return self.gna*self.m.inf(V)**3*self.h.inf(V)*(V - self.Ena)

    def iK_ss(self,V):
        return self.gk*self.n.inf(V)**4*(V - self.Ek)    

    def iL_ss(self,V):
        return self.gl*(V - self.El)

    def vfield(self, x, I):
        V, m, h, n = x
        dV = -self.i_int(V, m, h, n) + I
        dm = self.m.vfield(m,V)
        dh = self.h.vfield(h,V)
        dn = self.n.vfield(n,V)
        return [dV, dm, dh, dn]

##### NETWORK-RELATED CLASSES #####

class NeuroDynAMPA(NeuroDynActivation):
    """
    AMPA Synapse in the neurodyn chip.
    Physiological values taken from Ermentrout et al. 2010, p. 161
    """
    def __init__(self,gsyn=1,Esyn=0,kappa=0.7,C=5e-12,Vt=26e-3,I_tau=33e-9):
        self.gsyn = gsyn
        self.Esyn = Esyn
        # Physiological constants
        Tmax, ar, ad, Kp, V_T = 0.001, 1.1, 0.19, 0.005, 0.002
        dIb = [[Tmax*ar*C*Vt, 0, 0, 0, 0, 0, 0],
               [0, Tmax*ad*C*Vt, 0, 0, 0, 0, 0]]
        Vb = [V_T,-10,0,0,0,0,0] 
        # IMPORTANT: WE CAN'T REALLY USE THE SIGMOIDS THIS WAY.
        # WE NEED TO FIT THE 7 SIGMOIDS TO THE AMPA SIGMOID
        super().__init__(dIb,kappa,C,Kp*kappa,Vb,I_tau)  

class AMPA(HHKinetics):
    """
    AMPA gating variable kinetics  
    Physiological values taken from Ermentrout et al. 2010, p. 161
    """
    def __init__(self,Tmax=1,Kp=5,V_T=2,ar=1.1,ad=0.19):
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
        super().__init__(gsyn, 0, AMPA())
        
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
            for j, neuron_j in enumerate(self.neurons):
                Vpre = x[j*4]
                if (self.synAdj[i][j]):
                    for syn in self.syns[i][j]:
                        r = x[idx_syn] # activation of the synapse
                        i_syn += syn.Iout(r, Vpost)
                        dx_syn.append(syn.r.vfield(r, Vpre, Vpost))
                        idx_syn += 1
                
                if (self.gapAdj != []):
                    i_gap += self.gapAdj[i][j] * (Vpre - Vpost)
                
            Iext = I[i] + i_syn + i_gap
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