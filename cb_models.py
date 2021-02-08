from abc import ABC, abstractmethod  # for abstract classes
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

    def diff(self,V,x):
        return self.alpha(V)*(1 - x) - self.beta(V)*x

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
#             dx.append(self.gates[n].diff(V,X[n]))
#         return dx
    
#     def I(self,V,X):
#         i_out = self.g_max * (V - self.E_rev)
#         for n in range(np.size(X))

class HHModel:
    """
        Hodgkin-Huxley model 
    """
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

    def dynamics(self, V, m, h, n, I):
        dV = -self.i_int(V, m, h, n) + I
        dm = self.m.diff(V,m)
        dh = self.h.diff(V,h)
        dn = self.n.diff(V,n)
        return [dV, dm, dh, dn]