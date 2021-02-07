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