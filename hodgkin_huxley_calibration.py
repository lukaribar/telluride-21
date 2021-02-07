# Add uncertainty to Hodgkin-Huxley parameters, try 'recalibrating' by
# adjusting the maximal conductance parameters to keep onset of spiking
# unperturbed
#%%
from abc import ABC, abstractmethod  # for abstract classes
import numpy as np
from numpy import exp

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

#%% Plot 'IV' curves
V = np.arange(-20,130,0.5)

# HH Nernst potentials and maximal conductances for potassium, sodium and leak
gk = 36
gna = 120
gl = 0.3
Ek = -12
Ena = 120
El = 10.6

# Nominal HH kinetics
m_HH = HHActivation(25, 0.1, 10, 0, 4, 18)
h_HH = HHInactivation(0, 0.07, 20, 30, 1, 10)
n_HH = HHActivation(10, 0.01, 10, 0, 0.125, 80)

# IV curves for nominal HH model
Ifast_HH = gl*(V - El) + gna*m_HH.inf(V)**3*h_HH.inf(V)*(V - Ena)
Islow_HH = Ifast_HH + gk*n_HH.inf(V)**4*(V - Ek)

# Offsets to perturb alpha/beta HH functions, generate randomly
mag = 10
Valpham = 25 + mag*np.random.normal(0,1)
Vbetam = 0 + mag*np.random.normal(0,1)
Valphah = 0 + mag*np.random.normal(0,1)
Vbetah = 30 + mag*np.random.normal(0,1)
Valphan = 10 + mag*np.random.normal(0,1)
Vbetan = 0 + mag*np.random.normal(0,1)

# Perturbed HH kinetics
m_P = HHActivation(Valpham, 0.1, 10, Vbetam, 4, 18)
h_P = HHInactivation(Valphah, 0.07, 20, Vbetah, 1, 10)
n_P = HHActivation(Valphan, 0.01, 10, Vbetan, 0.125, 80)

# IV basis functions for Perturbed HH model
Na_bf = m_P.inf(V)**3*h_P.inf(V)*(V - Ena)
K_bf = n_P.inf(V)**4*(V - Ek)
L_bf = (V - El)

# IV curves for nominal HH model
Ifast = gl*L_bf + gna*Na_bf
Islow = Ifast + gk*K_bf

plt.figure()
plt.plot(V, Ifast_HH, V, Ifast)
plt.legend(['HH','perturbed HH'])
plt.figure()
plt.plot(V, Islow_HH, V, Islow)
plt.legend(['HH','perturbed HH'])

#%% Simulation
# Define length of the simulation (in ms)
T = 500

# Constant current stimulus (in uA / cm^2)
I0 = 8

# Define the applied current as function of time
def ramp(t):
    # Ramp function from I1 to I2
    I1 = 0
    I2 = 15
    I = (t>=0)*I1 + (t/T)*(I2 - I1)
    return I

def odesys(t, y, gates):
    V, m, h, n = y
    
    I = I0
    #I = ramp(t)
    
    M = gates[0]
    H = gates[1]
    N = gates[2]

    dV = -gl*(V - El) - gna*m**3*h*(V - Ena) - gk*n**4*(V - Ek) + I
    dm = M.diff(V,m)
    dh = H.diff(V,h)
    dn = N.diff(V,n)
    return [dV, dm, dh, dn]

trange = (0, T)

# Initial state y = [V0, m0, h0, n0], set at Vrest = 0
V0 = 0.001
y0 = [V0, m_HH.inf(V0), h_HH.inf(V0), n_HH.inf(V0)]

gates_HH = [m_HH,h_HH,n_HH]
gates_P = [m_P,h_P,n_P]

sol_HH = solve_ivp(lambda t,y : odesys(t,y,gates_HH), trange, y0)
sol_P = solve_ivp(lambda t,y : odesys(t,y,gates_P), trange, y0)

# Plot the simulation
plt.figure()
plt.plot(sol_HH.t, sol_HH.y[0],sol_P.t, sol_P.y[0])
plt.legend(['HH','perturbed HH'])
plt.figure()
plt.plot(sol_HH.t, ramp(sol_HH.t))