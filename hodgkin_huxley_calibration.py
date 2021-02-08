# Add uncertainty to Hodgkin-Huxley parameters, try 'recalibrating' by
# adjusting the maximal conductance parameters to keep onset of spiking
# unperturbed
#%%
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cb_models import HHActivation, HHInactivation, HHModel

# Create nominal HH model
HH = HHModel()

# Offsets to perturb alpha/beta HH functions, generate randomly
mag = 10
Valpham = HH.m.aVh + mag*np.random.normal(0,1)
Vbetam = HH.m.bVh + mag*np.random.normal(0,1)
Valphah = HH.h.aVh + mag*np.random.normal(0,1)
Vbetah = HH.h.bVh + mag*np.random.normal(0,1)
Valphan = HH.n.aVh + mag*np.random.normal(0,1)
Vbetan = HH.n.bVh + mag*np.random.normal(0,1)

# Perturbed HH kinetics
m_P = HHActivation(Valpham, 0.1, 10, Vbetam, 4, 18)
h_P = HHInactivation(Valphah, 0.07, 20, Vbetah, 1, 10)
n_P = HHActivation(Valphan, 0.01, 10, Vbetan, 0.125, 80)

# Create perturbed HH model
HH_P = HHModel(gates=[m_P,h_P,n_P])

#%% Plot 'IV' curves
V = np.arange(-20,100,0.5)

# IV curves for nominal HH model
Ifast_HH = HH.iL_ss(V) + HH.iNa_ss(V)
Islow_HH = Ifast_HH + HH.iK_ss(V)
 
# # USE THIS TO ADJUST PERTURBED IV CURVES:
# # IV basis functions for Perturbed HH model
# Na_bf = m_P.inf(V)**3*h_P.inf(V)*(V - HH.Ena)
# K_bf = n_P.inf(V)**4*(V - HH.Ek)
# L_bf = (V - HH.El)

# IV curves for nominal HH model
Ifast_P = HH_P.iL_ss(V) + HH_P.iNa_ss(V)
Islow_P = Ifast_P + HH_P.iK_ss(V)

plt.figure()
plt.plot(V, Ifast_HH, V, Ifast_P)
plt.legend(['HH','perturbed HH'])
plt.figure()
plt.plot(V, Islow_HH, V, Islow_P)
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

def odesys(t, y, model):
    V, m, h, n = y
    I = I0
    #I = ramp(t)
    return model.dynamics(V, m, h, n, I)

trange = (0, T)

# Initial state y = [V0, m0, h0, n0], set at Vrest = 0
V0 = 0.001
y0 = [V0, HH.m.inf(V0), HH.h.inf(V0), HH.n.inf(V0)]

sol_HH = solve_ivp(lambda t,y : odesys(t,y,HH), trange, y0)
sol_P = solve_ivp(lambda t,y : odesys(t,y,HH_P), trange, y0)

# Plot the simulation
plt.figure()
plt.plot(sol_HH.t, sol_HH.y[0],sol_P.t, sol_P.y[0])
plt.legend(['HH','perturbed HH'])
plt.figure()
plt.plot(sol_HH.t, ramp(sol_HH.t))