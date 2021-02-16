# Add uncertainty to Hodgkin-Huxley parameters, try 'recalibrating' by
# adjusting the maximal conductance parameters to keep onset of spiking
# unperturbed

#%%
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cb_models import HHActivation, HHInactivation, HHModel

# Create nominal HH model
HH = HHModel()

# Offsets to perturb alpha/beta HH functions, generate randomly
mag = 10
# np.random.seed(5)

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
HH_C = copy.deepcopy(HH_P) 

#%% Plot 'IV' curves
V = np.arange(-20,100,0.5)
v_start = -10
v_end = 30
ind_start = (v_start-(-20))*2
ind_end = (v_end-(-20))*2

# IV curves for nominal HH model
Ifast_HH = HH.iL_ss(V) + HH.iNa_ss(V)
Islow_HH = Ifast_HH + HH.iK_ss(V)

# IV curves for perturbed HH model
Ifast_P = HH_P.iL_ss(V) + HH_P.iNa_ss(V)
Islow_P = Ifast_P + HH_P.iK_ss(V)
 
# USE THIS TO ADJUST PERTURBED IV CURVES:
# IV basis functions for Perturbed HH model
Na_bf = m_P.inf(V)**3*h_P.inf(V)*(V - HH.Ena)
K_bf = n_P.inf(V)**4*(V - HH.Ek)
L_bf = (V - HH.El)

# Fit fast IV curve by least squares in a range close to threshold
A = np.array([Na_bf[ind_start:ind_end], L_bf[ind_start:ind_end]]).T
Apinv = np.linalg.pinv(A)
sol = Apinv.dot(Ifast_HH[ind_start:ind_end])

# fast IV curve for corrected HH model
HH_C.gna = sol[0]
HH_C.gl = sol[1]
Ifast_C = HH_C.iL_ss(V) + HH_C.iNa_ss(V)

# Fit slow IV curve
V = np.arange(-20,100,0.5)
v_start = -20
v_end = 100
ind_start = (v_start-(-20))*2
ind_end = (v_end-(-20))*2
A = np.array([K_bf[ind_start:ind_end]]).T
Apinv = np.linalg.pinv(A)
sol = Apinv.dot(Islow_HH[ind_start:ind_end]-Ifast_C[ind_start:ind_end])

HH_C.gk = sol[0]
Islow_C = Ifast_C + HH_C.iK_ss(V)

# Find local maximum of the fast nominal fast IV curve
Ifast_HH_grad = np.gradient(Ifast_HH) / vstep
th_index = (np.diff(np.sign(Ifast_HH_grad)) < 0).nonzero()[0][0]
Vth = V[th_index]
print(Vth)

# Adjust gna in the calibrated model to keep Vth const
Ileak_P_grad = np.gradient(HH_P.iL_ss(V))[th_index] / vstep
Ina_P_grad = np.gradient(HH_P.iNa_ss(V))[th_index] / vstep
HH_C.gna = HH_C.gna * (-Ileak_P_grad/Ina_P_grad)

# Calibrated fast IV curve
Ifast_C = HH_C.iL_ss(V) + HH_C.iNa_ss(V)
 
# Adjust gk in the calibrated model to keep Islow slope around Vth const
Islow_grad = np.gradient(Islow_HH) / vstep
desired_slope = Islow_grad[th_index]
Ifast_C_grad = np.gradient(Ifast_C) / vstep
desired_slope_k = desired_slope - Ifast_C_grad[th_index]
k_C_grad = np.gradient(HH_P.iK_ss(V))[th_index] / vstep
HH_C.gk = HH_C.gk * (desired_slope_k / k_C_grad)

# Calibrated slow IV curve
Islow_C = Ifast_C + HH_C.iK_ss(V)

plt.figure()
plt.plot(V, Ifast_HH, V, Ifast_P, V, Ifast_C)
plt.legend(['HH','perturbed HH','corrected perturbed HH'])
plt.figure()
plt.plot(V, Islow_HH, V, Islow_P, V, Islow_C)
plt.legend(['HH','perturbed HH','corrected perturbed HH'])

# Simulation
# Define length of the simulation (in ms)
T = 500

# Constant current stimulus (in uA / cm^2)
I0 = 20

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
sol_HH_P = solve_ivp(lambda t,y : odesys(t,y,HH_P), trange, y0)
sol_HH_C = solve_ivp(lambda t,y : odesys(t,y,HH_C), trange, y0)

# Plot the simulation
plt.figure()
plt.plot(sol_HH.t, sol_HH.y[0])
plt.legend(['HH'])
plt.figure()
plt.plot(sol_HH_P.t, sol_HH_P.y[0],'orange')
plt.legend(['perturbed HH'])
plt.figure()
plt.plot(sol_HH_C.t, sol_HH_C.y[0],'green')
plt.legend(['corrected perturbed HH'])
# plt.figure()
# plt.plot(sol_HH.t, ramp(sol_HH.t))
