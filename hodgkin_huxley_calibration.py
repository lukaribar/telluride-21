# Add uncertainty to Hodgkin-Huxley parameters, try 'recalibrating' by
# adjusting the maximal conductance parameters to keep onset of spiking
# unperturbed

import numpy as np
from numpy import exp

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Offsets to alpha/beta functions, generate randomly
mag = 10
dValpham = mag*np.random.normal(0,1)
dVbetam = mag*np.random.normal(0,1)
dValphah = mag*np.random.normal(0,1)
dVbetah = mag*np.random.normal(0,1)
dValphan = mag*np.random.normal(0,1)
dVbetan = mag*np.random.normal(0,1)

def alpha_m(V):
    V = V + dValpham # add offset due to uncertainty
    return 0.1 * (25 - V) / (exp((25 - V) / 10) - 1)

def beta_m(V):
    V = V + dVbetam # add offset due to uncertainty
    return 4 * exp(-V / 18)

def alpha_h(V):
    V = V + dValphah # add offset due to uncertainty
    return 0.07 * exp(-V / 20)

def beta_h(V):
    V = V + dVbetah # add offset due to uncertainty
    return 1 / (exp((30 - V) / 10) + 1)

def alpha_n(V):
    V = V + dValphan # add offset due to uncertainty
    return 0.01 * (10 - V) / (exp((10 - V)/10) - 1)

def beta_n(V):
    V = V + dVbetan # add offset due to uncertainty
    return 0.125 * exp(-V / 80)

def x_inf(V, alpha, beta):
    return alpha(V) / (alpha(V) + beta(V))

def tau(V, alpha, beta):
    return 1 / (alpha(V) + beta(V))

# Nernst potentials and maximal conductances for potassium, sodium and leak
gk = 36
gna = 120
gl = 0.3

Ek = -12
Ena = 120
El = 10.6

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

# Define x_inf and find their gradient functions
m_inf = lambda V: x_inf(V, alpha_m, beta_m)
h_inf = lambda V: x_inf(V, alpha_h, beta_h)
n_inf = lambda V: x_inf(V, alpha_n, beta_n)

# Define tau functions
tau_m = lambda V: tau(V, alpha_m, beta_m)
tau_h = lambda V: tau(V, alpha_h, beta_h)
tau_n = lambda V: tau(V, alpha_n, beta_n)

def odesys(t, y):
    V, m, h, n = y
    
    I = I0
    #I = ramp(t)
    
    dV = -gl*(V - El) - gna*m**3*h*(V - Ena) - gk*n**4*(V - Ek) + I
    dm = alpha_m(V)*(1 - m) - beta_m(V)*m
    dh = alpha_h(V)*(1 - h) - beta_h(V)*h
    dn = alpha_n(V)*(1 - n) - beta_n(V)*n
    return [dV, dm, dh, dn]

trange = (0, T)

# Initial state y = [V0, m0, h0, n0], set at Vrest = 0
V0 = 0.001
y0 = [V0, m_inf(V0), h_inf(V0), n_inf(V0)]

sol = solve_ivp(odesys, trange, y0)

# Plot the simulation
plt.figure()
plt.plot(sol.t, sol.y[0])
plt.figure()
plt.plot(sol.t, ramp(sol.t))

# Plot 'IV' curves
# Note: Division by 0 in alpha_m(V) and alpha_n(V)
V = np.arange(-20,130,0.5)
Ifast = gl*(V - El) + gna*m_inf(V)**3*h_inf(V)*(V - Ena)
Islow = Ifast + gk*n_inf(V)**4*(V - Ek)

plt.figure()
plt.plot(V, Ifast)
plt.figure()
plt.plot(V, Islow)