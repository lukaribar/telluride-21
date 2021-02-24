#%%
from cb_models import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

# Fitting the HH activation functions
# This should eventually become part of the initialization of 
# Neurodyn activation / inactivation kinetics.

ND = NeuroDynModel()
HH = HHModel()
kappa,C,Vt,I_tau,I_ref,V_ref = ND.get_default_rate_pars()

# Notice multiplications/divisions by 1e3 to convert from mV to V
m = HHActivation(25/1e3+V_ref, 0.1*1e3, 10/1e3, 0+V_ref, 4, 18/1e3)

# Maybe we could automatize this by adding vHigh and vLow
# (or their average) as a free parameter to be optimized for. 
Vb = np.zeros(7)
vHigh = V_ref  + HH.Ena/1e3  + 0.1
vLow = V_ref   + HH.Ek/1e3   - 0.1 
I_factor = (vHigh - vLow) / 700e3
Vb[0] = vLow + (I_factor * 50e3)
for i in range(1, 7):
    Vb[i] = Vb[i-1] + (I_factor * 100e3)

V = np.arange(start=V_ref+HH.Ek/1e3, stop=V_ref+HH.Ena/1e3, step=5e-4).T

A_alpha = np.zeros((np.size(V),7))
A_beta = np.zeros((np.size(V),7))
b_alpha = m.alpha(V) / np.amax(m.alpha(V)) 
b_beta = m.beta(V) / np.amax(m.beta(V))
for i in range(7):
    A_alpha[:,i] = 1 / (1 + np.exp(1 * kappa * (Vb[i] - V)  / Vt))
    A_beta[:,i] = 1 / (1 + np.exp(-1 * kappa * (Vb[i] - V)  / Vt))
Ib_alpha = nnls(A_alpha,b_alpha)[0]
Ib_beta = nnls(A_beta,b_beta)[0]

plt.figure()
plt.plot(V,b_alpha)
plt.plot(V,np.dot(A_alpha,Ib_alpha))
plt.plot(V,A_alpha,'black')

plt.figure()
plt.plot(V,b_beta)
plt.plot(V,np.dot(A_beta,Ib_beta))
plt.plot(V,A_beta,'black')

#%%

ND = NeuroDynModel()

I0 = 0
Iapp = lambda t : I0
def Ibump(t):
    if t < 0.004:
        return I0
    else:
        return I0 + 1e-3*t**2*np.exp(-(t-0.004)/1e-5)

T = 0.02
trange = (0, T)

# Simulate different perturbed instances of the neuron
#np.random.seed(0)

fig1 = plt.figure(1)
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')

fig2, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3,2)
fig2.suptitle('Gating variables')

fig3, ([ax7, ax8]) = plt.subplots(2)
fig3.suptitle('IV curves')

V1 = np.arange(-1,1,0.01)
V2 = np.arange(-0.25,0.25,0.01)

for i in range(1):
    sol = ND.simulate(trange,[-0.175,0,0,0],Iapp)
    
    # Time plot
    plt.figure(1)
    plt.plot(sol.t, sol.y[0])
    
    # Gating variable steady-state functoins
    ax1.plot(V1, ND.m.inf(V1))
    ax1.set_title(r'$m_{\infty}(V)$')
    ax2.plot(V1, ND.m.tau(V1))
    ax2.set_title(r'$\tau_{m}(V)$')
    ax3.plot(V1, ND.h.inf(V1))
    ax3.set_title(r'$h_{\infty}(V)$')
    ax4.plot(V1, ND.h.tau(V1))
    ax4.set_title(r'$\tau_{h}(V)$')
    ax5.plot(V1, ND.n.inf(V1))
    ax5.set_title(r'$n_{\infty}(V)$')
    ax5.set_xlabel('V')
    ax6.plot(V1, ND.n.tau(V1))
    ax6.set_title(r'$\tau_{n}(V)$')
    ax6.set_xlabel('V')
    
    # IV curves
    Ifast = ND.iL_ss(V2) + ND.iNa_ss(V2)
    Islow = Ifast + ND.iK_ss(V2)
    ax7.plot(V2, Ifast)
    ax7.set_title(r'$I_{fast}$')
    ax8.plot(V2, Islow)
    ax8.set_title(r'$I_{slow}$')
    ax8.set_xlabel('V')
    
    ND.perturb()

plt.show()