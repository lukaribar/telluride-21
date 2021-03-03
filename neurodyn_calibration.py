#%%
from cb_models import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize, Bounds

# Fitting the HH activation functions
# This should eventually become part of the initialization of 
# Neurodyn activation / inactivation kinetics.

ND = NeuroDynModel()
HH = HHModel()
kappa,C,Vt,I_tau,I_ref,V_ref = ND.get_default_rate_pars()
kappa = 2

# Create gating variables to be fit
# Notice multiplications/divisions by 1e3 to convert from mV to V
m = HHActivation(25/1e3+V_ref, 0.1*1e3, 10/1e3, 0/1e3+V_ref, 4, 18/1e3)
h = HHInactivation(0/1e3+V_ref, 0.07, 20/1e3, 30/1e3+V_ref, 1, 10/1e3)
n = HHActivation(10/1e3+V_ref, 0.01*1e3, 10/1e3, 0/1e3+V_ref, 0.125, 80/1e3)
X = [m,h,n]

# Output of sum of sigmoids
def I_rate(Vrange,c,sign,kappa,Vhalf):
    I=0
    for i in range(len(Vhalf)):
        I += c[i] / (1 + np.exp(sign * kappa * (Vhalf[i] - Vrange)  / Vt))
    return I

# Cost function
def cost(Z,X,Vrange,kappa,Vt):
    """
    inputs:
        C_a is list of vector of coefficients to fit alpha functions
        C_b is list of vector of coefficients to fit beta functions
        ...
        X is a list of HHKinetics objects to fit
        Vrange is a vector with voltage values used for fitting
    output:
        value of the cost
    """
    Vmean = Z[-2]
    Vstep = Z[-1]
    Vhalf = Vmean + np.arange(start=-3,stop=4,step=1)*Vstep

    out = 0
    for i, x in enumerate(X):
        c_a = Z[i*7:(i+1)*7]
        c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
        norm_a = max(x.alpha(Vrange))
        norm_b = max(x.beta(Vrange))        
        if isinstance(x,HHActivation):    
            out += sum(((x.alpha(Vrange) - I_rate(Vrange,c_a,1,kappa,Vhalf))/norm_a)**2) 
            out += sum(((x.beta(Vrange) - I_rate(Vrange,c_b,-1,kappa,Vhalf))/norm_b)**2) 
        else:
            out += sum(((x.alpha(Vrange) - I_rate(Vrange,c_a,-1,kappa,Vhalf))/norm_a)**2) 
            out += sum(((x.beta(Vrange) - I_rate(Vrange,c_b,1,kappa,Vhalf))/norm_b)**2)  

    return out  

# Range to do the fit
Vstart = -0.02  # V_ref+HH.Ek/1e3
Vend   = 0.08   # V_ref+HH.Ena/1e3
Vrange = np.arange(start=Vstart, stop=Vend, step=5e-4).T

# Initial parameter values
C_a = np.array([])
C_b = np.array([])
for i, x in enumerate(X):
    C_a = np.append(C_a,max(x.alpha(Vrange))*np.ones(7)/7)
    C_b = np.append(C_b,max(x.beta(Vrange))*np.ones(7)/7)
Vmean = V_ref   #(V_ref+HH.Ek/1e3 + V_ref+HH.Ena/1e3)/2
Vstep = (V_ref+HH.Ena/1e3 - V_ref+HH.Ek/1e3)/100
Z0 = np.concatenate([C_a,C_b,np.array([Vmean,Vstep])])

lowerbd = np.append(np.zeros(14*len(X)),np.array([-np.inf,-np.inf]))
upperbd = np.append(np.ones(14*len(X))*np.inf,np.array([np.inf,np.inf]))
bd = Bounds(lowerbd,upperbd)

Z = minimize(lambda Z : cost(Z,X,Vrange,kappa,Vt), Z0, bounds = bd)
Z = Z.x
#%%

Vmean = Z[-2]
Vstep = Z[-1]
Vhalf = Vmean + np.arange(start=-3,stop=4,step=1)*Vstep

for i,x in enumerate(X):
    c_a = Z[i*7:(i+1)*7]
    c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
    if isinstance(x,HHActivation):
        alpha = I_rate(Vrange,c_a,1,kappa,Vhalf)
        beta = I_rate(Vrange,c_b,-1,kappa,Vhalf)
    else:
        alpha = I_rate(Vrange,c_a,-1,kappa,Vhalf)
        beta = I_rate(Vrange,c_b,1,kappa,Vhalf)

    plt.figure()
    plt.plot(Vrange,x.alpha(Vrange))
    plt.plot(Vrange,alpha)

    plt.figure()
    plt.plot(Vrange,x.beta(Vrange))
    plt.plot(Vrange,beta)

print("Vstep:", Vstep)
print("Vmean:", Vmean)

# plt.figure()
# plt.plot(Vrange,b_alpha)
# plt.plot(V,np.dot(A_alpha,Ib_alpha))
# plt.plot(V,A_alpha,'black')

# plt.figure()
# plt.plot(V,b_beta)
# plt.plot(V,np.dot(A_beta,Ib_beta))
# plt.plot(V,A_beta,'black')

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