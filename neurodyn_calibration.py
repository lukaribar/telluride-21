#%%
from cb_models import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize, Bounds

ND = NeuroDynModel()
kappa,C,C_ND,Vt,I_tau,I_ref,V_ref = ND.get_default_rate_pars()
kappa = 0.7

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 3
# Create a HH that is scaled in voltage
HH = HHModel(scl=scl_v*1e-3)
X = [HH.m,HH.h,HH.n]

plots = False

#%% Finding optimal Vstep and Vmean and initial parameters for coefficients

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
        Z contains the list of free variables
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
Vstart = V_ref + HH.Ek
Vend   = 0.08*scl_v # V_ref + HH.Ena # 0.08
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

Vmean = Z[-2]
Vstep = Z[-1]
Vhalf = Vmean + np.arange(start=-3,stop=4,step=1)*Vstep

print("Vstep:", Vstep)
print("Vmean:", Vmean)

#Plot the nonlinear fitting results
if (plots):
    for i,x in enumerate(X):
        c_a = Z[i*7:(i+1)*7]
        c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
        if isinstance(x,HHActivation):
            alpha = I_rate(Vrange,c_a,1,kappa,Vhalf)
            beta = I_rate(Vrange,c_b,-1,kappa,Vhalf)
        else:
            alpha = I_rate(Vrange,c_a,-1,kappa,Vhalf)
            beta = I_rate(Vrange,c_b,1,kappa,Vhalf)
    
        gatelabels = ['m','h','n']
        plt.figure()
        plt.plot(Vrange,x.alpha(Vrange),label='HH α_'+gatelabels[i])
        plt.plot(Vrange,alpha,label='fit α_'+gatelabels[i])
        plt.legend()
    
        plt.figure()
        plt.plot(Vrange,x.beta(Vrange),label='HH β_'+gatelabels[i])
        plt.plot(Vrange,beta,label='fit β_'+gatelabels[i])
        plt.legend()

#%% Now adjust each I_alpha and I_beta individually

# IMPORTANT: c_a and c_b returned by this function ignores the factor of 
# 1000 due to HH's time units, which are in miliseconds
def lsqfit(x,Vrange,Vhalf,kappa,Vt):
    A_alpha = np.zeros((np.size(Vrange),7))
    A_beta = np.zeros((np.size(Vrange),7))
    b_alpha = x.alpha(Vrange)
    b_beta = x.beta(Vrange)
    for i in range(7):
        if isinstance(x,HHActivation):
            A_alpha[:,i] = 1 / (1 + np.exp(1 * kappa * (Vhalf[i] - Vrange)  / Vt))
            A_beta[:,i] = 1 / (1 + np.exp(-1 * kappa * (Vhalf[i] - Vrange)  / Vt))
        else:
            A_alpha[:,i] = 1 / (1 + np.exp(-1 * kappa * (Vhalf[i] - Vrange)  / Vt))
            A_beta[:,i] = 1 / (1 + np.exp(1 * kappa * (Vhalf[i] - Vrange)  / Vt))
    c_a = nnls(A_alpha,b_alpha)[0]
    c_b = nnls(A_beta,b_beta)[0]

    return c_a,c_b,A_alpha,A_beta

# Find maximal sigmoidal basis function coefficients
cmax = 0

# Find optimal sigmoidal basis functions coefficients and plot the fits
c = []
A = []
for i,x in enumerate(X):
    c_a,c_b,A_alpha,A_beta = lsqfit(x,Vrange,Vhalf,kappa,Vt)
    for j,coeff in enumerate(c_a):
        if coeff > cmax:
            cmax = coeff 
    for j,coeff in enumerate(c_b):
        if coeff > cmax:
            cmax = coeff 
    c.append([c_a, c_b])
    A.append([A_alpha, A_beta])

    alpha = np.dot(A[i][0],c[i][0])
    beta = np.dot(A[i][1],c[i][1])
    tau = 1/(alpha+beta)
    inf = alpha/(alpha+beta)
    
    if (plots):
        gatelabels = ['m','h','n']
        plt.figure()
        plt.plot(Vrange,x.alpha(Vrange),label='HH α_'+gatelabels[i])
        plt.plot(Vrange,alpha,label='fit α_'+gatelabels[i])
        plt.legend()
    
        plt.figure()
        plt.plot(Vrange,x.beta(Vrange),label='HH β_'+gatelabels[i])
        plt.plot(Vrange,beta,label='fit β_'+gatelabels[i])
        plt.legend()
    
        plt.figure()
        plt.plot(Vrange,x.tau(Vrange),label='HH τ_'+gatelabels[i])
        plt.plot(Vrange,tau,label='fit τ_'+gatelabels[i])
        plt.legend()
    
        plt.figure()
        plt.plot(Vrange,x.inf(Vrange),label='HH '+gatelabels[i]+'_∞')
        plt.plot(Vrange,inf,label='fit '+gatelabels[i]+'_∞')
        plt.legend()

# Time scaling (important: assumes that HH is already written in SI units)
# Factor the time scaling into a factor that sets the ND capacitance (C_ND/C_HH)
# and a factor (s) that ensures the maximum Ib is given by 1023*I_tau/1024
C_HH = 1e-6
s = cmax * C * Vt / (1023*I_tau/1024) * 1000 * C_HH / C_ND
# s = 1/4*1e6             # Note: this value makes ND timescale equal to HH timescale
scl_t = s*C_ND/C_HH     

# Recover the (quantized) rate currents from fitted coefficients
Ib = []
dIb = []
for i,x in enumerate(X):
    # Exact (real numbers) current coefficients, before quantization
    i_a = c[i][0] * C * Vt * 1000 / scl_t 
    i_b = c[i][1] * C * Vt * 1000 / scl_t
    Ib.append([i_a, i_b])
    # Quantize current coefficients
    di_a = np.round(i_a*1024/I_tau)
    di_b = np.round(i_b*1024/I_tau)
    dIb.append([di_a, di_b])

Ib = np.array(Ib)*1024/I_tau
dIb = np.array(dIb)

#%%
ND = NeuroDynModel(np.array([120,36,0.3])*1e-3/s,np.array([120,-12,10.6])*1e-3*scl_v, dIb, Vmean+3.5*Vstep, Vmean-3.5*Vstep)

I0 = (10*1e-6)*scl_v/s
Iapp = lambda t : I0
def Ibump(t):
    if t < 0.004:
        return I0
    else:
        return I0 + 1e-3*t**2*np.exp(-(t-0.004)/1e-5)

T = 0.01
trange = (0, T)

# Simulate different perturbed instances of the neuron
#np.random.seed(0)

fig1 = plt.figure(1)
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')

#fig2, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3,2)
#fig2.suptitle('Gating variables')

#fig3, ([ax7, ax8]) = plt.subplots(2)
#fig3.suptitle('IV curves')

V1 = np.arange(-1,1,0.01)
V2 = np.arange(-0.25,0.25,0.01)

for i in range(1):
    sol = ND.simulate(trange,[0,0,0,0],Iapp)
    
    # Time plot
    plt.figure(1)
    plt.plot(sol.t, sol.y[0])
#    
#    # Gating variable steady-state functoins
#    ax1.plot(V1, ND.m.inf(V1))
#    ax1.set_title(r'$m_{\infty}(V)$')
#    ax2.plot(V1, ND.m.tau(V1))
#    ax2.set_title(r'$\tau_{m}(V)$')
#    ax3.plot(V1, ND.h.inf(V1))
#    ax3.set_title(r'$h_{\infty}(V)$')
#    ax4.plot(V1, ND.h.tau(V1))
#    ax4.set_title(r'$\tau_{h}(V)$')
#    ax5.plot(V1, ND.n.inf(V1))
#    ax5.set_title(r'$n_{\infty}(V)$')
#    ax5.set_xlabel('V')
#    ax6.plot(V1, ND.n.tau(V1))
#    ax6.set_title(r'$\tau_{n}(V)$')
#    ax6.set_xlabel('V')
#    
#    # IV curves
#    Ifast = ND.iL_ss(V2) + ND.iNa_ss(V2)
#    Islow = Ifast + ND.iK_ss(V2)
#    ax7.plot(V2, Ifast)
#    ax7.set_title(r'$I_{fast}$')
#    ax8.plot(V2, Islow)
#    ax8.set_title(r'$I_{slow}$')
#    ax8.set_xlabel('V')
#    
#    ND.perturb()

plt.show()