
from cb_models import NeuroDynModel
import numpy as np
import matplotlib.pyplot as plt

ND = NeuroDynModel()

I0 = 0.0
Iapp = lambda t : I0

T = 0.01
trange = (0, T)

# Simulate different perturbed instances of the neuron
np.random.seed(0)

fig1 = plt.figure(1)
plt.xlabel('t')
plt.title('NeuroDyn simulation')

fig2, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3,2)
fig2.suptitle('Gating variables')

fig3, ([ax7, ax8]) = plt.subplots(2)
fig3.suptitle('IV curves')

V1 = np.arange(-1,1,0.01)
V2 = np.arange(-0.25,0.25,0.01)

for i in range(5):
    sol = ND.simulate(trange,[0,0,0,0],Iapp)
    
    # Time plot
    plt.figure(1)
    plt.plot(sol.t, sol.y[0])
    
    # Gating variable steady-state functoins
    ax1.plot(V1, ND.m.inf(V1))
    ax2.plot(V1, ND.m.tau(V1))
    ax3.plot(V1, ND.h.inf(V1))
    ax4.plot(V1, ND.h.tau(V1))
    ax5.plot(V1, ND.n.inf(V1))
    ax6.plot(V1, ND.n.tau(V1))
    
    # IV curves
    Ifast = ND.iL_ss(V2) + ND.iNa_ss(V2)
    Islow = Ifast + ND.iK_ss(V2)
    ax7.plot(V2, Ifast)
    ax8.plot(V2, Islow)
    
    ND.perturb()

plt.show()