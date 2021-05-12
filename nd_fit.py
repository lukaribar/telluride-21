"""
Fit NeuroDyn model
"""
#%% Initial fit
from fitting_utilities import FitND
from cb_models import NeuroDynModel, HHModel
import matplotlib.pyplot as plt
import numpy as np

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 2.5

ND = NeuroDynModel()
HH = HHModel(scl_v=scl_v, SI_units=True)

fit = FitND(ND, HH)
#fit.plot_initial_fit()

#%% Fit gating variables individually and compute quantized parameters
c = fit.fit(plot_alpha_beta=True)
g0 = [120e-3,36e-3,0.3e-3]
E0 = [120e-3,-12e-3,10.6e-3]
dIb,dg,dE,scl_t = fit.quantize(c,g0,E0)
dIb[2][1] = dIb[2][1]*7 # DELETE THIS

#%% Calculate the NeuroDyn parameters and simulate
I0 = 0e-6
Iapp = lambda t : fit.convert_I(I0)

#V_ref = 0.9
V_ref = 0

ND = NeuroDynModel(dg, dE, dIb, V_ref, fit.I_voltage, fit.I_tau)

vrange = np.arange(HH.Ek, HH.Ena, 5e-4).T
T = 0.01
trange = (0, T)

sol = ND.simulate(trange,[0,0,0,0],Iapp)

plt.figure()
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')
plt.plot(sol.t, sol.y[0])
plt.show()