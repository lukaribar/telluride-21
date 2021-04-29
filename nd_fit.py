"""
Fit NeuroDyn model
"""
#%% Initial fit
from fitting_utilities import FitND
from cb_models import NeuroDynModel, HHModel
import matplotlib.pyplot as plt

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 5

ND = NeuroDynModel()
HH = HHModel(scl=scl_v*1e-3)

fit = FitND(ND, HH)
# fit.plot_initial_fit()

#%% Fit gating variables individually and compute quantized parameters
c = fit.fit(plot_alpha_beta=True)
g0 = [120,36,0.3]
E0 = [120,-12,10.6]
dIb,dg,dE = fit.quantize(c,g0,E0)

#%% Calculate the NeuroDyn parameters and simulate
I0 = fit.convert_I(10)
Iapp = lambda t : I0

V_ref = 0.9

ND = NeuroDynModel(dg, dE, dIb, V_ref, fit.I_voltage, fit.I_tau)

T = 0.01
trange = (0, T)

sol = ND.simulate(trange,[0.9,0,0,0],Iapp)

plt.figure()
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')
plt.plot(sol.t, sol.y[0])
plt.show()