"""
Fit NeuroDyn model
"""
#%% Initial fit
from fitting_utilities import FitND
from cb_models import NeuroDynModel, HHModel
import matplotlib.pyplot as plt

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 3

ND = NeuroDynModel()
HH = HHModel(scl=scl_v*1e-3)

fit = FitND(ND, HH)

#%% Fit gating variables individually and compute quantized parameters
c = fit.fit(plot_inf_tau=False)
g0 = [120,36,0.3]
dIb,dg = fit.quantize(c,g0)

#%% Calculate the NeuroDyn parameters and simulate
E0 = [120,-12,10.6]
I0 = fit.convert_I(0)
Iapp = lambda t : I0

# g = fit.convert_gmax(g0)
E = fit.convert_Erev(E0)
Vhigh, Vlow = fit.get_Vb_bounds()

ND = NeuroDynModel(dg, E, dIb, Vhigh, Vlow)

T = 0.01
trange = (0, T)

sol = ND.simulate(trange,[0,0,0,0],Iapp)

plt.figure()
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')
plt.plot(sol.t, sol.y[0])
plt.show()