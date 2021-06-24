"""
Fit NeuroDyn model
"""
#%% Initial fit
from fitting_utilities import FitND
from cb_models import NeuroDynModel, HHModel
import matplotlib.pyplot as plt

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 3

#ND = NeuroDynModel()
HH = HHModel(scl_v=scl_v, SI_units=True)

fit = FitND(HH)

#%% Fit gating variables individually and compute quantized parameters
c = fit.fit(plot_alpha_beta=True)
g0 = [120e-3,36e-3,0.3e-3]
E0 = [120e-3,-12e-3,10.6e-3]
dIb,dg,dE,scl_t = fit.quantize(c,g0,E0)
dIb[2][1] = dIb[2][1]*15 # This parameter is too small for some reason!!!

#%% Calculate the NeuroDyn parameters and simulate
I0 = 0e-6
Iapp = lambda t : fit.convert_I(I0)

V_ref = 0.9

ND = NeuroDynModel(dg, dE, dIb, V_ref, fit.I_voltage, fit.I_master)

T = 0.02
trange = (0, T)

sol = ND.simulate(trange,[0.7,0,0,0],Iapp)

plt.figure()
plt.xlabel('t')
plt.ylabel('V')
plt.title('NeuroDyn simulation')
plt.plot(sol.t, sol.y[0])
plt.show()

#%% Print the parameter values
print('\nImaster = ', ND.I_master)
print('Ivoltage = ', ND.I_voltage)
print('Vref = ', V_ref, '\n')

print('Digital values for maximal conductances:')
print('[gna, gk, gl] = ', ND.dg, '\n')

print('Digital values for reversal potentials:')
print('[Ena, Ek, El] = ', dE, '\n')

print('Digital values for gating variable kinetics:')
print('alpha_m = ', dIb[0][0])
print('beta_m = ', dIb[0][1], '\n')
print('alpha_h = ', dIb[1][0])
print('beta_h = ', dIb[1][1], '\n')
print('alpha_n = ', dIb[2][0])
print('beta_n = ', dIb[2][1], '\n')