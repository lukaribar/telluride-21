#%%
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cb_models import HHActivation, HHInactivation, HHModel

# Create nominal HH model
HH = HHModel(SI_units=True)
sclHH = HHModel(scl_v=2, scl_t=1, SI_units=True)

I0 = 8
Iapp = lambda t : I0

T = 200e-3
trange = (0, T)

sol_HH = HH.simulate(trange,[0,0,0,0],Iapp)
sol_sclHH = sclHH.simulate(trange,[0,0,0,0],Iapp)

# Plot the simulation
plt.figure()
plt.plot(sol_HH.t, sol_HH.y[0])
plt.plot(sol_sclHH.t, sol_sclHH.y[0])
plt.legend(['HH'])