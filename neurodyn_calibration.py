#%% 
from cb_models import NeuroDynModel
import numpy as np
import matplotlib.pyplot as plt

ND = NeuroDynModel()
I0 = 0.0
Iapp = lambda t : I0

T = 0.01
trange = (0, T)
sol = ND.simulate(trange,[0,0,0,0],Iapp)

plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.legend(['v(t)'], shadow=True)
plt.title('NeuroDyn simulation')
plt.show()