# -*- coding: utf-8 -*-
"""
An example of a network simulation

@author: Luka
"""

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from neuron_model import Neuron
from network_model import CurrentSynapse, ResistorInterconnection, Network

# Define timescales
tf = 0
ts = 50
tus = 50*50

# Define circuit elements
n = 2 # number of neurons
neurons = [] # list of all neurons in the network

for j in range(2):
    # Define empty neurons and then interconnect the elements
    neuron = Neuron()
    R = neuron.add_conductance(1)
    i1 = neuron.add_current(-2, 0, tf) # fast negative conductance
    i2 = neuron.add_current(2, 0, ts) # slow positive conductance
    i3 = neuron.add_current(-1.5, -1.5, ts) # slow negative conductance
    i4 = neuron.add_current(1.5, -1.5, tus) # ultraslow positive conductance
    
    neurons.append(neuron)

# Define the connectivity matrices
g_inh = [[0, 0], [0, 0]] # inhibitory connection neuron 1 -| neuron 2
g_exc = [[0, 0], [0, 0]] # excitatory connection neuron 1 <- neuron 2
g_res = [[0, 0], [0, 0]] # no resistive connections

voff = -1
inh_synapse = CurrentSynapse(-1, voff, ts)
exc_synapse = CurrentSynapse(+1, voff, ts)
resistor = ResistorInterconnection()

# Define the network
network = Network(neurons, (inh_synapse, g_inh), (exc_synapse, g_exc),
                  (resistor, g_res))

# Simulate the network
trange = (0, 20000)

# Define i_app as a function of t: returns an i_app for each neuron
i_app = lambda t: [-2.1, -2]

# Return dy/dt of the system
def odesys(t, y):
    return network.sys(i_app(t),y)

# Initial conditions
y0 = network.get_init_conditions()

# ODE solver
sol = solve_ivp(odesys, trange, y0)

# Plot simulation
# y[0] = neuron 1 membrane voltage, y[3] = neuron 2 membrane voltage
plt.figure()
plt.plot(sol.t, sol.y[0], sol.t, sol.y[3])