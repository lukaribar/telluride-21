# -*- coding: utf-8 -*-
"""
Graphical interface for controlling single neuron behavior.
Neuron consists of 4 current source elements representing fast -ve, slow +ve,
slow -ve, and ultra-slow +ve conductance.

@author: Luka
"""
from gui_utilities import GUI
from neuron_model import Neuron

# Initial conductance parameters
a_f = -2
voff_f = 0
a_s1 = 2
voff_s1 = 0
a_s2 = -1.5
voff_s2 = -0.9
a_us = 1.5
voff_us = -0.9

# Define timescales
tf = 0
ts = 50
tus = 50*50

# Define an empty neuron and then interconnect the elements
neuron = Neuron()
R = neuron.add_conductance(1)
i1 = neuron.add_current(a_f, voff_f, tf) # fast negative conductance
i2 = neuron.add_current(a_s1, voff_s1, ts) # slow positive conductance
i3 = neuron.add_current(a_s2, voff_s2, ts) # slow negative conductance
i4 = neuron.add_current(a_us, voff_us, tus) # ultraslow positive conductance

gui = GUI(neuron, i0 = -2, plot_fixed_point = True)

gui.add_IV_curve(neuron, "Fast", tf, [0.1, 0.75, 0.2, 0.2])
gui.add_IV_curve(neuron, "Slow", ts, [0.4, 0.75, 0.2, 0.2])
gui.add_IV_curve(neuron, "Ultraslow", tus, [0.7, 0.75, 0.2, 0.2])

gui.add_label(0.25, 0.34, "Fast -ve")
s1 = gui.add_slider("Gain", [0.1, 0.3, 0.3, 0.03], 0, 4, a_f, i1.update_a, sign=-1)
s2 = gui.add_slider("$V_{off}$", [0.1, 0.25, 0.3, 0.03], -2, 2, voff_f, i1.update_voff)

gui.add_label(0.25, 0.19, "Slow +ve")
s3 = gui.add_slider("Gain", [0.1, 0.15, 0.3, 0.03], 0, 4, a_s1, i2.update_a)
s4 = gui.add_slider("$V_{off}$", [0.1, 0.1, 0.3, 0.03], -2, 2, voff_s1, i2.update_voff)

gui.add_label(0.75, 0.34, "Slow -ve")
s5 = gui.add_slider("Gain", [0.6, 0.3, 0.3, 0.03], 0, 4, a_s2, i3.update_a, sign=-1)
s6 = gui.add_slider("$V_{off}$", [0.6, 0.25, 0.3, 0.03], -2, 2, voff_s2, i3.update_voff)

gui.add_label(0.75, 0.19, "UltraSlow +ve")
s7 = gui.add_slider("Gain", [0.6, 0.15, 0.3, 0.03], 0, 4, a_us, i4.update_a)
s8 = gui.add_slider("$V_{off}$", [0.6, 0.1, 0.3, 0.03], -2, 2, voff_us, i4.update_voff)

s9 = gui.add_iapp_slider([0.1, 0.02, 0.5, 0.03], -3, 3)

b = gui.add_button("Pause", [0.8, 0.02, 0.1, 0.03], gui.pause)

gui.run()