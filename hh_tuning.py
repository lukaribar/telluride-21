# -*- coding: utf-8 -*-
"""
Graphical interface for controlling single neuron behavior.
Neuron consists of 4 current source elements representing fast -ve, slow +ve,
slow -ve, and ultra-slow +ve conductance.

@author: Luka
"""
from gui_utilities import GUI
from cb_models import HHModel

neuron = HHModel()

gui = GUI(neuron)