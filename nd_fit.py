"""
Fit NeuroDyn model
"""

from fitting_utilities import FitND
from cb_models import NeuroDynModel, HHModel

# Voltage scaling (important: assumes that HH is already written in SI units)
scl_v = 3

ND = NeuroDynModel()
HH = HHModel(scl=scl_v*1e-3)

fit = FitND(ND, HH)
