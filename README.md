# Telluride 2021 - NMC21: Neuromodulation Control resources
This repository provides the code for defining and simulating a software model of the NeuroDyn hardware ([T. Yu and G. Cauwenberghs, 2010](https://isn.ucsd.edu/pub/papers/tbiocas10_neurodyn.pdf)), as well as Hodgkin-Huxley-type conductance-based models and their networks.

The repository also contains Jupyter notebooks that provide a guide to building and simulating neural networks using either the NeuroDyn or Hodgkin-Huxley neuronal models.

## Model Definitions
`cb_models.py`

The file provides the main model definitions for NeuroDyn and Hodgkin-Huxley type neurons, as well as for defining arbitrary neural networks using the two models.

## Fitting Utilities
`fitting_utilities.py`

The file provides the class and the methods for mapping the parameters of a Hodgkin-Huxley-type neuron to the parameters of the NeuroDyn software model.

## Notebooks

### Notebook 1 - Designing excitable behaviors
`Designing excitable behaviors.ipynb`

This notebook offers an introduction to designing and controlling spiking and bursting behaviors using a simplified neuronal model described in [Neuromodulation of Neuromorphic Circuits](https://arxiv.org/abs/1805.05696) and [Neuromorphic Control](https://arxiv.org/abs/2011.04441). The model definitions used by this notebook are contained in `iv_model.py`.

Additional resources for this model are provided at https://github.com/lukaribar/Circuit-Neuromodulation.

### Notebook 2 - NeuroDyn Python model
`NeuroDyn Python model.ipynb`

This notebook provides a guide to using the Python model of a NeuroDyn circuit, along with the environment for simulating and fitting Hodgkin-Huxley-type conductance-based models.

### Notebook 3 - Complex biophysical models in NeuroDyn
`Complex biophysical models in NeuroDyn.ipynb`

This notebook provides a guide to interconnecting elementary Hodgkin-Huxley circuits in order to build more complex neuronal behavior such as burst firing.

### Notebook 4 - Networks in NeuroDyn
`Networks in NeuroDyn.ipynb`

This notebook provides a guide to defining networks using Hodgkin-Huxley and NeuroDyn models.
