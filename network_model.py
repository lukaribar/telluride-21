# -*- coding: utf-8 -*-
"""
Network of neurons with synaptic and resistive interconnections.
Synaptic connections use either the 'Current' or 'Conductance' model

@author: Luka
"""

from neuron_model import sigmoid
import numpy as np

class Interconnection():
    """
    Arbitrary interconnecting element between two neurons
    """
    
    def __init__(self, timescale):
        self.timescale = timescale # element is instantaneous by default
    
    def check_connectivity_matrix(self, g, n):
        if np.array(g).shape != (n, n):
            raise ValueError("Invalid connectivity matrix size")
    
class CurrentSynapse(Interconnection):
    """
    Current source model of a synapse of the form:
        Isyn = +- S(k*(Vpre - voff))
        +: excitatory synapse
        -: inhibitory synapse
    """
    
    def __init__(self, sign, voff, timescale, k = 2):
        super().__init__(timescale)
        
        self.voff = voff
        self.sign = sign
        self.k = k
        
    def out(self, Vpre, Vpost = None):
        return self.sign * sigmoid(Vpre - self.voff, self.k)
    
class ConductanceSynapse(Interconnection):
    """
    Conductance-based model of a synapse of the form:
        Isyn = x * (Vpost - E_rev)
        x = S(k*(Vpre_x - voff))
        tau dVpre_x/dt = Vpre - Vpre_x
    """
    
    def __init__(self, slope, voff, E_rev, timescale):
        super().__init__(timescale)
        
        self.slope = slope
        self.voff = voff
        self.E_rev = E_rev
    
    def out(self, Vpre, Vpost):
        x = sigmoid(Vpre - self.voff, self.slope)
        return x * (Vpost - self.E_rev)

class ResistorInterconnection(Interconnection):
    """
    Ires = (Vpre - Vpost)
    """
    
    def __init__(self):
        super().__init__(0)
    
    def check_connectivity_matrix(self, g, n):
        super().check_connectivity_matrix(g,n)
        if not(np.allclose(np.array(g), np.array(g).T)):
            raise ValueError("Resistive matrix is not symmetric")
    
    def out(self, Vpre, Vpost):
        return (Vpre - Vpost)

class Network:
    """
    Neural network:
        neurons: list containing all neurons
        args: list of (synapse_model, g) tuples, so that for each synapse
        there is a connectivity matrix g describing the connection strengths.
        
        g[i][j] is the weight of the synaptic connection FROM neuron i TO
        neuron j.
    """
    
    def __init__(self, neurons, *args):    
        self.neurons = neurons # List containing all neurons
        self.n = len(self.neurons) # Number of neurons
        self.neuron_index = [] # Starting state index for every neuron
        self.y0 = [] # Initial conditions
        
        i = 0
        
        for neuron in self.neurons:
            # Set starting indices
            self.neuron_index.append(i)
            i += len(neuron.timescales)
            
            # Collect initial conditions
            self.y0.extend(neuron.get_init_conditions())
        
        # Check if connectivity matrices are valid
        for syn, g in args:
            syn.check_connectivity_matrix(g, self.n)
        
        self.synapses = args # List of synapse model/connectivity matrix pairs
    
    def get_init_conditions(self):
        return self.y0
        
    def sys(self, i_app, y):
        """
        Returns the state vector update
        y = vector containing states of all neurons, in order of definition
        """
        dy = []
        
        for i, neuron_i in enumerate(self.neurons):
            i_syn = 0
            
            index_i = self.neuron_index[i]
            Vpost = y[index_i]
            for j, neuron_j in enumerate(self.neurons):
                index_j = self.neuron_index[j]
                
                # Iterate through all synaptic and resistive connections
                for syn, g in self.synapses:
                    tau = syn.timescale
                    Vpre = y[index_j + neuron_j.timescales.index(tau)]
                    i_syn = i_syn + g[j][i] * syn.out(Vpre, Vpost)
                        
            i_external = i_app[i] + i_syn
            index_i_end = index_i + len(neuron_i.timescales)
            dv = neuron_i.sys(i_external, y[index_i:index_i_end])
            dy.extend(dv)
        
        return np.array(dy)