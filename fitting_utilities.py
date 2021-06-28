"""
Classes and methods for fitting NeuroDyn model
Important: all fitting is done assuming that the rest potential 
(or the threhsold potential) of the biophysical model is at 0V.
"""

import numpy as np
from scipy.optimize import nnls #, minimize, Bounds
import matplotlib.pyplot as plt
from cb_models import NeuroDynModel

class FitND:
    """
    Class takes the physical constants from a NeuroDyn model and provides
    methods for setting the parameters of the NeuroDyn chip to fit a Hodgkin-
    Huxley model that is passed as the argument.
    """
    def __init__(self, HHModel, vrange = None, capacitance_scaling = 1, 
                 I_voltage = 150e-9, I_master = 200e-9):
        
        NDModel = NeuroDynModel(I_voltage = I_voltage, I_master = I_master,
                               capacitance_scaling = capacitance_scaling,
                               V_ref = 0)
        self.NDModel = NDModel
        self.HHModel = HHModel
        
        # Set default fitting range
        if (vrange == None):
            vstart = HHModel.Ek
            vend   = HHModel.Ena / 2 # this can be changed
            vrange = np.arange(vstart, vend, 5e-4).T
        self.vrange = vrange
        
        # Vmean: mid-point of the middle (4th) sigmoid
        #   -> Set to middle of voltage range (or resting potential?)
        self.Vmean = (HHModel.Ek + HHModel.Ena) / 2
        
        # Update dictionary with physical constants from NeuroDyn model
        params = NDModel.get_pars()
        self.__dict__.update(params)
        
        # NeuroDyn currents (should be between ~50nA-400nA)
        self.I_voltage = I_voltage
        self.I_master = I_master
                
        # Calculate base voltages (V_ref = 0 so Vb will be centered around 0)
        self.Vb = NDModel.get_Vb() + self.Vmean # add Vmean to center around it
        
        # Maximal coefficients
        self.wmax = 0
        self.gmax = 0
        self.scl_t = 0
        
        # Ratio of membrane capacitors C_HH / C_ND
        C_HH = self.HHModel.C_m
        self.C_ratio = C_HH / self.C_ND
        
    def fit_gating_variable(self, x):
        """
        Fit a single gating variable using non-negative least squares
        """
        Vrange = self.vrange
        Vb = self.Vb
        
        A_alpha = np.zeros((np.size(Vrange), 7))
        A_beta = np.zeros((np.size(Vrange), 7))
        b_alpha = x.alpha(Vrange)
        b_beta = x.beta(Vrange)
        
        # Check if the gating variable is activation or inactivation
        isActivation = b_alpha[-1] > b_alpha[0]
        
        for i in range(7):
            if isActivation:
                A_alpha[:,i] = 1 / (1 + np.exp(1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
                A_beta[:,i] = 1 / (1 + np.exp(-1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
            else:
                A_alpha[:,i] = 1 / (1 + np.exp(-1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
                A_beta[:,i] = 1 / (1 + np.exp(1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
        
        w_a = nnls(A_alpha, b_alpha)[0]
        w_b = nnls(A_beta, b_beta)[0]
        
        w = [w_a, w_b]
        A = [A_alpha, A_beta]
        
        return w, A
    
    def convert_w_to_Ib(self, w):
        """
        Convert spline weights (units: 1/s) into appropriate NeuroDyn current
        biases (units: A)
        """
        w = np.asarray(w)
        Ib = w * self.C * self.Vt
        return Ib
    
    def fitHH(self, plot_alpha_beta = False, plot_inf_tau = False):
        """
        Fit the parameters to the Hodgkin-Huxley model passed during
        initialization
        """
        
        # Update scl_t based on maximal conductances of the Hodgkin-Huxley model
        hh = self.HHModel
        self.update_scl_t(g = [hh.gna, hh.gk, hh.gl])
        
        X = [self.HHModel.m, self.HHModel.h, self.HHModel.n]
        labels = ['m', 'h', 'n']
        
        return self.fit(X, labels, plot_alpha_beta, plot_inf_tau)
    
    def fit(self, X, labels = None, plot_alpha_beta = False,
            plot_inf_tau = False):
        """
        Fits a list of gating variables in X with names in labels list.
        Returns a list of sigmoid basis functions weights.
        """
        Vrange = self.vrange

        if (labels == None):
            labels = ['x' for el in X]
        
        # Fit the variables and plot results
        weights = []
        A = []
        for x in X:
            w_x, A_x = self.fit_gating_variable(x)
            weights.append(w_x)
            A.append(A_x)
        
        # Find the quantized fit
        dIb = self.get_digital_Ib(weights) 
        scl_t = self.scl_t
        weights_quant = dIb * (self.I_master / 1024) / (self.C * self.Vt) * scl_t
        
        # Plot alpha and beta functions / tau and inf functions
        for w, w_quant, Aj, x, label in zip(weights, weights_quant, A, X, labels):
            alpha = np.dot(Aj[0], w[0])
            beta = np.dot(Aj[1], w[1])
            
            alpha_quant = np.dot(Aj[0], w_quant[0])
            beta_quant = np.dot(Aj[1], w_quant[1])
            
            # Plot alpha and beta fits
            if (plot_alpha_beta):
                plt.figure()
                plt.title('α_' + label)
                plt.plot(Vrange, x.alpha(Vrange), label = 'Original')
                plt.plot(Vrange, alpha, label = 'Fit')
                plt.plot(Vrange, alpha_quant, label = 'Fit quantized')
                plt.legend()
            
                plt.figure()
                plt.title('β_'+label)
                plt.plot(Vrange, x.beta(Vrange), label = 'Original')
                plt.plot(Vrange, beta, label = 'Fit')
                plt.plot(Vrange, beta_quant, label = 'Fit quantized')
                plt.legend()
            
            # Plot xinf and tau fits
            if (plot_inf_tau):
                tau = 1 / (alpha + beta)
                inf = alpha / (alpha + beta)
                
                tau_quant = 1 / (alpha_quant + beta_quant)
                inf_quant = alpha_quant / (alpha_quant + beta_quant)
    
                plt.figure()
                plt.title('τ_' + label)
                plt.plot(Vrange, x.tau(Vrange), label='Original')
                plt.plot(Vrange, tau, label = 'Fit')
                plt.plot(Vrange, tau_quant, label ='Fit quantized')
                plt.legend()
        
                plt.figure()
                plt.title(label + '_∞')
                plt.plot(Vrange, x.inf(Vrange), label = 'Original')
                plt.plot(Vrange, inf, label = 'Fit')
                plt.plot(Vrange,inf_quant,label = 'Fit quantized')
                plt.legend()
                
        return weights
    
    def get_scl_t(self):
        """
        Return scl_t variable that can be passed to a HHModel.
        """
        return (1 / self.scl_t)
    
    def update_scl_t(self, w = None, g = None):
        """
        Update scl_t based on the input array of weights (w) and/or maximal
        conductances (g). scl_t is updated so that all weights and max
        conductances can be fitted within the NeuroDyn physical range.
        """
        Imax = (1023 * self.I_master / 1024) # max current in the circuit
        
        if w is not None:
            w = np.asarray(w)
            wmax = w.max()
        
            # Find maximum coefficient
            self.wmax = max(self.wmax, wmax)
            
            # Find the scaling factor that maximizes coefficient resolution
            scl_t = self.convert_w_to_Ib(self.wmax) / Imax
            self.scl_t = max(scl_t, self.scl_t)
        
        if g is not None:
            g = np.asarray(g)
            gmax = g.max()
            
            # Find maximum conductance
            self.gmax = max(gmax, self.gmax)
        
            # Find the scaling factor that maximizes conductance resolution
            scl_t = self.gmax / (Imax * self.kappa / self.Vt) / self.C_ratio
            self.scl_t = max(scl_t, self.scl_t)
        
    def get_Ib(self, weights):
        """
        Returns analog values for sigmoid bias currents.
        """
        Ib = self.convert_w_to_Ib(weights) / self.scl_t
        return Ib
    
    def get_digital_Ib(self, weights, update_scale = True):
        """
        Returns digital sigmoid basis functions coefficients from the weights
        obtained through fitting.
        """
        if update_scale:
            self.update_scl_t(w = weights)
        
        # Find the bias currents and quantize
        Ib = self.get_Ib(weights)
        dIb = np.round(Ib * 1024 / self.I_master)
        
        if not((abs(dIb) <= 1023).all()):
            print("Some digital values are out of range")
            
        return dIb
    
    def get_digital_g(self, g, update_scale = True):
        """
        Returns digital values for the maximal conductance parameters g.
        """
        g = np.asarray(g)
        
        if update_scale:
            self.update_scl_t(g = g)
        
        # Quantize conductances
        g_factor = (self.kappa / self.Vt) * (self.I_master / 1024)
        dg = np.round(g / self.scl_t / self.C_ratio / g_factor)
        
        if not((abs(dg) <= 1023).all()):
            print("Some digital values are out of range")
        
        return dg
    
    def get_digital_E(self, E):
        """
        Returns digital values for the reversal potentials E.
        """
        E = np.asarray(E)
        
        # Quantize reversal potentials
        E_factor = (self.I_voltage / 1024) * self.Res
        scl_v = self.HHModel.scl_v
        dE = np.round((E * scl_v - self.Vmean) / E_factor)
        
        if not((abs(dE) <= 1023).all()):
            print("Some digital values are out of range")
        
        return dE
    
    def get_analog_Ib(self, weights, update_scale = True):
        """
        Get analog current values for sigmoid biases corresponding to weights.
        """
        if update_scale:
            self.update_scl_t(w = weights)
        
        Ib = self.get_Ib(weights)
        
        return Ib
    
    def get_analog_g(self, g, update_scale = True):
        """
        Get analog current values for maximal conductances.
        """
        g = np.asarray(g)
        
        if update_scale:
            self.update_scl_t(g = g)
        
        g = g / self.scl_t / self.C_ratio
        
        return g
    
    def get_analog_E(self, E):
        """
        Get analog current values for reversal potentials.
        """
        E = np.asarray(E) * self.HHModel.scl_v - self.Vmean
        
        return E
    
    def convert_I(self, I0):
        """
        Scale an input to a standard Hodgkin-Huxley model to a NeuroDyn input.
        """
        scl_v = self.HHModel.scl_v
        I = I0 * scl_v / self.scl_t / self.C_ratio
        return I
            
    # INITIAL FIT METHODS
        
    # def I_rate(self,c,sign,Vb):
    #     I=0
    #     for i in range(len(Vb)):
    #         I += c[i] / (1 + np.exp(sign * self.kappa * (Vb[i] - self.vrange)  / self.Vt))
    #     return I
    
    # # Only used for the initial fit
    # def cost(self,Z,X):
    #     """
    #     inputs:
    #         Z contains the list of free variables
    #         X is a list of HHKinetics objects to fit
    #     output:
    #         value of the cost
    #     """
    #     Vmean = Z[-2]
    #     I_voltage = Z[-1]
    #     Vstep = I_voltage*(1.85*1e6)/3.5
    #     Vb = Vmean + np.arange(start=-3,stop=4,step=1)*Vstep
    #     Vrange = self.vrange
    
    #     out = 0
    #     for i, x in enumerate(X):
    #         c_a = Z[i*7:(i+1)*7]
    #         c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
    #         norm_a = max(x.alpha(Vrange))
    #         norm_b = max(x.beta(Vrange))        
    #         if isinstance(x,HHActivation):    
    #             out += sum(((x.alpha(Vrange) - self.I_rate(c_a,1,Vb))/norm_a)**2) 
    #             out += sum(((x.beta(Vrange) - self.I_rate(c_b,-1,Vb))/norm_b)**2) 
    #         elif isinstance(x, HHInactivation):
    #             out += sum(((x.alpha(Vrange) - self.I_rate(c_a,-1,Vb))/norm_a)**2) 
    #             out += sum(((x.beta(Vrange) - self.I_rate(c_b,1,Vb))/norm_b)**2)
    #         else:
    #             print("Kinetics object not supported")
    #             return
            
    #     return out
        
    # def initial_fit(self):
    #     """
    #     Initial fit to find optimal Vmean and I_voltage
    #     """
    #     # Fit Hodgkin-Huxley gating variables
    #     X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        
    #     # Initial parameter values
    #     C_a = np.array([])
    #     C_b = np.array([])
    #     for x in X:
    #         C_a = np.append(C_a,max(x.alpha(self.vrange))*np.ones(7)/7)
    #         C_b = np.append(C_b,max(x.beta(self.vrange))*np.ones(7)/7)
    #     Z0 = np.concatenate([C_a,C_b,np.array([self.Vmean, self.I_voltage])])
        
    #     # Bounds for Vmean and Ivoltage
    #     vmin = self.HHModel.Ek
    #     vmax = self.HHModel.Ena
    #     Imin = 50e-9
    #     Imax = 400e-9
        
    #     lowerbd = np.append(np.zeros(14*len(X)),np.array([vmin, Imin]))
    #     upperbd = np.append(np.ones(14*len(X))*np.inf,np.array([vmax, Imax]))
    #     bd = Bounds(lowerbd,upperbd)
        
    #     Z = minimize(lambda Z : self.cost(Z,X), Z0, bounds = bd)
    #     Z = Z.x
        
    #     # Check here if reversal potentials are covered by the voltage range
    #     #
    #     #
        
    #     self.Vmean = Z[-2]
    #     self.Ivoltage = Z[-1] 
        
    #     # Save the initial fit
    #     self.Z0 = Z
    #     return
    
    # This should probably go into initial_fit method
    # def plot_initial_fit(self, plot_alpha_beta = True, plot_inf_tau = False):
    #     print("Vmean:", self.Vmean)
    #     print("Vstep:", self.Vstep)
        
    #     X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
    #     Z = self.Z0
        
    #     Vb = self.get_Vb()
    #     Vrange = self.vrange
        
    #     for i,x in enumerate(X):
    #         c_a = Z[i*7:(i+1)*7]
    #         c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
    #         if isinstance(x,HHActivation):
    #             alpha = self.I_rate(c_a,1,Vb)
    #             beta = self.I_rate(c_b,-1,Vb)
    #         else:
    #             alpha = self.I_rate(c_a,-1,Vb)
    #             beta = self.I_rate(c_b,1,Vb)
        
    #         gatelabels = ['m','h','n']
            
    #         # Plot alpha and beta fits
    #         if (plot_alpha_beta):
    #             plt.figure()
    #             plt.plot(Vrange,x.alpha(Vrange),label='HH α_'+gatelabels[i])
    #             plt.plot(Vrange,alpha,label='fit α_'+gatelabels[i])
    #             plt.legend()
            
    #             plt.figure()
    #             plt.plot(Vrange,x.beta(Vrange),label='HH β_'+gatelabels[i])
    #             plt.plot(Vrange,beta,label='fit β_'+gatelabels[i])
    #             plt.legend()
            
    #         # Plot xinf and tau fits
    #         if (plot_inf_tau):
    #             tau = 1/(alpha+beta)
    #             inf = alpha/(alpha+beta)
    
    #             plt.figure()
    #             plt.plot(Vrange,x.tau(Vrange),label='HH τ_'+gatelabels[i])
    #             plt.plot(Vrange,tau,label='fit τ_'+gatelabels[i])
    #             plt.legend()
        
    #             plt.figure()
    #             plt.plot(Vrange,x.inf(Vrange),label='HH '+gatelabels[i]+'_∞')
    #             plt.plot(Vrange,inf,label='fit '+gatelabels[i]+'_∞')
    #             plt.legend()