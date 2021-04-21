"""
Classes and methods for fitting NeuroDyn model
"""
from cb_models import HHActivation, HHInactivation
import numpy as np
from scipy.optimize import nnls, minimize, Bounds
import matplotlib.pyplot as plt

class FitND:
    """
    Class takes the physical constants from a NeuroDyn instance and provides
    methods for setting the parameters of the NeuroDyn chip to fit a Hodgkin-
    Huxley model
    """
    def __init__(self, NDModel, HHModel, vrange=[]):
        self.NDModel = NDModel
        self.HHModel = HHModel
        
        # Set default fitting range
        if (vrange == []):
            scl_v = HHModel.scl
            vstart = HHModel.Ek # add vrefs here?
            vend   = 80*scl_v # V_ref + HH.Ena -> why this??
            vrange = np.arange(vstart, vend, 5e-4).T
        self.vrange = vrange
        
        # Physical constants -> params = kappa,C,C_ND,Vt,I_tau,I_ref,V_ref
        self.params = self.NDModel.get_pars()
        
        # Initial fit to find optimal Vstep and Vmean for spline bias voltages
        self.initial_fit()
    
    def fit_gating_variable(self, x):
        """
        Fit a single gating variable using non-negative least squares
        IMPORTANT: c_a and c_b returned by this function ignores the factor of 
        1000 due to HH's time units, which are in miliseconds
        """
        kappa,C,C_ND,Vt,I_tau,I_ref,V_ref = self.params
        Vrange = self.vrange
        Vb = self.get_Vb()
        
        A_alpha = np.zeros((np.size(Vrange),7))
        A_beta = np.zeros((np.size(Vrange),7))
        b_alpha = x.alpha(Vrange)
        b_beta = x.beta(Vrange)
        for i in range(7):
            if isinstance(x,HHActivation):
                A_alpha[:,i] = 1 / (1 + np.exp(1 * kappa * (Vb[i] - Vrange)  / Vt))
                A_beta[:,i] = 1 / (1 + np.exp(-1 * kappa * (Vb[i] - Vrange)  / Vt))
            else:
                A_alpha[:,i] = 1 / (1 + np.exp(-1 * kappa * (Vb[i] - Vrange)  / Vt))
                A_beta[:,i] = 1 / (1 + np.exp(1 * kappa * (Vb[i] - Vrange)  / Vt))
        c_a = nnls(A_alpha,b_alpha)[0]
        c_b = nnls(A_beta,b_beta)[0]
    
        return c_a,c_b,A_alpha,A_beta

    
    def I_rate(self,c,sign,Vb):
        kappa,C,C_ND,Vt,I_tau,I_ref,V_ref = self.params
        I=0
        for i in range(len(Vb)):
            I += c[i] / (1 + np.exp(sign * kappa * (Vb[i] - self.vrange)  / Vt))
        return I
    
    def cost(self,Z,X):
        """
        inputs:
            Z contains the list of free variables
            X is a list of HHKinetics objects to fit
        output:
            value of the cost
        """
        Vmean = Z[-2]
        Vstep = Z[-1]
        Vb = Vmean + np.arange(start=-3,stop=4,step=1)*Vstep
        Vrange = self.vrange
    
        out = 0
        for i, x in enumerate(X):
            c_a = Z[i*7:(i+1)*7]
            c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
            norm_a = max(x.alpha(Vrange))
            norm_b = max(x.beta(Vrange))        
            if isinstance(x,HHActivation):    
                out += sum(((x.alpha(Vrange) - self.I_rate(c_a,1,Vb))/norm_a)**2) 
                out += sum(((x.beta(Vrange) - self.I_rate(c_b,-1,Vb))/norm_b)**2) 
            elif isinstance(x, HHInactivation):
                out += sum(((x.alpha(Vrange) - self.I_rate(c_a,-1,Vb))/norm_a)**2) 
                out += sum(((x.beta(Vrange) - self.I_rate(c_b,1,Vb))/norm_b)**2)
            else:
                print("Kinetics object not supported")
                return
            
        return out
    
    def initial_fit(self):
        """
        Initial fit to find optimal bias voltages
        """
        kappa,C,C_ND,Vt,I_tau,I_ref,V_ref = self.params
        
        # Fit Hodgkin-Huxley gating variables
        X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        
        # Initial parameter values
        C_a = np.array([])
        C_b = np.array([])
        for i, x in enumerate(X):
            C_a = np.append(C_a,max(x.alpha(self.vrange))*np.ones(7)/7)
            C_b = np.append(C_b,max(x.beta(self.vrange))*np.ones(7)/7)
        Vmean = V_ref
        Vstep = (V_ref+self.HHModel.Ena/1e3 - V_ref+self.HHModel.Ek/1e3)/100
        Z0 = np.concatenate([C_a,C_b,np.array([Vmean,Vstep])])
        
        lowerbd = np.append(np.zeros(14*len(X)),np.array([-np.inf,-np.inf]))
        upperbd = np.append(np.ones(14*len(X))*np.inf,np.array([np.inf,np.inf]))
        bd = Bounds(lowerbd,upperbd)
        
        Z = minimize(lambda Z : self.cost(Z,X), Z0, bounds = bd)
        Z = Z.x
        
        self.Vmean = Z[-2]
        self.Vstep = Z[-1] 
        
        # Save the initial fit
        self.Z0 = Z
        return
    
    def get_Vb(self):
        return self.Vmean + np.arange(start=-3,stop=4,step=1)*self.Vstep
    
    def plot_initial_fit(self, plot_alpha_beta = True, plot_inf_tau = False):
        print("Vmean:", self.Vmean)
        print("Vstep:", self.Vstep)
        
        X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        Z = self.Z0
        
        Vb = self.get_Vb()
        Vrange = self.vrange
        
        for i,x in enumerate(X):
            c_a = Z[i*7:(i+1)*7]
            c_b = Z[len(X)*7+i*7:len(X)*7+(i+1)*7]
            if isinstance(x,HHActivation):
                alpha = self.I_rate(c_a,1,Vb)
                beta = self.I_rate(c_b,-1,Vb)
            else:
                alpha = self.I_rate(c_a,-1,Vb)
                beta = self.I_rate(c_b,1,Vb)
        
            gatelabels = ['m','h','n']
            
            # Plot alpha and beta fits
            if (plot_alpha_beta):
                plt.figure()
                plt.plot(Vrange,x.alpha(Vrange),label='HH α_'+gatelabels[i])
                plt.plot(Vrange,alpha,label='fit α_'+gatelabels[i])
                plt.legend()
            
                plt.figure()
                plt.plot(Vrange,x.beta(Vrange),label='HH β_'+gatelabels[i])
                plt.plot(Vrange,beta,label='fit β_'+gatelabels[i])
                plt.legend()
            
            # Plot xinf and tau fits
            if (plot_inf_tau):
                tau = 1/(alpha+beta)
                inf = alpha/(alpha+beta)
    
                plt.figure()
                plt.plot(Vrange,x.tau(Vrange),label='HH τ_'+gatelabels[i])
                plt.plot(Vrange,tau,label='fit τ_'+gatelabels[i])
                plt.legend()
        
                plt.figure()
                plt.plot(Vrange,x.inf(Vrange),label='HH '+gatelabels[i]+'_∞')
                plt.plot(Vrange,inf,label='fit '+gatelabels[i]+'_∞')
                plt.legend()