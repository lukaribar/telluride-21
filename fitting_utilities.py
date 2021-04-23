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
            vstart = HHModel.Ek     # add vrefs here?
            vend   = 80*scl_v       # V_ref + HH.Ena -> why this??
            vrange = np.arange(vstart, vend, 5e-4).T
        self.vrange = vrange
        
        # Update dictionary with physical constants from NeuroDyn model
        params = NDModel.get_pars()
        self.__dict__.update(params)
        
        # Maximal coefficients
        self.cmax = 0
        self.gmax = 0
        self.scl_t = 0 
        
        # Initial fit to find optimal Vstep and Vmean for spline bias voltages
        self.initial_fit()
                
    
    def fit(self, X=[], labels=[], plot_alpha_beta=False, plot_inf_tau=False):
        """
        Fits a list of gating variables in X.
        Returns sigmoid basis functions coefficients c prior to transformation
        into currents.
        """
        Vrange = self.vrange
        c = []
        A = []
        
        # By default just fit the original HH gating variables
        if (X == []):
            X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
            labels = ['m','h','n']
        elif (labels == []):
            labels = []*len(X) # put empty labels if none provided
        
        # Fit the variables and plot results
        for x,label in zip(X,labels):
            c_a,c_b,A_alpha,A_beta = self.fit_gating_variable(x)
            c.append([c_a, c_b])
            A.append([A_alpha, A_beta])
            
            alpha = np.dot(A_alpha,c_a)
            beta = np.dot(A_beta,c_b)
            tau = 1/(alpha+beta)
            inf = alpha/(alpha+beta)
            
            # Plot alpha and beta fits
            if (plot_alpha_beta):
                plt.figure()
                plt.plot(Vrange,x.alpha(Vrange),label='HH α_'+label)
                plt.plot(Vrange,alpha,label='fit α_'+label)
                plt.legend()
            
                plt.figure()
                plt.plot(Vrange,x.beta(Vrange),label='HH β_'+label)
                plt.plot(Vrange,beta,label='fit β_'+label)
                plt.legend()
            
            # Plot xinf and tau fits
            if (plot_inf_tau):
                tau = 1/(alpha+beta)
                inf = alpha/(alpha+beta)
    
                plt.figure()
                plt.plot(Vrange,x.tau(Vrange),label='HH τ_'+label)
                plt.plot(Vrange,tau,label='fit τ_'+label)
                plt.legend()
        
                plt.figure()
                plt.plot(Vrange,x.inf(Vrange),label='HH '+label+'_∞')
                plt.plot(Vrange,inf,label='fit '+label+'_∞')
                plt.legend()

        return c

    def quantize(self,c,g):
        """
        Returns quantized sigmoid basis functions coefficients after transformation
        of the coefficients (c) into quantizated currents (dIb). 
        Also returns quantized maximal conductances (dg).
        To perform quantization, a suitable time scaling has to be found.
        This is done so as to jointly maximize the resolution of the conductances 
        and the coefficients.
        """
        #kappa,C,C_ND,Vt,I_tau,_,_ = self.params

        # Find maximum coefficient
        cmax = np.array(c).max()
        if (cmax > self.cmax):
            self.cmax = cmax

        # Find the scaling factor that maximizes coefficient resolution
        C_HH = 1e-6
        scl_t = self.cmax * self.C * self.Vt / (1023*self.I_tau/1024) * 1000
        if (scl_t > self.scl_t):
            self.scl_t = scl_t

        # Find maximum conductance
        gmax = np.array(g).max()
        if (gmax > self.gmax):
            self.gmax = gmax

        # Find the scaling factor that maximizes conductance resolution
        scl_t = self.gmax * 1e-3 * self.Vt / self.kappa / (1023*self.I_tau/1024) * self.C_ND / C_HH
        if (scl_t > self.scl_t):
            self.scl_t = scl_t

        # Recover the (quantized) rate currents from fitted coefficients
        self.s = self.scl_t * C_HH / self.C_ND
        Ib = []
        dIb = []
        dg = []
        for i in range(len(c)):
            # Exact (real numbers) current coefficients, before quantization
            i_a = c[i][0] * self.C * self.Vt * 1000 / self.scl_t 
            i_b = c[i][1] * self.C * self.Vt * 1000 / self.scl_t
            Ib.append([i_a, i_b])
            # Quantize current coefficients
            di_a = np.round(i_a*1024/self.I_tau)
            di_b = np.round(i_b*1024/self.I_tau)
            dIb.append([di_a, di_b])

        Ib = np.array(Ib)*1024/self.I_tau
        dIb = np.array(dIb)

        # Quantize conductances
        dg = np.round(np.array(g)*1e-3*1024*self.Vt/self.kappa/self.I_tau/self.s)

        return dIb,dg
        

    def fit_gating_variable(self, x):
        """
        Fit a single gating variable using non-negative least squares
        IMPORTANT: c_a and c_b returned by this function ignores the factor of 
        1000 due to HH's time units, which are in miliseconds
        """
        #kappa,_,_,Vt,_,_,_ = self.params
        Vrange = self.vrange
        Vb = self.get_Vb()
        
        A_alpha = np.zeros((np.size(Vrange),7))
        A_beta = np.zeros((np.size(Vrange),7))
        b_alpha = x.alpha(Vrange)
        b_beta = x.beta(Vrange)
        for i in range(7):
            if isinstance(x,HHActivation):
                A_alpha[:,i] = 1 / (1 + np.exp(1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
                A_beta[:,i] = 1 / (1 + np.exp(-1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
            else:
                A_alpha[:,i] = 1 / (1 + np.exp(-1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
                A_beta[:,i] = 1 / (1 + np.exp(1 * self.kappa * (Vb[i] - Vrange)  / self.Vt))
        c_a = nnls(A_alpha,b_alpha)[0]
        c_b = nnls(A_beta,b_beta)[0]
    
        return c_a,c_b,A_alpha,A_beta

    def get_Vb_bounds(self):
        return self.Vmean+3.5*self.Vstep, self.Vmean-3.5*self.Vstep
    
    # This should return digital values
    def convert_Erev(self, E0_list):
        scl_v = self.HHModel.scl # note: includes V->mV conversion
        E_list = [E0*scl_v for E0 in E0_list]
        return E_list
    
    def convert_I(self, I0):
        scl_v = self.HHModel.scl # note: includes V->mV conversion
        I = (I0*1e-3)*scl_v/self.s # Note e-3 instead of 1e-6 because of scl_v
        return I
        
    def I_rate(self,c,sign,Vb):
        #kappa,_,_,Vt,_,_,_ = self.params
        I=0
        for i in range(len(Vb)):
            I += c[i] / (1 + np.exp(sign * self.kappa * (Vb[i] - self.vrange)  / self.Vt))
        return I
    
    # Only used for the initial fit
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
    
    def get_Vb(self):
        return self.Vmean + np.arange(start=-3,stop=4,step=1)*self.Vstep
    
    def initial_fit(self):
        """
        Initial fit to find optimal bias voltages
        """
        #_,_,_,_,_,_,V_ref = self.params
        
        # Fit Hodgkin-Huxley gating variables
        X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        
        # Initial parameter values
        C_a = np.array([])
        C_b = np.array([])
        for x in X:
            C_a = np.append(C_a,max(x.alpha(self.vrange))*np.ones(7)/7)
            C_b = np.append(C_b,max(x.beta(self.vrange))*np.ones(7)/7)
        Vmean = self.V_ref
        Vstep = (self.V_ref+self.HHModel.Ena/1e3 - self.V_ref+self.HHModel.Ek/1e3)/100
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
    
    # This should probably go into initial_fit method
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