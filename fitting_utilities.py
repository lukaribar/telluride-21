"""
Classes and methods for fitting NeuroDyn model
Important: all fitting is done assuming that the rest potential 
(or the threhsold potential) of the biophysical model is at 0V.
"""
from cb_models import HHActivation, HHInactivation
import numpy as np
from scipy.optimize import nnls, minimize, Bounds
import matplotlib.pyplot as plt
from cb_models import NeuroDynModel

class FitND:
    """
    Class takes the physical constants from a NeuroDyn model and provides
    methods for setting the parameters of the NeuroDyn chip to fit a Hodgkin-
    Huxley model that is passed as the argument.
    """
    def __init__(self, HHModel, vrange=[], capacitance_scaling = 1, 
                 I_voltage=150e-9, I_master = 200e-9, initial_fit=False):
        NDModel = NeuroDynModel(I_voltage = I_voltage, I_master = I_master,
                               capacitance_scaling = capacitance_scaling)
        self.NDModel = NDModel
        self.HHModel = HHModel
        
        # Set default fitting range
        if (vrange == []):
            vstart = HHModel.Ek
            vend   = HHModel.Ena/2 # this is arbitrary!!!
            vrange = np.arange(vstart, vend, 5e-4).T
        self.vrange = vrange
        # Set Vmean to middle of voltage range (or resting potential?)
        self.Vmean = (HHModel.Ek+HHModel.Ena)/2
        
        # Update dictionary with physical constants from NeuroDyn model
        params = NDModel.get_pars()
        self.__dict__.update(params)
        
        # Set initial currents (should be between ~50nA-400nA)
        self.I_voltage = I_voltage
        self.I_master = I_master
        
        # Initial fit to find optimal Vmean and I_voltage
        if (initial_fit):
            self.initial_fit()
        
        # Calculate base voltages
        self.Vstep = self.I_voltage*(1.85*1e6) / 3.5
        self.Vb = self.Vmean + np.arange(start=-3,stop=4,step=1)*self.Vstep
        
        # Maximal coefficients
        self.cmax = 0
        self.gmax = 0
        self.scl_t = 0
    
    # def fitHH(self, plot_alpha_beta = False, plot_inf_tau = False):
    #     """
    #     Fit the Hodgkin-Huxley model
    #     """
        
    #     Vrange = self.vrange
    #     c = []
    #     A = []
        
    #     X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        
    #     # Fit the variables and plot results
    #     for x in X:
    #         c_a,c_b,A_alpha,A_beta = self.fit_gating_variable(x)
    #         c.append([c_a, c_b])
    #         A.append([A_alpha, A_beta])
            
        
    
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
        for x in X:
            c_a,c_b,A_alpha,A_beta = self.fit_gating_variable(x)
            c.append([c_a, c_b])
            A.append([A_alpha, A_beta])
        
        # Calculate quantized plots
        g = [120e-3,36e-3,0.3e-3]
        E = [120e-3,-12e-3,10.6e-3]
        dIb,dg,dE,scl_t = self.quantize(c, g, E)
        ND = NeuroDynModel(dg, dE, dIb, self.Vmean, self.I_voltage, self.I_master)    
        X_ND = [ND.m, ND.h, ND.n]
        
        # PLOT
        for cj, Aj, x, x_ND, label in zip(c,A,X,X_ND,labels):
            alpha = np.dot(Aj[0],cj[0])
            beta = np.dot(Aj[1],cj[1])
                
            # Plot alpha and beta fits
            if (plot_alpha_beta):
                plt.figure()
                plt.title('α_'+label)
                plt.plot(Vrange,x.alpha(Vrange),label='Hodgkin-Huxley')
                plt.plot(Vrange,alpha,label='Fit')
                plt.plot(Vrange,x_ND.alpha(Vrange)*scl_t,label='Fit quantized')
                plt.legend()
            
                plt.figure()
                plt.title('β_'+label)
                plt.plot(Vrange,x.beta(Vrange),label='Hodgkin-Huxley')
                plt.plot(Vrange,beta,label='Fit')
                plt.plot(Vrange,x_ND.beta(Vrange)*scl_t,label='Fit quantized')
                plt.legend()
            
            # Plot xinf and tau fits
            if (plot_inf_tau):
                tau = 1/(alpha+beta)
                inf = alpha/(alpha+beta)
    
                plt.figure()
                plt.title('τ_'+label)
                plt.plot(Vrange,x.tau(Vrange),label='Hodgkin-Huxley')
                plt.plot(Vrange,tau,label='Fit')
                plt.plot(Vrange,x_ND.tau(Vrange)/scl_t,label='Fit quantized')
                plt.legend()
        
                plt.figure()
                plt.title(label+'_∞')
                plt.plot(Vrange,x.inf(Vrange),label='Hodgkin-Huxley')
                plt.plot(Vrange,inf,label='Fit')
                plt.plot(Vrange,x_ND.inf(Vrange),label='Fit quantized')
                plt.legend()
                
        return c
    
    
    def quantize(self,c,g,E):
        """
        Returns quantized sigmoid basis functions coefficients after transformation
        of the coefficients (c) into quantizated currents (dIb). 
        Also returns quantized maximal conductances (dg).
        To perform quantization, a suitable time scaling has to be found.
        This is done so as to jointly maximize the resolution of the conductances 
        and the coefficients.
        """
        # Find maximum coefficient
        cmax = np.array(c).max()
        if (cmax > self.cmax):
            self.cmax = cmax
            
        Imax = (1023*self.I_master/1024)
        
        # Find the scaling factor that maximizes coefficient resolution
        C_HH = 1e-6
        scl_t = self.cmax / (Imax / (self.C*self.Vt))
        if (scl_t > self.scl_t):
            self.scl_t = scl_t

        # Find maximum conductance
        gmax = np.array(g).max()
        if (gmax > self.gmax):
            self.gmax = gmax

        # Find the scaling factor that maximizes conductance resolution
        scl_t = self.gmax / (Imax * self.kappa / self.Vt) * self.C_ND / C_HH
        if (scl_t > self.scl_t):
            self.scl_t = scl_t

        # Recover the (quantized) rate currents from fitted coefficients
        self.s = self.scl_t * C_HH / self.C_ND
        Ib = []
        dIb = []
        dg = []
        for i in range(len(c)):
            # Exact (real numbers) current coefficients, before quantization
            i_a = c[i][0] * self.C * self.Vt / self.scl_t 
            i_b = c[i][1] * self.C * self.Vt / self.scl_t
            Ib.append([i_a, i_b])
            # Quantize current coefficients
            di_a = np.round(i_a*1024/self.I_master)
            di_b = np.round(i_b*1024/self.I_master)
            dIb.append([di_a, di_b])

        dIb = np.array(dIb)
                
        # Quantize conductances
        g_factor = (self.kappa / self.Vt) * (self.I_master / 1024)
        dg = np.round(np.array(g) / self.s / g_factor)
        
        # Quantize reversal potentials
        E_factor = (self.I_voltage / 1024) * self.Res
        scl_v = self.HHModel.scl_v
        dE = np.round((np.array(E)*scl_v - self.Vmean) / E_factor)
        
        # Check if all digital values are in the range [0, 1023]
        for d in dIb, dg, dE:
            if not((abs(d)<=1023).all()):
                print("The digital value is out of range:")
                print(d)
        
        return dIb,dg,dE,scl_t

    def fit_gating_variable(self, x):
        """
        Fit a single gating variable using non-negative least squares
        """
        Vrange = self.vrange
        Vb = self.Vb
        
        A_alpha = np.zeros((np.size(Vrange),7))
        A_beta = np.zeros((np.size(Vrange),7))
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
        
        c_a = nnls(A_alpha,b_alpha)[0]
        c_b = nnls(A_beta,b_beta)[0]
    
        return c_a, c_b, A_alpha, A_beta
        
    def convert_I(self, I0):
        scl_v = self.HHModel.scl_v
        I = I0*scl_v/self.s
        return I
        
    def I_rate(self,c,sign,Vb):
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
        I_voltage = Z[-1]
        Vstep = I_voltage*(1.85*1e6)/3.5
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
        Initial fit to find optimal Vmean and I_voltage
        """
        # Fit Hodgkin-Huxley gating variables
        X = [self.HHModel.m,self.HHModel.h,self.HHModel.n]
        
        # Initial parameter values
        C_a = np.array([])
        C_b = np.array([])
        for x in X:
            C_a = np.append(C_a,max(x.alpha(self.vrange))*np.ones(7)/7)
            C_b = np.append(C_b,max(x.beta(self.vrange))*np.ones(7)/7)
        Z0 = np.concatenate([C_a,C_b,np.array([self.Vmean, self.I_voltage])])
        
        # Bounds for Vmean and Ivoltage
        vmin = self.HHModel.Ek
        vmax = self.HHModel.Ena
        Imin = 50e-9
        Imax = 400e-9
        
        lowerbd = np.append(np.zeros(14*len(X)),np.array([vmin, Imin]))
        upperbd = np.append(np.ones(14*len(X))*np.inf,np.array([vmax, Imax]))
        bd = Bounds(lowerbd,upperbd)
        
        Z = minimize(lambda Z : self.cost(Z,X), Z0, bounds = bd)
        Z = Z.x
        
        # Check here if reversal potentials are covered by the voltage range
        #
        #
        
        self.Vmean = Z[-2]
        self.Ivoltage = Z[-1] 
        
        # Save the initial fit
        self.Z0 = Z
        return
    
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