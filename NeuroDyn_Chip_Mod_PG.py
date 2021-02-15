import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.signal import argrelmax, argrelmin

"""
In this file there are three classes:
1) NeuroDyn_Neuron: This is the basis for the other two classes. It attempts to simulate a single neuron using the parameters and Hodgkin-Huxley
                    equations as on the NeuroDyn chip. It has a default setting which sets the parameter values to defaults used during testing,
                    which should produce spiking in the neuron with a constant input current of ~1e-9 A. There are also several methods used to
                    plot various I-V curves, but these mostly relate to my project. The attribute 'ode_solution' store the solution of the four
                    ODEs for the single neuron and can be easily used for plotting if one refers to the documentation on 'scipy.integrate.solve_ivp'
                    for the form of the output.

2) NeuroDyn_Chip: This simulates the entire NeuroDyn chip, with all 12 synaptic connections. In order to do this, four instances of a NeuroDyn_Neuron
                    are created and then the system is treated as a single dynamical system, with the ODE solver solving a system of 28 differential
                    equations to find the resulting membrane potentials for each neuron. This chip can perform all functions that Coupled_Neurons
                    can perform by simply setting the parameters of the other two neurons to zeros and 'switching them off'. One should be
                    careful when setting the 7-point spline regression parameters to 'switch-off' a gating variable, the alpha should be
                    set to a list of zeros, but the beta should not be, as this will cause an error in the ODE solver.
"""

##########################################################################################################################################

class NeuroDyn_Neuron:
    '''The NeuroDyn single neuron class'''

    def __init__(self, default=False):
        '''Initialise all the parameters as attributes of the class, setting default values from the example code as appropriate'''
        # There are a LOT of parameters to initialise so this may look daunting
        # Master parameters controlling scales, times, etc.
        self.V_ref = 0 # Unit V , 1 volt
        self.I_master = 33e-9 # Unit A
        self.I_voltage = 230e-9 # Unit A
        self.I_ref = 15e-9 # Unit A, 100nA
        self.K = (0.127) * 1e-6 # Factor for the injecting current

        # Membrane & gate capacitances
        self.C_m = 4e-12 # Unit f, 4pF
        self.C_gate = 5e-12 # Unit F, 5pF

        self.shift = 0 # Injection current shift

        # Scaling parameters (e.g. parameters that set the voltage scale, time scale..)
        self.time_len = 50e-3 # Unit second
        self.kappa = 0.7
        self.Vt = 26e-3 # Unit volt, 26mV
        self.Res = 1.63e6 # Unit ohm, 1.63M ohm

        # DAC parameters for conductances & reversal potentials, can be set to default values
        if default:
            self.g0 = np.array([400, 160, 12, 0]) # Now including the 3 possible synapse conductances
            self.e_rev = np.array([450, -250, -150, 0]) # Now including the reversal potentials for the synapses
        else:
            self.g0 = np.array([0, 0, 0, 0]) # Digital value of conductance, they need to be converted to analog
            self.e_rev = np.array([0, 0, 0, 0]) # Digital value of reversal potential

        # The analogue equivalent of the conductances, the bias voltage for the alpha/beta splines
        # and the maximum swing of the splines seems to be what is defined here
        # Convert digital conductance & reversal potential to analog
        self.g = self.g0 * (self.kappa / self.Vt) * (self.I_master / 1024)
        self.E_rev = self.e_rev * (self.I_voltage / 1024) * self.Res + self.V_ref

        # Bias voltages for the 7-point spline regression
        self.vBias = np.zeros(7) # Define the 7 bias voltages
        self.vHigh = self.V_ref + 0.426
        self.vLow = self.V_ref - 0.434
        self.I_factor = (self.vHigh - self.vLow) / 700e3
        self.vBias[0] = self.vLow + (self.I_factor * 50e3)
        for i in range(1, 7):
            self.vBias[i] = self.vBias[i-1] + (self.I_factor * 100e3)

        # Signs for the spline, including for a single synapse
        self.iSign = np.array([1, -1, -1, 1, 1, -1, 1, -1])

        # The spline-regression parameters for the alphas and betas, now including values for a single synapse
        if default:
            self.alpha_beta = np.array([[0, 0, 120, 400, 800, 1023, 1023],
                        [1023, 1023, 1023, 1023, 0, 0, 0],
                        [237, 5, 7, 6, 0, 0, 0],
                        [0, 0, 0, 0, 41, 25, 8],
                        [0, 0, 0, 0, 80, 40, 250],
                        [4, 0, 0, 10, 0, 0, 4],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        else:
            self.alpha_beta = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])

        self.iBias_alpha_beta = (self.I_master/1024) * self.alpha_beta
        self.g_f = 1 / (self.C_gate * self.Vt) # Gate factor

        # Create a placeholder for the ode solutions
        self.ode_solution = None
        self.n_0 = None # Placeholder for the initial n-value, needed for fast I-V curve

        # Define the default input current parameters
        self.const_val = 0.0 # Value of the constant input current
        self.lin_grad = 0.125e-6 # Gradient of the current ramp
        self.sVal = 1e-9 # Step value

        self.I_extA = lambda t: 0 + 0 * t

    ###########################################################################################################

    # The seven point spline regression for the alpha and betas for the gating variables
    def alpha_beta_spline(self, V, iBias_alpha_beta, vBias, iSign):
        kappa = 0.7
        Ut = 26e-3
        I = 0
        for spline in range(7):
            I += np.array(iBias_alpha_beta[spline]) / (1 + np.exp(kappa * (vBias[spline] - V) * iSign / Ut))
        return I

    ###########################################################################################################

    # Method for updating the spline coefficients
    def update_spline_coeffs(self, new_coeffs):
        self.alpha_beta = new_coeffs
        self.iBias_alpha_beta = (self.I_master/1024) * self.alpha_beta

    # Method for updating the conductance gains
    def update_conductances(self, new_cond):
        self.g0 = new_cond # Digital value of conductance, they need to be converted to analog
        # Convert to analogue value
        self.g = self.g0 * (self.kappa / self.Vt) * (self.I_master / 1024)
    
    # Method for updating the reversal potentials
    def update_reversal_pots(self, new_pots):
        self.e_rev = new_pots # Digital value of reversal potential
        # Convert to analogue value
        self.E_rev = self.e_rev * (self.I_voltage / 1024) * self.Res + self.V_ref

    ###########################################################################################################

    '''Now we define each of the opening and closing variable functions'''
    # m variable functions
    def am_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[0, :], self.vBias, self.iSign[0])
    def bm_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[1, :], self.vBias, self.iSign[1])
    # h variable functions
    def ah_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[2, :], self.vBias, self.iSign[2])
    def bh_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[3, :], self.vBias, self.iSign[3])
    # n variable functions
    def an_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[4, :], self.vBias, self.iSign[4])
    def bn_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[5, :], self.vBias, self.iSign[5])

    # Opening and closing functions for the synapse gating variables
    # Note!!!! The opening variable is a function of the presynaptic potential and the closing variable is a
    # function of the post-synaptic potential
    # N indicates whether its the first, second or third synapse
    def a_rij_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[6, :], self.vBias, self.iSign[6])
    def b_rij_v(self, V):
        return self.g_f * self.alpha_beta_spline(V, self.iBias_alpha_beta[7, :], self.vBias, self.iSign[7])   

    '''Now define the current functions'''
    # Sodium current
    def I_Na(self, V, m, h):
        return self.g[0] * m**3 * h * (V - self.E_rev[0])
    # Postassium current
    def I_K(self, V, n):
        return self.g[1] * n**4 * (V - self.E_rev[1])
    # Passive current
    def I_L(self, V):
        return self.g[2] * (V - self.E_rev[2])
    # Synaptic current
    def I_syn_ij(self, V, r):
        return self.g[3] * r * (V - self.E_rev[3])

    ###########################################################################################################

    '''Define methods holding the differentials & the ode solver'''
    def dXdtA(self, t, X):
        # The differential equations for solving
        return np.array([(self.I_extA(t) - self.I_Na(X[0], X[1], X[2]) - self.I_K(X[0], X[3]) - self.I_L(X[0])) / self.C_m,
                    self.am_v(X[0]) * (1 - X[1]) - self.bm_v(X[0]) * X[1],
                    self.ah_v(X[0]) * (1 - X[2]) - self.bh_v(X[0]) * X[2],
                    self.an_v(X[0]) * (1 - X[3]) - self.bn_v(X[0]) * X[3]]).T

    def solve_odes(self, dXdtA, init_conds, plot=False):
        # Solve the ODEs, we'll define them outside of the class
        # Inits should be a list of four values for [V_0, m_0, h_0, n_0]
        A = integ.solve_ivp(dXdtA, [0, self.time_len], init_conds)
        self.ode_solution = A
        self.n_0 = A.y[3, :][0]
        # Plot if desired
        if plot:
            fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, figsize=[18,10])
            fig.suptitle('Time-Domain Plots')

            ax1.plot(A.t, [self.I_extA(i) for i in A.t])
            ax1.set_xlabel('Time / s')
            ax1.set_ylabel('Current / A')
            ax1.set_title('Input Current')
            ax1.grid()

            ax2.plot(A.t, A.y[0, :], 'k', linewidth=0.5)
            ax2.set_xlabel('Time / s')
            ax2.set_ylabel('Voltage / V')
            ax2.set_title('Generated Membrane Potential')
            ax2.grid()

            ax3.plot(A.t, A.y[1, :], 'k', linewidth=0.5)
            ax3.set_xlabel('Time / s')
            ax3.set_ylabel('Value')
            ax3.set_title('m Gating Variable')
            ax3.grid()

            ax4.plot(A.t, A.y[3, :], 'k', linewidth=0.5)
            ax4.set_xlabel('Time / s')
            ax4.set_ylabel('Value')
            ax4.set_title('n Gating Variable')
            ax4.grid()

            plt.show()
        return A

    ###########################################################################################################

    'Define the steady state gating variable functions'

    def steady_states(self, V):
        # Returns all steady state gating variables for a given input voltage
        m_inf = self.am_v(V) / (self.am_v(V) + self.bm_v(V))
        h_inf = self.ah_v(V) / (self.ah_v(V) + self.bh_v(V))
        n_inf = self.an_v(V) / (self.an_v(V) + self.bn_v(V))

        return [m_inf, h_inf, n_inf]

    def get_timescales(self, V):
        # Returns the timescales currents
        gvs = self.steady_states(V)

        passive = self.g[2] * (V - self.E_rev[2])
        ff = self.g[0] * gvs[0]**3 * gvs[1] * (V - self.E_rev[0])
        sf = self.g[1] * gvs[2]**4 * (V - self.E_rev[1])

        return [ff, sf, passive]

    def plot_timescales(self, V):
        # Plots the I-V curves for the different timescales
        timescales = self.get_timescales(V)# Obtainf the steady-state gating variables

        # Obtain the turning points on the fast curve
        maxp = int(argrelmax(timescales[1])[0])
        minp = int(argrelmin(timescales[1])[0])

        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle('Timescale I-V Curves')

        axs[0].set_title('Passive')
        axs[0].plot(V, timescales[0], 'b')
        axs[0].set_xlabel('Voltage / V')
        axs[0].set_ylabel('Current / A')
        axs[0].grid()

        axs[1].set_title('Fast')
        axs[1].plot(V[0:maxp], timescales[1][0:maxp], 'b')
        axs[1].plot(V[maxp:minp], timescales[1][maxp:minp], 'r')
        axs[1].plot(V[minp:], timescales[1][minp:], 'b')
        axs[1].set_xlabel('Voltage / V')
        axs[1].set_ylabel('Current / A')
        axs[1].grid()

        axs[2].set_title('Slow')
        axs[2].plot(V[0:maxp], timescales[2][0:maxp], 'b')
        axs[2].plot(V[maxp:minp], timescales[2][maxp:minp], 'r')
        axs[2].plot(V[minp:], timescales[2][minp:], 'b')
        axs[2].set_xlabel('Voltage / V')
        axs[2].set_ylabel('Current / A')
        axs[2].grid()

        plt.show()
   
    ###########################################################################################################

    'A method to simulate a single neuron, plotting all desired features'

    def simulate_single_neuron(self, init_conds,  V, ode_sol=True, iv_curves=True):
        # This will call all the functions necessary to simulate the single neuron
        if ode_sol:
            A = self.solve_odes(self.dXdtA, init_conds, plot=True)
        else:
            A = self.solve_odes(self.dXdtA, init_conds, plot=False)

        # Now see if we want to plot the I-V curves
        if iv_curves:
            self.plot_timescales(V)
        else:
            return A
        
        return A

##########################################################################################################################################
##########################################################################################################################################

class NeuroDyn_Chip:

    # Initialisation of the class with all of the parameters for the chip, initialising four neurons
    def __init__(self, default_params=True):
        # First create the four NeuroDyn neurons
        # N1 and N3 will be paired to form an HCO
        self.N1 = NeuroDyn_Neuron(default=True) # N1 and N2 are to be paired to form a single bursting neuron
        self.N2 = NeuroDyn_Neuron(default=True)
        self.N3 = NeuroDyn_Neuron(default=True) # N3 and N4 are to be paired to form a single bursting neuron
        self.N4 = NeuroDyn_Neuron(default=True)

        if default_params:
            # The default neuron spline coefficients, we initialise each neuron as identical in default mode
            self.DEFAULT_SPLINE_COEFFS = np.array([[0, 0, 120, 400, 800, 1023, 1023],
                        [1023, 1023, 1023, 1023, 0, 0, 0],
                        [237, 5, 7, 6, 0, 0, 0],
                        [0, 0, 0, 0, 41, 25, 8],
                        [0, 0, 0, 0, 18, 15, 43],
                        [15, 0, 0, 15, 0, 0, 15],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
            # Default neuron conductances
            self.DEFAULT_CONDUCTANCES = np.array([400, 160, 12, 0])
            # Default neuron reversal potentials
            self.DEFAULT_REVERSAL_POTENTIALS = np.array([450, -100, -150, 0])
            
            # Set the neuron spline coefficients
            self.N1.update_spline_coeffs(self.DEFAULT_SPLINE_COEFFS)
            self.N2.update_spline_coeffs(self.DEFAULT_SPLINE_COEFFS)
            self.N3.update_spline_coeffs(self.DEFAULT_SPLINE_COEFFS)
            self.N4.update_spline_coeffs(self.DEFAULT_SPLINE_COEFFS)

            # The default neuron conductances, including three possible synapses
            self.N1.update_conductances(self.DEFAULT_CONDUCTANCES)
            self.N2.update_conductances(self.DEFAULT_CONDUCTANCES)
            self.N3.update_conductances(self.DEFAULT_CONDUCTANCES)
            self.N4.update_conductances(self.DEFAULT_CONDUCTANCES)

            # The default neuron reversal potentials
            self.N1.update_reversal_pots(self.DEFAULT_REVERSAL_POTENTIALS)
            self.N2.update_reversal_pots(self.DEFAULT_REVERSAL_POTENTIALS)
            self.N3.update_reversal_pots(self.DEFAULT_REVERSAL_POTENTIALS)
            self.N4.update_reversal_pots(self.DEFAULT_REVERSAL_POTENTIALS)
        
        # Require a fast negative, slow positive, slow negative and ultraslow positive conductance in N2 AND N4
        self.N1.iSign = np.array([1, -1, -1, 1, 1, -1, 1, -1])
        self.N2.iSign = np.array([1, -1, -1, 1, 1, -1, 1, -1])
        self.N3.iSign = np.array([1, -1, -1, 1, 1, -1, 1, -1])
        self.N4.iSign = np.array([1, -1, -1, 1, 1, -1, 1, -1])

        # Define the simulation duration for the system
        self.time_len = self.N1.time_len

        # The output of the ODE solution
        self.coupled_output = None

    # Define the system of ODEs solved on the chip. This is going to be complicated so all is defined beforehand
    def chip_ODE_system(self, t, X):
        '''Vi = Membrane Voltage
           m, h, n = Gating Variables
           rij = Synapse Gating Variable for connection from Nj in to Ni
           X[0] = V1,    X[1] = V2,    X[2] = m1,    X[3] = m2,    X[4] = m3,    
           X[5] = m4,    X[6] = h1,    X[7] = h2,    X[8] = h3,    X[9] = h4,   
           X[10] = n1,   X[11] = n2,   X[12] = n3,   X[13] = n4,   X[14] = r13,  
           X[15] = r31'''
        # Output is a system of 28 differential equations, one for each X variable described above
        return np.array([(self.N1.I_extA(t) - self.N1.I_Na(X[0], X[2], X[6]) - self.N1.I_K(X[0], X[10]) - self.N1.I_L(X[0])
                    - self.N1.I_syn_ij(X[0], X[14]) - self.N2.I_Na(X[0], X[3], X[7]) - self.N2.I_K(X[0], X[11])
                    - self.N2.I_L(X[0])) / self.N1.C_m,
                    (self.N3.I_extA(t) - self.N3.I_Na(X[1], X[4], X[8]) - self.N3.I_K(X[1], X[12]) - self.N3.I_L(X[1]) 
                    - self.N3.I_syn_ij(X[1], X[15]) - self.N4.I_Na(X[1], X[5], X[9]) - self.N4.I_K(X[1], X[13])
                    - self.N4.I_L(X[1])) / self.N3.C_m,
                    self.N1.am_v(X[0]) * (1 - X[2]) - self.N1.bm_v(X[0]) * X[2],
                    self.N2.am_v(X[0]) * (1 - X[3]) - self.N2.bm_v(X[0]) * X[3],
                    self.N3.am_v(X[1]) * (1 - X[4]) - self.N3.bm_v(X[1]) * X[4],
                    self.N4.am_v(X[1]) * (1 - X[5]) - self.N4.bm_v(X[1]) * X[5],
                    self.N1.ah_v(X[0]) * (1 - X[6]) - self.N1.bh_v(X[0]) * X[6],
                    self.N2.ah_v(X[0]) * (1 - X[7]) - self.N2.bh_v(X[0]) * X[7],
                    self.N3.ah_v(X[1]) * (1 - X[8]) - self.N3.bh_v(X[1]) * X[8],
                    self.N4.ah_v(X[1]) * (1 - X[9]) - self.N4.bh_v(X[1]) * X[9],
                    self.N1.an_v(X[0]) * (1 - X[10]) - self.N1.bn_v(X[0]) * X[10],
                    self.N2.an_v(X[0]) * (1 - X[11]) - self.N2.bn_v(X[0]) * X[11],
                    self.N3.an_v(X[1]) * (1 - X[12]) - self.N3.bn_v(X[1]) * X[12],
                    self.N4.an_v(X[1]) * (1 - X[13]) - self.N4.bn_v(X[1]) * X[13],
                    self.N1.a_rij_v(X[1]) * (1 - X[14]) - self.N1.b_rij_v(X[0]) * X[14],
                    self.N3.a_rij_v(X[0]) * (1 - X[15]) - self.N3.b_rij_v(X[1]) * X[15]]).T

    # Use an ODE Solver to solve the system of equations
    def simulate_chip(self, odeSystem, conds, plot_Vs=True, show=True):
        # This will hopefully solve the system of above equations given a set of initial conditions, HOPEFULLY
        A = integ.solve_ivp(odeSystem, [0, self.time_len], conds, method='BDF')
        self.coupled_output = A
        # Need to set the neuron initial n values to use the approximation 
        self.N1.n_0 = A.y[10,:][0]
        self.N2.n_0 = A.y[11,:][0]
        self.N3.n_0 = A.y[12,:][0]
        self.N4.n_0 = A.y[13,:][0]
        # # Get the voltages in a nice list
        # voltages = np.array([A.y[0,:], A.y[1,:]])
        # # Get the neurons too
        # neurons = np.array([[self.N1, self.N2], [self.N3, self.N4]])

        return A

##########################################################################################################################################
'''Use here to test out the class

# Create an instance of the chip
neurodyn = NeuroDyn_Chip(True)

# Define the initial conditions
conds = np.zeros(28)

# Define the input current to each chip
neurodyn.N1.I_extA = lambda t: 0 + 0 * t
neurodyn.N2.I_extA = lambda t: 0 + 0 * t
neurodyn.N3.I_extA = lambda t: 0 + 0 * t
neurodyn.N4.I_extA = lambda t: 0 + 0 * t

# Simulate the chip by solving the system of ODEs
neurodyn.simulate_chip(neurodyn.chip_ODE_system, conds, plot_Vs=True, show=True)'''
##########################################################################################################################################