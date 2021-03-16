import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

class GUI:
    """
    Graphical user interface class with methods for plotting the IV curves and
    simulation results along with methods for adding sliders and buttons for
    changing the neuronal parameters.
    
    args:
        system: Neuron or Network class to be simulated
    """  
    def __init__(self, system):              
        self.system = system # associate GUI with a neuron or a network
        
        # Initial parameters
        self.i0 = 0
        self.t_start_list = [50, 100, 150]
        self.tau_list = [1, 1, 1]
        self.mag_list = [2, 4, 6]
        self.alpha_list = []
        
        # Define inputs
        for t_start, tau, mag in zip(self.t_start_list, self.tau_list,
                                     self.mag_list):
            self.alpha_list.append(self.Alpha(t_start, tau, mag))
        
        # Parameters for plots
        self.V_min = -20
        self.V_max = 120
        self.i_min = -1
        self.i_max = 10
        self.t_max = 200
        self.t = np.arange(0, self.t_max, 0.1)
        
        # Create empty plot
        plt.close("all")
        self.fig = plt.figure()
        
        # Add voltage plot
        self.ax_out = self.fig.add_subplot(2, 1, 1)
        self.ax_out.set_position([0.1, 0.75, 0.8, 0.2])
        self.run(0)
        
        # Add Iapp plot
        self.ax_in = self.fig.add_subplot(2, 1, 2)
        self.ax_in.set_position([0.1, 0.55, 0.8, 0.1])
        self.update_input()
        
        # Add sliders for magnitudes
        self.s1 = self.add_slider("Input 1", [0.1, 0.45, 0.3, 0.03], 0, 10,
                                  self.mag_list[0], self.alpha_list[0].set_mag)
        self.s2 = self.add_slider("Input 2", [0.1, 0.4, 0.3, 0.03], 0, 10,
                                  self.mag_list[1], self.alpha_list[1].set_mag)
        self.s3 = self.add_slider("Input 3", [0.1, 0.35, 0.3, 0.03], 0, 10,
                                  self.mag_list[2], self.alpha_list[2].set_mag)
        
        # Add sliders for taus
        self.s4 = self.add_slider("", [0.6, 0.45, 0.3, 0.03], 0.1, 5,
                                  self.tau_list[0], self.alpha_list[0].set_tau)
        self.s5 = self.add_slider("", [0.6, 0.4, 0.3, 0.03], 0.1, 5,
                                  self.tau_list[1], self.alpha_list[1].set_tau)
        self.s6 = self.add_slider("", [0.6, 0.35, 0.3, 0.03], 0.1, 5,
                                  self.tau_list[2], self.alpha_list[2].set_tau)
        
        # Add sliders for maximal conductances
        self.s7 = self.add_slider("$g_{Na}$", [0.1, 0.25, 0.3, 0.03], 0, 200,
                                  self.system.gna, self.update_gna)
        self.s8 = self.add_slider("$g_{K}$", [0.1, 0.2, 0.3, 0.03], 0, 100,
                                  self.system.gk, self.update_gk)
        self.s9 = self.add_slider("$g_{l}$", [0.1, 0.15, 0.3, 0.03], 0, 2,
                                  self.system.gl, self.update_gl)
        
        # Add run button
        self.b = self.add_button("Run", [0.8, 0.02, 0.1, 0.03], self.run)
    
    class Alpha:
        def __init__(self, t_start, tau, mag):
            self.t_start = t_start
            self.tau = tau
            self.mag = mag
        
        def set_mag(self, mag):
            self.mag = mag
            
        def set_tau(self, tau):
            self.tau = tau
        
        def out(self, t):
            t1 = (t - self.t_start) / self.tau
            t2 = (t1>0) * t1
            I_out = self.mag * t2*np.exp(1 - t2)
            return I_out    
    
    def update_input(self):
        I = self.i_app(self.t)
        self.ax_in.cla()
        self.ax_in.set_xlim((0, self.t_max))
        self.ax_in.set_ylim((self.i_min, self.i_max))
        self.ax_in.set_xlabel('Time')
        self.ax_in.set_ylabel('Iapp')
        self.ax_in.plot(self.t, I)
    
    def i_app(self, t):
        I_out = self.i0
        for alpha in self.alpha_list:
            I_out += alpha.out(t)
        return I_out

    def update_gna(self, val):
        self.system.gna = val
        
    def update_gk(self, val):
        self.system.gk = val
        
    def update_gl(self, val):
        self.system.gl = val
        
    def update_val(self, val, update_method):
        update_method(val)
        self.update_input()
        
    def add_slider(self, name, coords, val_min, val_max, val_init,
                   update_method):
        ax = plt.axes(coords)
        slider = Slider(ax, name, val_min, val_max, valinit = val_init)
        slider.on_changed(lambda val: self.update_val(val, update_method))
        return slider
            
    def add_button(self, name, coords, on_press_method):
        ax = plt.axes(coords)
        button = Button(ax, name)
        button.on_clicked(on_press_method)
        return button
        
    def add_label(self, x, y, text):
        plt.figtext(x, y, text, horizontalalignment = 'center')

    def run(self, event):
        trange = (0, self.t_max)
        x0 = [0, 0, 0, 0]
        sol = self.system.simulate(trange, x0, self.i_app)
        
        self.ax_out.cla()
        self.ax_out.set_xlim((0, self.t_max))
        self.ax_out.set_ylim((self.V_min, self.V_max))
        self.ax_out.set_ylabel('V')
        self.ax_out.plot(sol.t, sol.y[0])
        