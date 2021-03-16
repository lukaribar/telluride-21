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
    
    kwargs:
        vmin, vmax, dv: voltage range of the IV curves
        i0: initial applied current
    """  
    def __init__(self, system):              
        self.system = system # associate GUI with a neuron or a network
        
        # Initial parameters
        self.i0 = 0
        self.t_start_list = [30, 50, 70]
        self.tau_list = [1, 1, 1]
        self.mag_list = [1, 2, 3]
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
        self.t_max = 100
        
        # Create empty plot
        plt.close("all")
        self.fig = plt.figure()
        
        # Add simulation plot
        self.ax_out = self.fig.add_subplot(2, 1, 1)
        self.ax_out.set_position([0.1, 0.45, 0.8, 0.2])
        self.ax_out.set_xlim((0, self.t_max))
        self.ax_out.set_ylim((self.V_min, self.V_max))
        self.ax_out.set_xlabel('Time')
        self.ax_out.set_ylabel('V')
        
        # Add input current plot
        self.ax_in = self.fig.add_subplot(2, 1, 2)
        self.ax_in.set_position([0.1, 0.25, 0.8, 0.2])
        self.ax_in.set_ylim((self.i_min, self.i_max))
        self.ax_out.set_xlabel('Time')
        self.ax_out.set_ylabel('Iapp')
    
    class Alpha:
        def __init__(self, t_start, tau, mag):
            self.t_start = t_start
            self.tau = tau
            self.mag = mag
            
        def out(self, t):
            t1 = (t - self.t_start) / self.tau
            I_out = (t1 <= self.t_start) * 0
            + (t1 > self.t_start) * self.mag * t1*np.exp(1 - t1)
            return I_out    
    
    def i_app(self, t):
        I_out = self.i0
        for alpha in self.alpha_list:
            I_out += alpha.out(t)
        return I_out
            
    def update_val(self, val, update_method):
        update_method(val)
        self.update_IV_curves()
        
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
         