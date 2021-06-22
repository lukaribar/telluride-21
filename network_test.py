from cb_models import HHModel, NeuronalNetwork, AMPASynapse
import matplotlib.pyplot as plt

neuron1 = HHModel()
neuron2 = HHModel()
neurons = [neuron1, neuron2]

syn = AMPASynapse(0.1)

syns = [[None, None], [[syn], None]]

network = NeuronalNetwork(neurons, syns=syns)

T = 200
trange = (0, T)

I1 = 10
I2 = 0
Iapp = lambda t: [I1, I2]

x0 = [0,0,0,0]*2 + [0]

sol = network.simulate(trange, x0, Iapp)

plt.plot(sol.t, sol.y[0], sol.t, sol.y[4])