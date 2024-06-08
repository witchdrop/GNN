import numpy as np
import matplotlib.pyplot as plt

bias = [-3.37, 0.125]
weights = np.array([
    [-4,1.5],
    [-1.5,0]])

def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) -1


def activate(inp):
    return np.matmul(inp, weights) + bias

# lists to store outputs of neuron 1 and neuron 2 in
o1 = []
o2 = []

state = [0, 0] # initial hidden state
steps = 50

for i in range(steps):
    state = tanh(activate(state))
    o1.append(state[0])
    o2.append(state[1])

# Plot with timesteps on x-axis and neuron outputs on y-axis
plt.plot(range(steps),o1)
plt.plot(range(steps),o2)
plt.xlabel('Timesteps')
plt.ylabel('Neuron Output')
plt.legend(["o1", "o2"])
plt.title('Hidden State Changes over Time in RNN')
plt.show()
