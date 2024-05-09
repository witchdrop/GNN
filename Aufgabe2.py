import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Sigmoidfunktion

def deriv_sigmoid(x):
    return x * (1 - x) # Ableitung der Sigmoiden

def irprop_minus(weight, grad_old, grad_new, delta):

    for i,v in enumerate(grad_new):
        for j,e in enumerate(v):
            if grad_old[i,j]*grad_new[i,j]>0: # Lernrate vergrößern
                delta[i,j] = min(delta[i,j]*1.2, delta_max)
            if grad_old[i,j]*grad_new[i,j]<0: # Lernrate verkleinern
                delta[i,j] = max(delta[i,j]*0.5, delta_min)
                grad_new[i,j] = 0
    return weight-delta*np.sign(grad_new), delta, grad_new

# Das XOR-Problem, input [bias, x, y] und Target-Daten
inp    = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
target = np.array([[0], [1], [1], [0]])

# Die Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron

# Gewichte zufällig initialisieren (Mittelwert = 0)
w0 = np.random.random((inp_size, hid_size)) - 0.5
w1 = np.random.random((hid_size, out_size)) - 0.5

w0_delta = np.full((inp_size,hid_size),0.125)
w1_delta = np.full((hid_size,out_size),0.125)

w0_grad_old = np.zeros_like(w0)
w0_grad_new = np.zeros_like(w0) 

w1_grad_old = np.zeros_like(w1)
w1_grad_new = np.zeros_like(w1)

delta_max = 50 # Maximale Gewichtsänderung
delta_min = 0  # Minimale Gewichtsänderung

# Netzwerk trainieren
for i in range(50):

    # Vorwärtsaktivierung
    L0 = inp
    L1 = sigmoid(np.matmul(L0, w0))
    L1[0] = 1 # Bias-Neuron in der Hiddenschicht
    L2 = sigmoid(np.matmul(L1, w1))
    
    # Cost
    loss = (L2-target)**2
    cost = 0.25*(sum(loss))
    #print(cost)

    # Backpropagation
    L2_error = L2 - target 
    L2_delta = L2_error * deriv_sigmoid(L2)
    L1_error = np.matmul(L2_delta, w1.T)
    L1_delta = L1_error * deriv_sigmoid(L1)

    #Gradienten berechnen
    w0_grad_old = np.copy(w0_grad_new)
    w0_grad_new = np.matmul(L0.T, L1_delta)

    w1_grad_old = np.copy(w1_grad_new)
    w1_grad_new = np.matmul(L1.T, L2_delta)

    #Gewichte aktualisieren
    w0, w0_delta, w0_grad_new = irprop_minus(w0, w0_grad_old, w0_grad_new, w0_delta)
    w1, w1_delta, w1_grad_new = irprop_minus(w1, w1_grad_old, w1_grad_new, w1_delta)

# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, w0))
L1[0] = 1 # Bias-Neuron in der Hiddenschicht 
L2 = sigmoid(np.matmul(L1, w1))

#print(target)
print(L2)