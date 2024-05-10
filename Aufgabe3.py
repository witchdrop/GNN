import numpy as np
import gzip
import pickle
import sys
import matplotlib.pyplot as plt

def setup():
    #Daten laden
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, _), (x_test, _) = data
    # Normalisieren der Pixelwerte auf [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    #eindimensionale Darstellung pro Datensatz
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test


def sigmoid(summe): # Transferfunktion
    return 1.0/(1.0+np.exp(-1.0*summe))


def activate(v,weights,c):
    return sigmoid(np.matmul(v, weights) + hid_bias)


def reactivate(h,weights,b):
    return sigmoid(np.matmul(h,weights.T) + vis_bias)



def train(x_train,weights,hid_bias,vis_bias,epochs,dataset_size):
    for e in range(epochs):
        v0 = x_train[:dataset_size]
        h0 = activate(v0,weights,hid_bias)
        v1 = reactivate(h0,weights,vis_bias)
        h1 = activate(v1,weights,vis_bias)
        for i,data in enumerate(v0): #Gewicht anpassen pro Datensatz
            print(f"epoch: {e+1}/{epochs} | data: {i+1}/{len(v0)}")
            delta = np.outer(v0[i],h0[i]) - np.outer(v1[i],h1[i])
            weights += learningRate * delta
        vis_bias = learningRate * (v0-v1)
        hid_bias = learningRate * (h0-h1)
    return weights


def test(x_test,weights,hid_bias,vis_bias,dataset_size):
    v0 = x_test[0:dataset_size]
    h0 = activate(v0,weights,hid_bias)
    v1 = reactivate(h0,weights,vis_bias)
    return v1


def plot(num_pictures,x_test,v1):
    plt.figure(figsize=(20, 4))
    for i in range(num_pictures):
        # Original Bild
        ax = plt.subplot(2, num_pictures, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Rekonstruiertes Bild
        ax = plt.subplot(2, num_pictures, i + 1 + num_pictures)
        plt.imshow(v1[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


vis_size = 784 # Visible Neuronen
hid_size = 100 # Hidden Neuronen
weights = np.random.random((vis_size, hid_size)) - 0.5
learningRate = 0.1
hid_bias = np.full((hid_size),0.125) # Bias pro Neuron der Hidden Layer
vis_bias = np.full((vis_size),0.125) # Bias pro Neuron der Visible Layer
epochs = 1500
num_pictures = 10
dataset_size = 25
x_train, x_test = setup()

weights_trained = train(x_train,weights,hid_bias,vis_bias,epochs,dataset_size)
output = test(x_train,weights_trained,hid_bias,vis_bias,dataset_size)
plot(num_pictures,x_train,output)
