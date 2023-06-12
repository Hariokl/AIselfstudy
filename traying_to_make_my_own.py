import numpy as np
import numba as nb
from funcs import *
from matplotlib import pyplot as plt


class Network:
    def __init__(self, layers, loss=mse, prime_loss=prime_mse):
        self.layers = layers
        self.loss = loss
        self.prime_loss = prime_loss

    def get_data(self):
        return [layer.get_data() for layer in self.layers]

    def forward_prop(self, sample):
        output = sample
        for layer in self.layers:
            layer.input = output
            layer.output = np.dot(output, layer.weights) + layer.biases
            output = layer.activation(layer.output)
        return output

    def backward_prop(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = error * layer.derivative(layer.output)
            layer.biases -= learning_rate * error
            layer.weights -= learning_rate * error * layer.input.T
            error = np.dot(error, layer.weights.T)

    def fit(self, x_train, y_train, epochs, learning_rate, draw_map=True):
        for epoch in range(epochs):
            print(f"\rEpoch: {epoch+1}/{epochs}", end="")
            err = 0
            for i in range(len(x_train)):
                output = self.forward_prop(x_train[i])  # forward propagation

                error = self.prime_loss(output, y_train[i])  # error

                self.backward_prop(error, learning_rate)  # backward propagation

                err += self.loss(output, y_train[i])
            err /= len(x_train)
            if draw_map and epoch != 0:
                plt.plot((epoch - 1, epoch), (last_err, err), color="blue")
                plt.pause(0.05)
            if draw_map:
                last_err = err
        print("\nError:", err)

    def predict(self, x_test, y_test):
        err = 0
        count = 0
        for i in range(len(x_test)):
            sample = x_test[i]
            output = self.forward_prop(sample)[0]
            err += self.loss(output, y_test[i])
            count += self.index_of_max(output) == self.index_of_max(y_test[i])
            if i % 1 == 0:
                print(f"\rSamples: {(i+1)}/{len(x_test)}, Error: {err / (i+1)}, Count: {count}/{i+1}", end="")

    def index_of_max(self, array):
        return np.argpartition(array, -1)[-1]


class Layer:
    __slots__ = ("input_size", "output_size", "weights", "biases", "input", "output", "activation", "derivative")

    def __init__(self, input_size, output_size, activation=sigmoid, derivative=der_sigmoid, weights=None, biases=None):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        self.activation = activation
        self.derivative = derivative
        self.init_weights(weights)
        self.init_biases(biases)

    def init_weights(self, weights=None):
        self.weights = weights
        if weights is None:
            self.weights = np.random.rand(self.input_size, self.output_size) - 0.5

    def init_biases(self, biases=None):
        self.biases = biases
        if biases is None:
            self.biases = np.random.rand(1, self.output_size) - 0.5

    def get_data(self):
        return np.round(self.weights, 15).tolist(), np.round(self.biases, 15).tolist()
