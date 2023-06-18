from dataclasses import dataclass
from typing import Callable

import numpy as np
from funcs import *
# from matplotlib import pyplot as plt


class Network:  # TODO: need to try to add batch-normalisation
    def __init__(self, loss=mse, prime_loss=prime_mse):
        self.loss = loss
        self.prime_loss = prime_loss
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def get_data(self):
        return [self.convert_data(layer) for layer in self.layers]

    def convert_data(self, layer):
        return np.round(layer.weights, 15).tolist(), np.round(layer.biases, 15).tolist()

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
            # err = 0
            for i in range(len(x_train)):
                output = self.forward_prop(x_train[i])  # forward propagation

                error = self.prime_loss(output, y_train[i])  # error

                self.backward_prop(error, learning_rate)  # backward propagation

                # err += self.loss(output, y_train[i])
            # err /= len(x_train)
            # if draw_map and epoch != 0:
            #     plt.plot((epoch - 1, epoch), (last_err, err), color="blue")
            #     plt.pause(0.05)
            # if draw_map:
            #     last_err = err
        # print("\nError:", err)

    def predict(self, x_test, y_test, method=the_surest):  # TODO: need to add the ability to look into what number failed to guess (like, to an actual image)
        err = 0
        count = 0
        for i in range(len(x_test)):
            sample = x_test[i]
            output = self.forward_prop(sample)[0]
            err += self.loss(output, y_test[i])
            count += method(output, y_test[i])  # method can be changed from the_surest to rounding. For more details, see in funcs.py
            if (i + 1) % 10 == 0:
                print(f"\rSamples: {(i+1)}/{len(x_test)}, Error: {err / (i+1)}, Count: {count}/{i+1}", end="")


@dataclass
class Layer:
    input_size: int
    output_size: int
    weights: np.ndarray
    biases: np.ndarray
    input: np.ndarray = None
    output: np.ndarray = None
    activation: Callable = sigmoid
    derivative: Callable = der_sigmoid


def init_weights(input_size, output_size):
    return np.random.rand(input_size, output_size) - 0.5


def init_biases(output_size):
    return np.random.rand(1, output_size) - 0.5

