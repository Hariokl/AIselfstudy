import numpy as np


class Layer:
    def __init__(self):
        self.input_data = None
        self.output_data = None


class FLayer(Layer):
    def __init__(self, input_size, output_size, activ, prim_activ):
        # activation function and it's derivative
        self.activ = activ
        self.prim_activ = prim_activ

        # random weights and biases
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input_data = input_data  # A[n-1]
        self.output_data = np.dot(input_data, self.weights) + self.bias  # Z[n]
        return self.activ(self.output_data)  # A[n]

    def backward_propagation(self, error, learning_rate):
        error = self.prim_activ(self.output_data) * error
        input_error = np.dot(error, self.weights.T)
        weights_error = np.dot(self.input_data.T, error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * error

        return input_error


# class FCLayer(Layer):
#     def __init__(self, input_size, output_size):
#         self.weights = np.random.rand(input_size, output_size) - 0.5
#         self.bias = np.random.rand(1, output_size) - 0.5
#
#     def forward_propagation(self, input_data):
#         self.input_data = input_data
#         self.output_data = np.dot(input_data, self.weights) + self.bias
#         return self.output_data
#
#     def backward_propagation(self, error, learning_rate):
#         input_error = np.dot(error, self.weights.T)
#         weights_error = np.dot(self.input_data.T, error)
#
#         self.weights -= learning_rate * weights_error
#         self.bias -= learning_rate * error
#
#         return input_error
#
#
# class ActivationLayer(Layer):
#     def __init__(self, activation, dev_activation):
#         self.activation = activation
#         self.dev_activation = dev_activation
#
#     def forward_propagation(self, input_data):
#         self.input_data = input_data
#         self.output_data = self.activation(input_data)
#         return self.output_data
#
#     def backward_propagation(self, error, learning_rate):
#         return self.dev_activation(self.input_data) * error
