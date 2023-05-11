import numpy as np

from network import *
from funcs import *
from layers import *


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = NeuralNetwork()
net.add(FLayer(2, 3, sigmoida, sigmoida_prime))
net.add(FLayer(3, 1, sigmoida, sigmoida_prime))

net.use(mse, prime_mse)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)