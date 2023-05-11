import numpy as np


def mse(got, wanted):
    return np.mean(np.power(wanted-got, 2))


def prime_mse(got, wanted):
    return 2 * (got - wanted) / wanted.size


def sigmoida(x):
    return 1 / (1 + np.exp(-x))


def sigmoida_prime(x):
    return sigmoida(x) * (1 - sigmoida(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2