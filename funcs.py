import numpy as np


def mse(got, wanted):
    return np.mean(np.power(wanted-got, 2))


def prime_mse(got, wanted):
    return 2 * (got - wanted) / wanted.size


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def index_of_max(array):
    return np.argpartition(array, -1)[-1]


def the_surest(got, wanted):  # method of finding the surest (on average performs better)
    return index_of_max(got) == index_of_max(wanted)


def rounding(got, wanted):  # method of rounding (on average performs worse)
    return np.round(got).tolist() == np.round(wanted).tolist()
