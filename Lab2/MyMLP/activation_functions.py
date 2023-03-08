import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - x ** 2
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return 1. * (x > 0)
    return x * (x > 0)


def softmax(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)
