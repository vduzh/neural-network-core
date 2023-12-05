import numpy as np


# Activation function
def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    print('y_full', y_full)
    y_full[0, y] = 1
    return y_full
