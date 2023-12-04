import numpy as np


# Activation function
def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

