from utils import relu, softmax


# Forward propagation (Inference of the Neural Network )
def predict(inputs, weights, biases):
    deps = len(weights)

    x = inputs
    for i in range(deps):
        t = x @ weights[i] + biases[i]
        x = relu(t) if i < deps - 1 else t

    return softmax(x)
