import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


class XOR:

    def __init__(self, n_epochs, lr):
        self.n_epochs = n_epochs
        self.lr = lr

        # weights and bias initialization
        # Layer 1
        self.layer_1_weights = np.random.uniform(size=(2, 2))
        self.layer_1_bias = np.random.uniform(size=(1, 2))
        # Layer 2 - output
        self.layer_2_weights = np.random.uniform(size=(2, 1))
        self.layer_2_bias = np.random.uniform(size=(1, 1))

    def forward_propagation(self, X):
        layer_1_activation = np.dot(X, self.layer_1_weights)
        layer_1_activation += self.layer_1_bias
        self.layer_1_output = sigmoid(layer_1_activation)

        layer_2_activation = np.dot(self.layer_1_output, self.layer_2_weights)
        layer_2_activation += self.layer_2_bias
        prediction = sigmoid(layer_2_activation)

        return prediction

    def back_propagation(self, X, y, prediction):
        error = y - prediction
        d_prediction = error * sigmoid_deriv(prediction)

        error_layer_1 = d_prediction.dot(self.layer_2_weights.T)
        d_layer_1 = error_layer_1 * sigmoid_deriv(self.layer_1_output)

        # Updating Weights and Biases
        self.layer_2_weights += self.layer_1_output.T.dot(d_prediction) * self.lr
        self.layer_2_bias += np.sum(d_prediction, axis=0, keepdims=True) * self.lr
        self.layer_1_weights += X.T.dot(d_layer_1) * self.lr
        self.layer_1_bias += np.sum(d_layer_1, axis=0, keepdims=True) * self.lr

        # print(self.layer_2_bias.shape, self.layer_1_bias.shape) # (1, 1) (1, 2)

    def train(self, X, y):
        for _ in range(self.n_epochs):
            prediction = self.forward_propagation(X)
            self.back_propagation(X, y, prediction)

    def predict(self, input):
        prediction = self.forward_propagation(input)

        if prediction >= 0.5:
            return 1
        else:
            return 0
