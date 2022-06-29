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
        self.Z1_weights = np.random.uniform(size=(2, 2))
        self.Z1_bias = np.random.uniform(size=(1, 2))
        # Layer 2 - output
        self.Z2_weights = np.random.uniform(size=(2, 1))
        self.Z2_bias = np.random.uniform(size=(1, 1))

    def forward_propagation(self, X):
        # Forward Propagation
        Z1_activation = np.dot(X, self.Z1_weights)
        Z1_activation += self.Z1_bias
        self.Z1_output = sigmoid(Z1_activation)

        Z2_activation = np.dot(self.Z1_output, self.Z2_weights)
        Z2_activation += self.Z2_bias
        prediction = sigmoid(Z2_activation)

        return prediction

    def back_propagation(self, y, prediction):
        error = y - prediction
        d_prediction = error * sigmoid_deriv(prediction)

        error_Z1 = d_prediction.dot(self.Z2_weights.T)
        d_Z1 = error_Z1 * sigmoid_deriv(self.Z1_output)

        # Updating Weights and Biases
        self.Z2_weights += self.Z1_output.T.dot(d_prediction) * self.lr
        self.Z2_bias += np.sum(d_prediction, axis=0, keepdims=True) * self.lr
        self.Z1_weights += X.T.dot(d_Z1) * self.lr
        self.Z1_bias += np.sum(d_Z1, axis=0, keepdims=True) * self.lr

    def train(self, X, y):
        for _ in range(self.n_epochs):
            prediction = self.forward_propagation(X)
            self.back_propagation(y, prediction)

    def predict(self, input):
        prediction = self.forward_propagation(input)

        if prediction >= 0.5:
            return 1
        else:
            return 0

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

xor_obj = XOR(n_epochs=10000, lr=0.1)
xor_obj.train(X, y)


test = np.array([[1, 0]])
assert xor_obj.predict(test) == 1
test = np.array([[0, 0]])
assert xor_obj.predict(test) == 0
test = np.array([[0, 1]])
assert xor_obj.predict(test) == 1
test = np.array([[1, 1]])
assert xor_obj.predict(test) == 0
