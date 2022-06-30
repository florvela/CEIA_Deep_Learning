import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


class XOR:

    def __init__(self, n_epochs, lr):
        self.n_epochs = n_epochs
        self.lr = lr
        self.MSE = list()

        # weights and bias initialization
        # Layer 1
        self.layer_1_w1 = random.uniform(0, 1)
        self.layer_1_w2 = random.uniform(0, 1)
        self.layer_1_w3 = random.uniform(0, 1)
        self.layer_1_w4 = random.uniform(0, 1)
        self.layer_1_bias1 = random.uniform(0, 1)
        self.layer_1_bias2 = random.uniform(0, 1)

        # Layer 2 - output
        self.layer_2_w1 = random.uniform(0, 1)
        self.layer_2_w2 = random.uniform(0, 1)
        self.layer_2_bias = random.uniform(0, 1)

    def forward_propagation(self, X):
        x1, x2 = X[0], X[1]
        layer_1_z1 = self.layer_1_w1 * x1 + self.layer_1_w3 * x2 + self.layer_1_bias1
        layer_1_z2 = self.layer_1_w2 * x1 + self.layer_1_w4 * x2 + self.layer_1_bias2
        layer_1_z1_output = sigmoid(layer_1_z1)
        layer_1_z2_output = sigmoid(layer_1_z2)

        layer_2_z1 = self.layer_2_w1 * layer_1_z1_output + self.layer_2_w2 * layer_1_z2_output + self.layer_1_bias2
        layer_2_z1_output = layer_2_z1

        prediction = layer_2_z1_output
        return prediction, layer_1_z1_output, layer_1_z2_output

    def back_propagation(self, X, layer_1_z1_output, layer_1_z2_output, error):
        x1, x2 = X[0], X[1]

        d_prediction = - 2 * error
        error_layer_1_z1 = d_prediction * self.layer_2_w1
        error_layer_1_z2 = d_prediction * self.layer_2_w2
        d_layer_1_z1 = error_layer_1_z1 * sigmoid_deriv(layer_1_z1_output)
        d_layer_1_z2 = error_layer_1_z2 * sigmoid_deriv(layer_1_z2_output)

        # Updating Weights and Biases
        self.layer_2_w1 -= self.lr * d_prediction * layer_1_z1_output
        self.layer_2_w2 -= self.lr * d_prediction * layer_1_z2_output
        self.layer_2_bias -= d_prediction * self.lr
        self.layer_1_w1 -= self.lr * (d_layer_1_z1 * x1)
        self.layer_1_w3 -= self.lr * (d_layer_1_z1 * x2)
        self.layer_1_w2 -= self.lr * (d_layer_1_z2 * x1)
        self.layer_1_w4 -= self.lr * (d_layer_1_z2 * x2)
        self.layer_1_bias1 -= self.lr * (d_layer_1_z1)
        self.layer_1_bias2 -= self.lr * (d_layer_1_z2)


    def train(self, X, y):
        for _ in range(self.n_epochs):
            iteration_error = 0
            for i in range(len(X)):
                prediction, layer_1_z1_output, layer_1_z2_output = self.forward_propagation(X[i])
                error = y[i] - prediction
                iteration_error += error ** 2
                self.back_propagation(X[i], layer_1_z1_output, layer_1_z2_output, error)
            self.MSE.append(iteration_error/4)

    def predict(self, input):
        prediction, layer_1_z1_output, layer_1_z2_output = self.forward_propagation(input)

        if prediction.squeeze() >= 0.5:
            return 1
        else:
            return 0
