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
        layer_2_z1_output = sigmoid(layer_2_z1)

        prediction = layer_2_z1_output
        return prediction, layer_1_z1_output, layer_1_z2_output

    def back_propagation(self, X, prediction, layer_1_z1_output, layer_1_z2_output, error):
        # print(X)
        x1, x2 = X[0], X[1]
        d_prediction = error * sigmoid_deriv(prediction)

        error_layer_1_z1 = d_prediction * self.layer_2_w1
        error_layer_1_z2 = d_prediction * self.layer_2_w2
        d_layer_1_z1 = error_layer_1_z1 * sigmoid_deriv(layer_1_z1_output)
        d_layer_1_z2 = error_layer_1_z2 * sigmoid_deriv(layer_1_z2_output)

        # Updating Weights and Biases
        self.layer_2_w1 += layer_1_z1_output * d_prediction * self.lr
        self.layer_2_w2 += layer_1_z2_output * d_prediction * self.lr
        self.layer_2_bias += d_prediction * self.lr
        self.layer_1_w1 += x1 * d_layer_1_z1 * self.lr
        self.layer_1_w3 += x2 * d_layer_1_z1 * self.lr
        self.layer_1_w2 += x1 * d_layer_1_z2 * self.lr
        self.layer_1_w4 += x2 * d_layer_1_z2 * self.lr
        self.layer_1_bias1 += d_layer_1_z1 * self.lr
        self.layer_1_bias2 += d_layer_1_z2 * self.lr


    def train(self, X, y):
        for _ in range(self.n_epochs):
            for i in range(len(X)):
                prediction, layer_1_z1_output, layer_1_z2_output = self.forward_propagation(X[i])
                error = y[i] - prediction
                error_2 = error ** 2
                self.back_propagation(X[i], prediction, layer_1_z1_output, layer_1_z2_output, error)

    def predict(self, input):
        prediction, layer_1_z1_output, layer_1_z2_output = self.forward_propagation(input)
        return prediction
        # if prediction >= 0.5:
        #     return 1
        # else:
        #     return 0


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

xor_obj = XOR(n_epochs=100000, lr=0.1)
xor_obj.train(X, y)

################ tests ################
tests = [[1,0],[0,1],[1,1],[0,0]]
for t in tests:
    print(xor_obj.predict(t))

# assert xor_obj.predict([1, 0]) == 1
# assert xor_obj.predict([0, 0]) == 0
# assert xor_obj.predict([0, 1]) == 1
# assert xor_obj.predict([1, 1]) == 0
