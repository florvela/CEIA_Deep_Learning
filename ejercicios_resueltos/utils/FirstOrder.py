import numpy as np
import random


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    return mini_batches


class FirstOrder:

    def __init__(self):
        self.MSE = list()
        self.batch_size = 0
        self.lr = 0
        self.S = 0.01

        # weights and bias initialization
        self.w1 = random.uniform(0, 1)
        self.w2 = random.uniform(0, 1)
        self.bias = random.uniform(0, 1)

        # velocities initialization
        self.w1_velocity = random.uniform(0, 1)
        self.w2_velocity = random.uniform(0, 1)
        self.bias_velocity = random.uniform(0, 1)

    def forward_propagation(self, X):
        x1, x2 = X[:,0], X[:,1]
        return self.w1 * x1 + self.w2 * x2 + self.bias

    def back_propagation(self, X, error):
        x1, x2 = X[:,0], X[:,1]

        # calculando derivadas
        d_prediction = - 2 * error
        w1_loss = (d_prediction * x1).sum(axis=0)
        w2_loss = (d_prediction * x2).sum(axis=0)
        bias_loss = d_prediction.sum(axis=0)

        # update de velocity y de los pesos
        self.w1_velocity = self.S * self.w1_velocity + self.lr * w1_loss / self.batch_size
        self.w2_velocity = self.S * self.w2_velocity + self.lr * w2_loss / self.batch_size
        self.bias_velocity = self.S * self.bias_velocity + self.lr * bias_loss / self.batch_size

        self.w1 -= self.w1_velocity
        self.w2 -= self.w2_velocity
        self.bias -= self.bias_velocity

    def train(self, X, y, n_epochs, lr, batch_size):
        self.batch_size = batch_size
        self.lr = lr

        for _ in range(n_epochs):
            iteration_error = 0
            mini_batches = create_mini_batches(X, y, self.batch_size)
            for batch in mini_batches:
                X_batch, y_batch = batch

                # forward
                predictions = self.forward_propagation(X_batch)

                # error
                error = y_batch.squeeze() - predictions
                iteration_error += np.sum(error ** 2) / self.batch_size

                # backprop and updating weights
                self.back_propagation(X_batch, error)

            self.MSE.append(iteration_error/n_epochs)

    def predict(self, input):
        predictions = self.forward_propagation(input)
        return predictions
