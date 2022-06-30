import numpy as np
from utils.XOR import XOR

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

xor_obj = XOR(n_epochs=10000, lr=0.1)
xor_obj.train(X, y)

################ tests ################
test = np.array([[1, 0]])
assert xor_obj.predict(test) == 1
test = np.array([[0, 0]])
assert xor_obj.predict(test) == 0
test = np.array([[0, 1]])
assert xor_obj.predict(test) == 1
test = np.array([[1, 1]])
assert xor_obj.predict(test) == 0
