import numpy as np
import matplotlib.pyplot as plt
from utils.XOR_v2 import XOR

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

xor_obj = XOR(n_epochs=5000, lr=0.1)
xor_obj.train(X, y)

################ tests ################
assert xor_obj.predict([1, 0]) == 1
assert xor_obj.predict([0, 0]) == 0
assert xor_obj.predict([0, 1]) == 1
assert xor_obj.predict([1, 1]) == 0

################ MSE ################
plt.plot(xor_obj.MSE)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()