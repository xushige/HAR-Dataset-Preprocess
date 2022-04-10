import numpy as np
a = np.load('acc_x.npy')

b = np.load('X_test.npy')
c = np.load('Y_test.npy')
print(a.shape, b.shape, c.shape, c)