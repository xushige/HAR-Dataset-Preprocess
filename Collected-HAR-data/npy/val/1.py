import numpy as np
a = np.load('acc_x.npy')

b = np.load('X_val.npy')
c = np.load('Y_val.npy')
print(a.shape, b.shape, c.shape)