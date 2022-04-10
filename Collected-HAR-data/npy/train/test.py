import numpy as np
import torch
import torch.nn as nn
a = np.load('acc_x.npy')

b = np.load('X_train.npy')
c = np.load('Y_train.npy')
print(a.shape, b.shape, c.shape)

a = torch.tensor([1, 2, 3])
print(a.size())
b = nn.functional.one_hot(a, 5).float()
print(b)
