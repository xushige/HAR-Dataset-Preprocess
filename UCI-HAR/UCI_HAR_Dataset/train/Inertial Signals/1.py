import matplotlib.pyplot as plt
import numpy as np

def unit(file):
    with open(file, 'r') as x:
        temp = []
        for eachline in x:
            if len(temp) == 10:
                break
            temp.append(eachline.replace('  ', ' ').strip().split(' '))
        result = []
        for eachline in temp:
            for each in eachline:
                result.append(each)
        return np.array(result)

x = unit('body_acc_x_train.txt')
y = unit('body_acc_y_train.txt')
z = unit('body_acc_z_train.txt')

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.legend()
plt.show()