import numpy as np
import models_and_tools as t

file_dict = {
    0: 'down',
    1: 'sit',
    2: 'stand',
    3: 'up',
    4: 'walk'
}

acc_x_train, acc_x_val, acc_x_test = t.get_accdata_and_save_label(file_dict, 128, 6, 0, save_label=True)
acc_y_train, acc_y_val, acc_y_test = t.get_accdata_and_save_label(file_dict, 128, 6, 1)
acc_z_train, acc_z_val, acc_z_test = t.get_accdata_and_save_label(file_dict, 128, 6, 2)

X_train = t.unit_axis(acc_x_train, acc_y_train, acc_z_train)
X_val = t.unit_axis(acc_x_val, acc_y_val, acc_z_val)
X_test = t.unit_axis(acc_x_test, acc_y_test, acc_z_test)

np.save('npy/train/X_train.npy', X_train)
np.save('npy/val/X_val.npy', X_val)
np.save('npy/test/X_test.npy', X_test)



