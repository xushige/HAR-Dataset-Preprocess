import numpy as np
import pandas as pd

def UCI(dataset_dir='UCI_HAR_Dataset'):
    dataset = dataset_dir

    signal_class = [
        'body_acc_x_',
        'body_acc_y_',
        'body_acc_z_',
        'body_gyro_x_',
        'body_gyro_x_',
        'body_gyro_x_',
        'total_acc_x_',
        'total_acc_y_',
        'total_acc_z_',
    ]

    def xload(X_path):
        x = []
        for each in X_path:
            with open(each, 'r') as f:
                x.append(np.array([eachline.replace('  ', ' ').strip().split(' ') for eachline in f], dtype=np.float32))
        x = np.transpose(x, (1, 2, 0))
        return x

    def yload(Y_path):
        with open(Y_path, 'r') as f:
            y = np.array([eachline.replace('  ', ' ').strip().split(' ') for eachline in f], dtype=np.int64).reshape(-1)
        return y-1

    X_train_path = [dataset + '/train/Inertial Signals/' + signal + 'train.txt' for signal in signal_class]
    X_test_path = [dataset + '/test/Inertial Signals/' + signal + 'test.txt' for signal in signal_class]
    Y_train_path = dataset + '/train/y_train.txt'
    Y_test_path = dataset + '/test/y_test.txt'

    X_train = xload(X_train_path)
    X_test = xload(X_test_path)
    Y_train = yload(Y_train_path)
    Y_test = yload(Y_test_path)
    Y_train_onehot = np.array(pd.get_dummies(Y_train), dtype=np.int8)
    Y_test_onehot = np.array(pd.get_dummies(Y_test), dtype=np.int8)

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    return X_train, X_test, Y_train, Y_test