import numpy as np
import os
'''
WINDOW_SIZE=128 # int
OVERLAP_RATE=0.5 # float in [0，1）
SPLIT_RATE=-- # tuple or list  
'''

def UCI(dataset_dir='UCI_HAR_Dataset', SAVE_PATH=''):
    print("\n原数据分析：原数据已经指定比例切分好，窗口大小128，重叠率50%\n")
    print("预处理思路：读取数据，txt转numpy array\n")

    if not os.path.exists(dataset_dir):
        print('HAR-Dataset-Preprocess工程克隆不完整，请重新clone')
        quit()
        
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

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'UCI-HAR')
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/x_train.npy', X_train)
        np.save(path + '/x_test.npy', X_test)
        np.save(path + '/y_train.npy', Y_train)
        np.save(path + '/y_test.npy', Y_test)

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    UCI()
