import numpy as np
import pandas as pd
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def UCI(dataset_dir='./UCI HAR Dataset', SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：原数据已经指定比例切分好，窗口大小128，重叠率50%\n")
    print("预处理思路：读取数据，txt转numpy array\n")

    # 下载数据集
    download_dataset(
        dataset_name='UCI-HAR',
        file_url='https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
        dataset_dir=dataset_dir
    )
        
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
        y = pd.read_csv(Y_path, header=None).to_numpy().reshape(-1)
        return y - 1  # label从0开始

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
        save_npy_data(
            dataset_name='UCI_HAR',
            root_dir=SAVE_PATH,
            xtrain=X_train,
            xtest=X_test,
            ytrain=Y_train,
            ytest=Y_test
        )
           
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    UCI()
