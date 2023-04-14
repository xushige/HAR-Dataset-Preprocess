import scipy.io as scio
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE=512 # int
OVERLAP_RATE=0.5 # float in [0，1）
SPLIT_RATE=(8,2) # tuple or list  
'''

def USC(dataset_dir='./USC-HAD', WINDOW_SIZE=100, OVERLAP_RATE=0.1, VALIDATION_SUBJECTS={13, 14}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录
        WINDOW_SIZE: 滑窗大小
        OVERLAP_RATE: 滑窗重叠率
        VALIDATION_SUBJECTS: 验证集所选取的Subjects
        Z_SCORE: 标准化
        SAVE_PATH: 预处理后npy数据保存目录
    '''
    
    print("\n原数据分析：共12个活动，由14个受试者采集，和DASA数据集不同，这里每个mat文件的长度并不一致，因此需要对每一个mat数据进行滑窗预处理后再合并。\n\
            切分数据集思路可以采取留一法，选取n个受试者数据作为验证集\n")
    
    #  保证验证选取的subjects无误
    assert VALIDATION_SUBJECTS
    for each in VALIDATION_SUBJECTS:
        assert each in set([*range(1, 15)])

    # 下载数据集
    download_dataset(
        dataset_name='USC-HAD',
        file_url='https://sipi.usc.edu/had/USC-HAD.zip', 
        dataset_dir=dataset_dir
    )
    
    xtrain, xtest, ytrain, ytest = [], [], [], [] # 最终数据

    subject_list = os.listdir(dataset_dir)
    os.chdir(dataset_dir)
    for subject in subject_list:
        if not os.path.isdir(subject):
            continue
        print('======================================================\n         current Subject sequence: 【%s】\n'%(subject))
        subject_id = int(subject[7:])
        mat_list = os.listdir(subject)
        os.chdir(subject)
        for mat in mat_list:
            label = int(mat[1:-6])-1 #获取类别
            content = scio.loadmat(mat)['sensor_readings']
            cur_data = sliding_window(content, WINDOW_SIZE, OVERLAP_RATE)

            # 区分训练集 & 验证集
            if subject_id not in VALIDATION_SUBJECTS: # 训练集
                xtrain += cur_data
                ytrain += [label] * len(cur_data)
            else: # 验证集
                xtest += cur_data
                ytest += [label] * len(cur_data)

        os.chdir('../')
    os.chdir('../')

    xtrain, xtest, ytrain, ytest = np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

    if Z_SCORE: # 标准化
        xtrain_2d, xtest_2d = xtrain.reshape(-1, xtrain.shape[-1]), xtest.reshape(-1, xtest.shape[-1])
        std = StandardScaler().fit(xtrain_2d)
        xtrain_2d, xtest_2d = std.transform(xtrain_2d), std.transform(xtest_2d)
        xtrain, xtest = xtrain_2d.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]), xtest_2d.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'USC_HAD')
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/x_train.npy', xtrain)
        np.save(path + '/x_test.npy', xtest)
        np.save(path + '/y_train.npy', ytrain)
        np.save(path + '/y_test.npy', ytest)
        print('\n.npy数据【xtrain，xtest，ytrain，ytest】已经保存在【%s】目录下\n' % (SAVE_PATH))
        build_npydataset_readme(SAVE_PATH)
    
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    USC()
