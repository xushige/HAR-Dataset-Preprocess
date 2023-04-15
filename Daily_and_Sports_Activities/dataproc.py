import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE = 125 # int
OVERLAP_RATE = 0.4 # float in [0，1）
SPLIT_RATE = None # 【p7,p8】 as the validation data
'''

def DASA(dataset_dir='./data', WINDOW_SIZE=125, OVERLAP_RATE=0.4, VALIDATION_SUBJECTS={7, 8}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录
        WINDOW_SIZE: 滑窗大小
        OVERLAP_RATE: 滑窗重叠率
        VALIDATION_SUBJECTS: 验证集所选取的Subjects
        Z_SCORE: 标准化
        SAVE_PATH: 预处理后npy数据保存目录
    '''
    
    print('\n原数据分析：原始文件共19个活动，每个活动都由8个受试者进行信号采集，每个受试者在每一类上采集5min的信号数据，采样频率25hz（每个txt是 125*45 的数据，包含5s时间长度，共60个txt）\n')
    print('预处理思路：数据集网站的介绍中说到60个txt是有5min连续数据分割而来,因此某一类别a下同一个受试者p的60个txt数据是时序连续的。\n\
            所以可以将a()p()下的所有txt数据进行时序维度拼接，选择窗口大小为125，重叠率为20%进行滑窗重采样。切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n')
    
    #  保证验证选取的subjects无误
    assert VALIDATION_SUBJECTS
    for each in VALIDATION_SUBJECTS:
        assert each in set([*range(1, 9)])

    # 下载数据集
    download_dataset(
        dataset_name='Daily_and_Sports_Activities',
        file_url='http://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], []
    adls = sorted(os.listdir(dataset_dir))
    os.chdir(dataset_dir)
    for category_idx, adl in enumerate(adls): # each adl
        print('======================================================\n         current activity sequence: 【%s】\n'%(adl))
        participants = sorted(os.listdir(adl))
        os.chdir(adl)
        for participant_idx, participant in enumerate(participants): # each subject
            subject_id = participant_idx + 1
            files = sorted(os.listdir(participant))
            os.chdir(participant)
            concat_data = np.vstack([pd.read_csv(file, sep=',', header=None).to_numpy() for file in files]) # concat series data (125*60, 45)
            series_data = sliding_window(array=concat_data, windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE) # sliding window [n, 125, 45]

            if subject_id not in VALIDATION_SUBJECTS: # train data
                xtrain += series_data
                ytrain += [category_idx] * len(series_data)
            else: # validation data
                xtest += series_data
                ytest += [category_idx] * len(series_data)
            os.chdir('../')
        os.chdir('../')
    os.chdir('../') 
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(ytrain, dtype=np.int64), np.array(ytest, dtype=np.int64)
    
    if Z_SCORE: # 标准化
        xtrain_2d, xtest_2d = xtrain.reshape(-1, xtrain.shape[-1]), xtest.reshape(-1, xtest.shape[-1])
        std = StandardScaler().fit(xtrain_2d)
        xtrain_2d, xtest_2d = std.transform(xtrain_2d), std.transform(xtest_2d)
        xtrain, xtest = xtrain_2d.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]), xtest_2d.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'Daily_and_Sports_Activities')
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
    DASA()
