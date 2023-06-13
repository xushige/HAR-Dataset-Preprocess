import os
import numpy as np
import pandas as pd
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def DASA(dataset_dir='./data', WINDOW_SIZE=125, OVERLAP_RATE=0.4, SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={7, 8}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print('\n原数据分析：原始文件共19个活动，每个活动都由8个受试者进行信号采集，每个受试者在每一类上采集5min的信号数据，采样频率25hz（每个txt是 125*45 的数据，包含5s时间长度，共60个txt）\n')
    print('预处理思路：数据集网站的介绍中说到60个txt是有5min连续数据分割而来,因此某一类别a下同一个受试者p的60个txt数据是时序连续的。\n\
            所以可以将a()p()下的所有txt数据进行时序维度拼接，选择窗口大小为125，重叠率为40%进行滑窗。\n')
    
    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(1, 9)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    download_dataset(
        dataset_name='Daily_and_Sports_Activities',
        file_url='http://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], []
    adls = sorted(os.listdir(dataset_dir))
    os.chdir(dataset_dir)
    for label_id, adl in enumerate(adls): # each adl

        print('======================================================\ncurrent activity sequence: 【%s】'%(adl))
        
        participants = sorted(os.listdir(adl))
        os.chdir(adl)
        for participant_idx, participant in enumerate(participants): # each subject

            subject_id = participant_idx + 1
            print('      current subject: 【%d】'%(subject_id), end='')
            print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

            files = sorted(os.listdir(participant))
            os.chdir(participant)
            concat_data = np.vstack([pd.read_csv(file, sep=',', header=None).to_numpy() for file in files]) # concat series data (125*60, 45)
            cur_data = sliding_window(array=concat_data, windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE) # sliding window [n, 125, 45]

            # 两种分割验证集的方法 [留一法 or 平均法]
            if VALIDATION_SUBJECTS: # 留一法
                # 区分训练集 & 验证集
                if subject_id not in VALIDATION_SUBJECTS: # 训练集
                    xtrain += cur_data
                    ytrain += [label_id] * len(cur_data)
                else: # 验证集
                    xtest += cur_data
                    ytest += [label_id] * len(cur_data)
            else: # 平均法
                trainlen = int(len(cur_data) * SPLIT_RATE[0] / sum(SPLIT_RATE)) # 训练集长度
                testlen = len(cur_data) - trainlen # 验证集长度
                xtrain += cur_data[:trainlen]
                xtest += cur_data[trainlen:]
                ytrain += [label_id] * trainlen
                ytest += [label_id] * testlen

            os.chdir('../')
        os.chdir('../')
    os.chdir('../') 
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(ytrain, dtype=np.int64), np.array(ytest, dtype=np.int64)
    
    if Z_SCORE: # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        save_npy_data(
            dataset_name='Daily_and_Sports_Activities',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
        
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    DASA()
