import os
import numpy as np
from sklearn.preprocessing import StandardScaler

'''
WINDOW_SIZE=125 # int
OVERLAP_RATE=0.4 # float in [0，1）
SPLIT_RATE=None # "p7,p8" as the validation data
'''

def DASA(dataset_dir='data', WINDOW_SIZE=125, OVERLAP_RATE=0.4, Z_SCORE=True):
    print('\n原数据分析：原始文件共19个活动，每个活动都由8个受试者进行信号采集，每个受试者在每一类上采集5min的信号数据，采样频率25hz（每个txt是 125*45 的数据，包含5s时间长度，共60个txt）\n')
    print('预处理思路：数据集网站的介绍中说到60个txt是有5min连续数据分割而来,因此某一类别a下同一个受试者p的60个txt数据是时序连续的。\n\
            所以可以将a()p()下的所有txt数据进行时序维度拼接，选择窗口大小为125，重叠率为20%进行滑窗重采样。p7p8受试者的数据作为验证集。\n')

    def slide_window(list_data, windowsize, overlaprate):
        '''
        list_data: list数据
        windowsize: 窗口尺寸
        overlaprate: 重叠率
        '''
        stride = int(windowsize * (1 - overlaprate)) # 计算stride
        times = (len(list_data)-windowsize)//stride + 1 # 滑窗次数，同时也是滑窗后数据长度
        res = []
        for i in range(times):
            x = list_data[i*stride : i*stride+windowsize]
            res.append(x)
        return res

    if not os.path.exists(dataset_dir):
        print('HAR-Dataset-Preprocess工程克隆不完整，请重新clone')
        quit()

    xtrain, xtest, ytrain, ytest = [], [], [], []
    adls = sorted(os.listdir(dataset_dir))
    os.chdir(dataset_dir)
    for category_id, adl in enumerate(adls): # each adl
        print('======================================================\n         current activity sequence: 【%s】\n'%(adl))
        participants = sorted(os.listdir(adl))
        os.chdir(adl)
        for participant_id, participant in enumerate(participants):
            files = sorted(os.listdir(participant))
            os.chdir(participant)
            series_data = [] # concat data (125*60, 45)
            for file in files:
                with open(file, 'r') as f:
                    for eachline in f:
                        series_data.append(eachline.strip().split(','))
            series_data = slide_window(series_data, windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE) # sliding window [74, 125, 45]
            if (participant_id+1) < 7: # train data
                xtrain += series_data
                ytrain += [category_id] * len(series_data)
            else: # validation data
                xtest += series_data
                ytest += [category_id] * len(series_data)
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
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    DASA()
