import scipy.io as scio
import os
import numpy as np
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def USC(dataset_dir='./USC-HAD', WINDOW_SIZE=100, OVERLAP_RATE=0.1, SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：共12个活动，由14个受试者采集，和DASA数据集不同，这里每个mat文件的长度并不一致，因此需要对每一个mat数据进行滑窗预处理后再合并。\n\
            切分数据集思路可以采取留一法，选取n个受试者数据作为验证集\n")
    
    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(1, 15)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    download_dataset(
        dataset_name='USC-HAD',
        file_url='https://sipi.usc.edu/had/USC-HAD.zip', 
        dataset_dir=dataset_dir
    )
    
    xtrain, xtest, ytrain, ytest = [], [], [], [] # 最终数据

    subject_list = os.listdir(dataset_dir)
    os.chdir(dataset_dir)
    print('Loading subject data')
    for subject in subject_list:
        if not os.path.isdir(subject):
            continue

        subject_id = int(subject[7:])
        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        mat_list = os.listdir(subject)
        os.chdir(subject)
        for mat in mat_list:
            label_id = int(mat[1:-6])-1 #获取类别
            content = scio.loadmat(mat)['sensor_readings']
            cur_data = sliding_window(content, WINDOW_SIZE, OVERLAP_RATE)

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

    xtrain, xtest, ytrain, ytest = np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

    if Z_SCORE: # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        save_npy_data(
            dataset_name='USC_HAD',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
        
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    USC()
