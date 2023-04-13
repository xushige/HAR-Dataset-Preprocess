import scipy.io as scio
import numpy as np
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def UNIMIB(dataset_dir='./UniMiB-SHAR/data', SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：原始文件共17个活动，acc_data.mat中已经滑窗切好了数据(11771, 453)，标签也已经准备好在acc_labels中(11771, 3)，不需要额外进行滑窗预处理。\n\
            观察数据分布可以发现unimib数据集的的数据是将xyz轴数据合并在了一起，length==453，表示前151是x轴数据，151-302为y轴数据，302-453为z轴数据\n")
    print("预处理思路：直接读取数据，匹配标签第一维，原论文里采用了5fold和leave-one-subject进行数据集切割评估，这里默认按5-fold比例进行训练集验证集切分\n")

    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(1, 31)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集[由于unimib数据集无法直接访问下载，这里我把unimib数据集上传到gitcdoe进行访问clone]
    download_dataset(
        dataset_name='UniMiB-SHAR',
        file_url='https://gitcode.net/m0_52161961/UniMiB-SHAR.git', 
        dataset_dir=dataset_dir
    )
    
    '''数据读取'''
    dir = dataset_dir
    data = scio.loadmat(os.path.join(dir, 'acc_data.mat'))['acc_data']
    label = scio.loadmat(os.path.join(dir, 'acc_labels.mat'))['acc_labels']
    
    '''数据集切分'''
    xtrain, ytrain, xtest, ytest = [], [], [], [] #存放最终数据

    print('Loading subject data')
    for subject_id in range(1, 31):

        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        for label_id in range(17): # 17类
            mask = np.logical_and(label[:, 0] == label_id+1, label[:, 1] == subject_id)
            cur_data = data[mask].tolist() # 当前subject和label_id类的所有传感数据（包含传感器模态轴x、y、z）
            
            # 两种分割验证集的方法 [留一法 or 平均法]
            if VALIDATION_SUBJECTS: # 留一法
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

    xtrain, ytrain, xtest, ytest = np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)

    '''x, y, z 模态轴拆分，需要利用reshape将其分开，再根据需要可以通过transpose将模态维转到最后一维'''
    xtrain = xtrain.reshape(xtrain.shape[0], 3, 151).transpose(0, 2, 1)
    xtest = xtest.reshape(xtest.shape[0], 3, 151).transpose(0, 2, 1)

    if Z_SCORE: # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        save_npy_data(
            dataset_name='UniMiB_SHAR',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
            
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    UNIMIB()
