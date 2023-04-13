import os
import numpy as np
import pandas as pd
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def PAMAP(dataset_dir='./PAMAP2_Dataset/Protocol', WINDOW_SIZE=171, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={105}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：共12个活动，文件包含9个受试者收集的数据，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n")
    print('预处理思路：提取有效列，重置活动label，数据降采样1/3，即100Hz -> 33.3Hz，进行滑窗，缺值填充，标准化等方法\n')

    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(101, 110)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    download_dataset(
        dataset_name='PAMAP2',
        file_url='http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data, 用于存放最终数据
    category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])) #12分类所对应的实际label，对应readme.pdf

    dir = dataset_dir
    filelist = os.listdir(dir)
    os.chdir(dir)
    print('Loading subject data')
    for file in filelist:
        
        subject_id = int(file.split('.')[0][-3:])
        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        content = pd.read_csv(file, sep=' ', usecols=[1]+[*range(4,16)]+[*range(21,33)]+[*range(38,50)]) # 取出有效数据列, 第2列为label，5-16，22-33，39-50都是可使用的传感数据
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy() # 线性插值填充nan
        
        # 降采样 1/3， 100Hz -> 33.3Hz
        data = content[::3, 1:] # 数据 （n, 36)
        label = content[::3, 0] # 标签

        data = data[label!=0] # 去除0类
        label = label[label!=0]

        for label_id in range(12):
            true_label = category_dict[label_id]
            cur_data = sliding_window(array=data[label==true_label], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)

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

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)

    if Z_SCORE: # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        save_npy_data(
            dataset_name='PAMAP2',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
        
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    PAMAP()
