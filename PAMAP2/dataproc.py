import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE = 171 # int
OVERLAP_RATE = 0.5 # float in [0，1）
SPLIT_RATE = None # 【subject105】as validation data

'''

def PAMAP(dataset_dir='./PAMAP2_Dataset/Protocol', WINDOW_SIZE=171, OVERLAP_RATE=0.5, VALIDATION_SUBJECTS={105}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录
        WINDOW_SIZE: 滑窗大小
        OVERLAP_RATE: 滑窗重叠率
        VALIDATION_SUBJECTS: 验证集所选取的Subjects
        Z_SCORE: 标准化
        SAVE_PATH: 预处理后npy数据保存目录
    '''
    
    print("\n原数据分析：共12个活动，文件包含9个受试者收集的数据，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n")
    print('预处理思路：提取有效列，重置活动label，数据降采样1/3，即100Hz -> 33.3Hz，进行滑窗，缺值填充，标准化等方法\n')

    #  保证验证选取的subjects无误
    assert VALIDATION_SUBJECTS
    for each in VALIDATION_SUBJECTS:
        assert each in set([*range(101, 110)])

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
    for file in filelist:
        subject_id = int(file.split('.')[0][-3:])
        content = pd.read_csv(file, sep=' ', usecols=[1]+[*range(4,16)]+[*range(21,33)]+[*range(38,50)]) # 取出有效数据列, 第2列为label，5-16，22-33，39-50都是可使用的传感数据
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy() # 线性插值填充nan
        
        # 降采样 1/3， 100Hz -> 33.3Hz
        data = content[::3, 1:] # 数据 （n, 36)
        label = content[::3, 0] # 标签

        data = data[label!=0] # 去除0类
        label = label[label!=0]

        print("==================================================================\n         【Loading File: “%s”】\nX-data shape: %s    Y-data shape: %s"%(file, data.shape, label.shape))
        for map_label in range(12):
            true_label = category_dict[map_label]
            cur_data = sliding_window(array=data[label==true_label], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)

            # 区分训练集 & 验证集
            if subject_id not in VALIDATION_SUBJECTS: # 训练集
                xtrain += cur_data
                ytrain += [map_label] * len(cur_data)
            else: # 验证集
                xtest += cur_data
                ytest += [map_label] * len(cur_data)

    os.chdir('../')
    print('==================================================================')
    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)

    if Z_SCORE: # 标准化
        xtrain_2d, xtest_2d = xtrain.reshape(-1, xtrain.shape[-1]), xtest.reshape(-1, xtest.shape[-1])
        std = StandardScaler().fit(xtrain_2d)
        xtrain_2d, xtest_2d = std.transform(xtrain_2d), std.transform(xtest_2d)
        xtrain, xtest = xtrain_2d.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]), xtest_2d.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'PAMAP2')
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
    PAMAP()
