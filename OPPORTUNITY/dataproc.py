import numpy as np
import pandas as pd
from pandas import Series
import os
from sklearn.preprocessing import StandardScaler
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE = 30 # int
OVERLAP_RATE = 0.5 # float in [0，1）
SPLIT_RATE = None # 【'S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat'】 as validation data
'''

def OPPO(dataset_dir='./OpportunityUCIDataset/dataset', WINDOW_SIZE=30, OVERLAP_RATE=0.5, VALIDATION_FILES={'S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat'}, Z_SCORE=False, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录
        WINDOW_SIZE: 滑窗大小
        OVERLAP_RATE: 滑窗重叠率
        VALIDATION_SUBJECTS: 验证集所选取的Subjects
        Z_SCORE: 标准化
        SAVE_PATH: 预处理后npy数据保存目录
    '''
    
    print("\n原数据分析：原始文件共17个活动（不含null），column_names.txt文件中需要提取有效轴，论文中提到['S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat']用作验证集\n")
    print("预处理思路：提取有效列，重置活动label，遍历文件进行滑窗，缺值填充，标准化等方法\n")

    # 下载数据集
    download_dataset(
        dataset_name='OPPORTUNITY',
        file_url='http://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data,用于存放最终数据
            
    '''标签转换 (17 分类不含 null 类)'''
    label_seq = {
        406516: ['Open Door 1', 0],  # 文件中类别对应编号: [ 类别名, 预处理后label ]
        406517: ['Open Door 2', 1],
        404516: ['Close Door 1', 2],
        404517: ['Close Door 2', 3],
        406520: ['Open Fridge', 4],
        404520: ['Close Fridge', 5],
        406505: ['Open Dishwasher', 6],
        404505: ['Close Dishwasher', 7],
        406519: ['Open Drawer 1', 8],
        404519: ['Close Drawer 1', 9],
        406511: ['Open Drawer 2', 10],
        404511: ['Close Drawer 2', 11],
        406508: ['Open Drawer 3', 12],
        404508: ['Close Drawer 3', 13],
        408512: ['Clean Table', 14],
        407521: ['Drink from Cup', 15],
        405506: ['Toggle Switch', 16]
    }

    '''数据读取，清洗'''
    Trainx, Testx, Trainy, Testy = [], [], [], [] # 存放清洗后的数据，并非最终数据
    filelist = os.listdir(dataset_dir)
    os.chdir(dataset_dir)
    select_col = [*range(1, 46)] + [*range(50, 59)] + [*range(63, 72)] + [*range(76, 85)] + [*range(89, 98)] + [*range(102, 134)] + [249] # 挑选的列：x-features 和 y-label(250列)
    print('Loading file data')
    for file in filelist:
        if '.dat' != file[-4:]:
            continue
        print('     current file: 【%s】'%(file), end='  ')
        content = pd.read_csv(file, sep=' ', usecols=select_col)
        x = content.iloc[:, :-1] # x
        y = content.iloc[:, -1].to_numpy() # y
        x = x.fillna(0.0).to_numpy() # 缺值0填充
        for col in range(x.shape[1]):
            x[:, col] = Series(x[:, col]).interpolate() # 每列线性插值填充
        x[np.isnan(x)] = 0 # 保证数据无NAN

        # 区分训练集 & 验证集
        if file in VALIDATION_FILES: #验证集
            print('----  Test data')
            Testx += x.tolist()
            Testy += y.tolist()
        else: # 训练集
            print()
            Trainx += x.tolist()
            Trainy += y.tolist()
    
    '''标准化'''
    if Z_SCORE:
        std = StandardScaler().fit(Trainx)
        Trainx = std.transform(Trainx)
        Testx = std.transform(Testx)
    
    '''按label滑窗'''
    for key in label_seq.keys():
        label = label_seq[key][1] # y 中目前还是key，因此需要根据key来确定送入滑窗的 x
        # 训练集滑窗
        train = sliding_window(array=Trainx[np.array(Trainy)==key], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)
        xtrain += train
        ytrain += [label] * len(train)
        # 验证集滑窗
        test = sliding_window(array=Testx[np.array(Testy)==key], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)
        xtest += test
        ytest += [label] * len(test)

    xtrain, xtest, ytrain, ytest = np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'OPPORTUNITY')
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
    OPPO()
