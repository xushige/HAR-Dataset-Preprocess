import numpy as np
import pandas as pd
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def OPPO(dataset_dir='./OpportunityUCIDataset/dataset', WINDOW_SIZE=30, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2), VALIDATION_FILES={'S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat'}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_FILES": tuple
        VALIDATION_FILES: 留一法分割验证集，表示验证集所选取的file : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：原始文件共17个活动（不含null），column_names.txt文件中需要提取有效轴，论文中提到['S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat']用作验证集\n")
    print("预处理思路：提取有效列，重置活动label，遍历文件进行滑窗，缺值填充，标准化等方法\n")

    #  保证验证选取的files无误
    if VALIDATION_FILES:
        print('\n---------- 采用【留一法】分割验证集，选取的files为:%s ----------\n' % (VALIDATION_FILES))
        for each in VALIDATION_FILES:
            assert each in {'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'}
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    download_dataset(
        dataset_name='OPPORTUNITY',
        file_url='http://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip',
        dataset_dir=dataset_dir
    )
     
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

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data,用于存放最终数据

    '''数据读取，清洗'''
    filelist = os.listdir(dataset_dir)
    os.chdir(dataset_dir)
    select_col = [*range(1, 46)] + [*range(50, 59)] + [*range(63, 72)] + [*range(76, 85)] + [*range(89, 98)] + [*range(102, 134)] + [249] # 挑选的列：x-features 和 y-label(250列)
    print('Loading file data')
    for file in filelist:
        if '.dat' != file[-4:]:
            continue

        print('     current file: 【%s】'%(file), end='')
        print('   ----   Validation Data' if file in VALIDATION_FILES else '')
        
        content = pd.read_csv(file, sep=' ', usecols=select_col)
        x = content.iloc[:, :-1] # x
        x = x.interpolate(method='linear', limit_direction='both', axis=0).to_numpy() # 线性插值填充nan
        y = content.iloc[:, -1].to_numpy() # y

        for key in label_seq.keys():
            label_id = label_seq[key][1] # y 中目前还是key，因此需要根据key来确定送入滑窗的 x
            cur_data = sliding_window(array=x[y==key], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)

            # 两种分割验证集的方法 [留一法 or 平均法]
            if VALIDATION_FILES: # 留一法
                # 区分训练集 & 验证集
                if file not in VALIDATION_FILES: # 训练集
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

    xtrain, xtest, ytrain, ytest = np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

    if Z_SCORE: # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH: # 数组数据保存目录
        save_npy_data(
            dataset_name='OPPORTUNITY',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
        
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    OPPO()
