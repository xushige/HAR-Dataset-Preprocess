import numpy as np 
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def WISDM(dataset_dir='./WISDM_ar_v1.1', WINDOW_SIZE=200, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''
    
    print("\n原数据分析：共6个活动，在WISDM_ar_v1.1_raw.txt文件中，第二列为类别，四五六列为传感信号，抛弃一列和三列即可。\n\
            数据较杂乱，需要数据清洗。按subject && label的数据进行滑窗\n")

    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(1, 37)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    download_dataset(
        dataset_name='WISDM',
        file_url='https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz', 
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data

    category_dict = {
        'Walking': 0,
        'Jogging': 1,
        'Sitting': 2,
        'Standing': 3,
        'Upstairs': 4,
        'Downstairs': 5
    }
    '''数据清洗+读取'''
    filename = r'%s/WISDM_ar_v1.1_raw.txt'%(dataset_dir)
    f = open(filename)
    content = f.read().strip('.\n').split('\n')
    temp = []
    for row in content:
        row = row.strip(';').strip(',').strip()
        if len(row.split(','))<6:
            continue
        for each in row.split(';'):
            temp.append(each.strip(';').strip(',').strip().split(','))
    f.close()
    temp = np.array(temp)

    '''label编码'''
    subject = temp[:, 0]
    label = temp[:, 1]
    for category in category_dict.keys():
        label[label==category] = category_dict[category]
    subject = subject.astype(np.int32)
    label = label.astype(np.int64)
    data = temp[:, 3:].astype(np.float32)

    '''滑窗'''
    print('Loading subject data')
    for subject_id in range(1, 37):

        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')
        
        for label_id in range(6):
            mask = np.logical_and(subject == subject_id, label == label_id)
            cur_data = sliding_window(data[mask], WINDOW_SIZE, OVERLAP_RATE)

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
            dataset_name='WISDM',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
          
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    WISDM()
