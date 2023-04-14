import numpy as np 
import os
from sklearn.preprocessing import StandardScaler
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE=200 # int
OVERLAP_RATE=0.5 # float in [0，1）
SPLIT_RATE=(7,3) # tuple or list  

'''
def WISDM(dataset_dir='./WISDM_ar_v1.1', WINDOW_SIZE=200, OVERLAP_RATE=0.5, VALIDATION_SUBJECTS={29, 31, 32, 33, 34, 36}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录
        WINDOW_SIZE: 滑窗大小
        OVERLAP_RATE: 滑窗重叠率
        VALIDATION_SUBJECTS: 验证集所选取的Subjects
        Z_SCORE: 标准化
        SAVE_PATH: 预处理后npy数据保存目录
    '''
    
    print("\n原数据分析：共6个活动，在WISDM_ar_v1.1_raw.txt文件中，第二列为类别，四五六列为传感信号，抛弃一列和三列即可。\n\
            数据较杂乱，需要数据清洗，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集\n")

    #  保证验证选取的subjects无误
    assert VALIDATION_SUBJECTS
    for each in VALIDATION_SUBJECTS:
        assert each in set([*range(1, 37)])

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
    for subject_id in range(1, 37):
        for label_id in range(6):
            mask = np.logical_and(subject == subject_id, label == label_id)
            cur_data = sliding_window(data[mask], WINDOW_SIZE, OVERLAP_RATE)

            # 区分训练集 & 验证集
            if subject_id not in VALIDATION_SUBJECTS: # 训练集
                xtrain += cur_data
                ytrain += [label_id] * len(cur_data)
            else: # 验证集
                xtest += cur_data
                ytest += [label_id] * len(cur_data)

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
        path = os.path.join(SAVE_PATH, 'WISDM')
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
    WISDM()
