import numpy as np 
import os
'''
WINDOW_SIZE=200 # int
OVERLAP_RATE=0.75 # float in [0，1）
SPLIT_RATE=(7,3) # tuple or list  

'''
def WISDM(WINDOW_SIZE=200, OVERLAP_RATE=0.75, SPLIT_RATE=(7,3), dataset_dir='WISDM_ar_v1.1'):

    if not os.path.exists(dataset_dir):
        print('HAR-Dataset-Preprocess工程克隆不完整，请重新clone')
        quit()

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data

    def slide_window(array, windowsize, overlaprate, label, split_rate=(7, 3)):
        '''
        array: 2d-numpy数组
        windowsize: 窗口尺寸
        overlaprate: 重叠率
        label: array对应的标签
        split_rate: train-test数据量比例
        '''
        nonlocal xtrain, xtest, ytrain, ytest
        stride = int(windowsize * (1 - overlaprate)) # 计算stride
        times = (array.shape[0]-windowsize)//stride + 1 # 滑窗次数，同时也是滑窗后数据长度
        tempx = []
        for i in range(times):
            x = array[i*stride : i*stride+windowsize]
            tempx.append(x)
        np.random.shuffle(tempx) # shuffle
        # 切分数据集
        trainleng = int(times*split_rate[0]/sum(split_rate))
        xtrain += tempx[: trainleng]
        xtest += tempx[trainleng: ]
        ytrain += [label]*trainleng
        ytest += [label]*(times-trainleng)


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
    label = temp[:, 1]
    for category in category_dict.keys():
        label[label==category] = category_dict[category]
    label = label.astype(np.int64)
    data = temp[:, 3:].astype(np.float32)

    # data, label = data[::2], label[::2] # 降采样1/2

    '''滑窗'''
    for i in range(6):
        slide_window(data[label==i], WINDOW_SIZE, OVERLAP_RATE, i, SPLIT_RATE)
    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    WISDM()