import scipy.io as scio
import os
import numpy as np
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE=512 # int
OVERLAP_RATE=0.5 # float in [0，1）
SPLIT_RATE=(8,2) # tuple or list  
'''

def USC(dataset_dir='./USC-HAD', WINDOW_SIZE=512, OVERLAP_RATE=0.5, SPLIT_RATE=(8,2), SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    print("\n原数据分析：共12个活动，由14个受试者采集，和DASA数据集不同，这里每个mat文件的长度并不一致，因此需要对每一个mat数据进行滑窗预处理后再合并。\n\
            切分数据集思路可以采取和DASA数据集一样的方法，选取1~2个受试者数据作为验证集（留一法），也可以打乱平均按比例切分（同PAMAP一样）。这里用的是打乱后按比例分\n")

    # 下载数据集
    if not os.path.exists(dataset_dir):
        download_dataset(
            dataset_name='USC-HAD',
            file_url='https://sipi.usc.edu/had/USC-HAD.zip', 
            dir_path=dataset_dir.split('/')[0]
        )
        
    '''数据处理'''
    def slide_window(array, w_s, stride):
        '''
        滑窗处理
        array: ---
        w_s: 窗口大小
        stride： 滑动步长
        '''
        x = []
        times = (array.shape[0] - w_s) // stride + 1
        i=0
        for i in range(times):
            x.append(array[stride*i: stride*i+w_s]) 
        #最后一个保留处理 
        if stride*i+w_s < array.shape[0]-1:
            x.append(array[-w_s:])
        return x
    def merge_data(path, w_s, stride):
        '''
        所有数据按类别进行合并
        path: 原始 USC_HAD 数据路径
        w_s： 指定滑窗大小
        stride： 指定步长
        '''
        result = [[] for i in range(12)] # 12类，按索引放置每一类数据
        '''对每一个数据进行滑窗处理，将滑窗后的数据按类别叠加合并放入result对应位置'''
        subject_list = os.listdir(path)
        os.chdir(path)
        for subject in subject_list:
            if not os.path.isdir(subject):
                continue
            print('======================================================\n         current Subject sequence: 【%s】\n'%(subject))
            mat_list = os.listdir(subject)
            os.chdir(subject)
            for mat in mat_list:
                category = int(mat[1:-6])-1 #获取类别
                content = scio.loadmat(mat)['sensor_readings']
                x = slide_window(content, w_s, stride)
                result[category].extend(x)
            os.chdir('../')
        os.chdir('../')
        return result
    def split_data(array, ratio=(8, 2)):
        '''
        数据切分
        array： 按类别合并后的数据（12，）
        ratio： 训练集与测试集长度比例
        '''
        xtrain, xtest, ytrain, ytest = [], [], [], []
        train_part = ratio[0] / sum(ratio)
        '''对每一类数据按比例切分，切分同时生成label'''
        for i, data in enumerate(array):
            np.random.shuffle(data)
            train_leng = int(len(data) * train_part)
            test_leng = len(data)-train_leng
            xtrain.extend(data[: train_leng])
            xtest.extend(data[train_leng: ])
            ytrain.extend([i] * train_leng)
            ytest.extend([i] * test_leng)
        xtrain, xtest, ytrain, ytest = np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)
        print('\n---------------------------------------------------------------------------------------------------------------------\n')
        print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

        if SAVE_PATH: # 数组数据保存目录
            path = os.path.join(SAVE_PATH, 'USC_HAD')
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/x_train.npy', xtrain)
            np.save(path + '/x_test.npy', xtest)
            np.save(path + '/y_train.npy', ytrain)
            np.save(path + '/y_test.npy', ytest)
            print('\n.npy数据【xtrain，xtest，ytrain，ytest】已经保存在【%s】目录下\n' % (SAVE_PATH))
            build_npydataset_readme(SAVE_PATH)
        
        return xtrain, xtest, ytrain, ytest

    return split_data(
        merge_data(
            path = dataset_dir, 
            w_s = WINDOW_SIZE, 
            stride = int(WINDOW_SIZE*(1-OVERLAP_RATE))
            ), 
        ratio = SPLIT_RATE)

if __name__ == '__main__':
    USC()
