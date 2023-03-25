import scipy.io as scio
import numpy as np
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
'''
WINDOW_SIZE=151 # int
OVERLAP_RATE=0 # float in [0，1）
SPLIT_RATE=(7,3) # tuple or list  
'''

def UNIMIB(dataset_dir='./UniMiB-SHAR/data', SPLIT_RATE=(7,3), SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    print("\n原数据分析：原始文件共17个活动，acc_data.mat中已经滑窗切好了数据(11771, 453)，标签也已经准备好在acc_labels中(11771, 3)，不需要额外进行滑窗预处理。\n\
            观察数据分布可以发现unimib数据集的的数据是将xyz轴数据合并在了一起，length==453，表示前151是x轴数据，151-302为y轴数据，302-453为z轴数据\n")
    print("预处理思路：直接读取数据，匹配标签第一维，按比例进行训练集验证集切分\n")

    # 下载数据集[由于unimib数据集无法直接访问下载，这里我把unimib数据集上传到github进行访问clone]
    if not os.path.exists(dataset_dir):
        download_dataset(
            dataset_name='UniMiB-SHAR',
            file_url='https://github.com/xushige/UniMiB-SHAR.git', 
            dir_path=dataset_dir.split('/')[0]
        )
    if not os.path.exists(dataset_dir):
        print('\ngithub 下载数据集失败，请检查网络后重试\n')
        quit()
        
    '''数据读取'''
    dir = dataset_dir
    data = scio.loadmat(os.path.join(dir, 'acc_data.mat'))['acc_data']
    label = scio.loadmat(os.path.join(dir, 'acc_labels.mat'))['acc_labels']
    print('total x shape: %s   total y shape: %s'%(data.shape, label.shape))
    print("label 一共包含三维，分别表示: [category, subject, trial]， 我们只需要第一维category信息")
    print(label, '\n')


    '''数据集切分'''
    train_data, train_label, test_data, test_label = [], [], [], []
    for category in range(17): # 17类
        cur_cate_data = data[label[:, 0]==category+1] # 当前类的所有传感数据（包含传感器模态轴x、y、z）
        np.random.shuffle(cur_cate_data) # 打乱
        cur_cate_len = cur_cate_data.shape[0]
        trainlen = int(cur_cate_len*SPLIT_RATE[0]/sum(SPLIT_RATE)) # 训练集长度，默认训练：测试==7：3
        testlen = cur_cate_len-trainlen # 测试集长度
        print('=================================\ndealing category-【%d】 data\ntotal data length==%d'%(category+1, cur_cate_len))
        
        '''整理'''
        train_data += cur_cate_data[:trainlen].tolist()
        test_data += cur_cate_data[trainlen:].tolist()
        train_label += [category] * trainlen
        test_label += [category] * testlen
    print('=================================')
    train_data, train_label, test_data, test_label = np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

    '''x, y, z 模态轴拆分，需要利用reshape将其分开，再根据需要可以通过transpose将模态维转到最后一维'''
    train_data = train_data.reshape(train_data.shape[0], 3, 151).transpose(0, 2, 1)
    test_data = test_data.reshape(test_data.shape[0], 3, 151).transpose(0, 2, 1)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(train_data.shape, test_data.shape, train_label.shape, test_label.shape))

    if SAVE_PATH: # 数组数据保存目录
        path = os.path.join(SAVE_PATH, 'UniMiB_SHAR')
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/x_train.npy', train_data)
        np.save(path + '/x_test.npy', test_data)
        np.save(path + '/y_train.npy', train_label)
        np.save(path + '/y_test.npy', test_label)
        print('\n.npy数据【xtrain，xtest，ytrain，ytest】已经保存在【%s】目录下\n' % (SAVE_PATH))

    return train_data, test_data, train_label, test_label

if __name__ == '__main__':
    UNIMIB()
