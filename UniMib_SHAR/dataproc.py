import scipy.io as scio
import numpy as np
import os


def UNIMIB(SPLIT_RATE=(7,3), dataset_dir='data'):
    '''数据读取'''
    dir = dataset_dir
    data = scio.loadmat(os.path.join(dir, 'acc_data.mat'))['acc_data']
    label = scio.loadmat(os.path.join(dir, 'acc_labels.mat'))['acc_labels']
    print('total x shape: %s   total y shape: %s'%(data.shape, label.shape))
    print("y label 的三轴信息分别表示: [category, subject, trial]， 我们只需要第一轴")
    print(label, '\n')


    '''数据集切分'''
    train_data, train_label, test_data, test_label = [], [], [], []
    for category in range(17): # 17类
        cur_cate_data = data[label[:, 0]==category+1] # 当前类的x数据
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

    '''观察数据分布可以发现unimib数据集的的453表示前151是x轴数据，151-302为y轴数据，302-453为z轴数据'''
    train_data = train_data.reshape(train_data.shape[0], 3, 151).transpose(0, 2, 1)
    test_data = test_data.reshape(test_data.shape[0], 3, 151).transpose(0, 2, 1)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(train_data.shape, test_data.shape, train_label.shape, test_label.shape))
    return train_data, test_data, train_label, test_label


