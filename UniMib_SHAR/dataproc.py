import scipy.io as scio
import numpy as np
import os

'''
WINDOW_SIZE=151 # int
OVERLAP_RATE=0 # float in [0，1）
SPLIT_RATE=(7,3) # tuple or list  
'''

def UNIMIB(SPLIT_RATE=(7,3), dataset_dir='data'):
    print("\n原数据分析：原始文件共17个活动，acc_data.mat中已经滑窗切好了数据(11771, 453)，标签也已经准备好在acc_labels中(11771, 3)，不需要额外进行滑窗预处理。\n\
            观察数据分布可以发现unimib数据集的的数据是将xyz轴数据合并在了一起，length==453，表示前151是x轴数据，151-302为y轴数据，302-453为z轴数据\n")
    print("预处理思路：直接读取数据，匹配标签第一维，按比例进行训练集验证集切分\n")

    if not os.path.exists(dataset_dir):
        print('HAR-Dataset-Preprocess工程克隆不完整，请重新clone')
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
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    UNIMIB()
