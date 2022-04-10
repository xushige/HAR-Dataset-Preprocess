import scipy.io as scio
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np
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
    print(np.array(xtrain).shape, np.array(xtest).shape, np.array(ytrain).shape, np.array(ytest).shape)
    return np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

'''模型训练'''
class CNN(nn.Module):
    def __init__(self, category):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (3, 1), (2, 1), (1, 0)), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.flc = nn.Linear(512*2*6, category)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x
    
if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = split_data(merge_data('USC-HAD', 512, 256)) # 获取数据
    xtrain = torch.from_numpy(xtrain).unsqueeze(1).float() # 增维 视作图像处理
    xtest = torch.from_numpy(xtest).unsqueeze(1).float()
    ytrain = torch.from_numpy(ytrain).long()
    ytest = torch.from_numpy(ytest).long()
    category = len(Counter(ytest.numpy())) # 统计类别
    print('xtrain_size: 【%s】\nxtest_size: 【%s】\ncategory: 【%d】' % (xtrain.size(), xtest.size(), category))

    net = CNN(category=category).cuda()
    EP = 100
    B_S = 128
    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001) # lr正则化
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.5) # 学习率衰减，20次衰减一半
    loss_fn = nn.CrossEntropyLoss() # 损失函数

    '''数据集封装'''
    train_data = TensorDataset(xtrain, ytrain) 
    test_data = TensorDataset(xtest, ytest)
    train_loader = DataLoader(train_data, B_S ,shuffle=True)
    test_loader = DataLoader(test_data, B_S, shuffle=True)

    view_loss = [] # loss可视化
    view_acc = [] # acc 可视化
    best_acc = 0 # 最优acc挑模型
    for i in range(EP):
        '''训练'''
        for data, label in train_loader:
            data, label = data.cuda(), label.cuda() # 数据放到gpu上
            net.train() # 训练模式，需要bn，dropout
            out = net(data) # （batch， category）
            loss = loss_fn(out, label) # loss
            optimizer.zero_grad() # 清除每个节点上次计算的梯度
            loss.backward() # 根据loss计算本次梯度，
            optimizer.step() # 更新权重参数
        scheduler.step() # 学习率step
        view_loss.append(loss) # 添加训练第一个epoch的loss进行可视化
        cor = 0 # 统计预测正确的数量
        '''测试'''
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            net.eval() # 测试模式
            out = net(data)
            _, prediction = torch.max(out, dim=1) # 获取每一行最大值对应索引，即类别
            cor += (prediction == label).sum() # 累加预测正确数目
        acc = cor.item()/len(test_data) # 计算acc
        view_acc.append(acc) # 添加acc进行可视化
        print('epoch: %d, acc: %s' % (i, acc))
        '''模型挑选：80次迭代后进行最优acc挑选'''
        if acc > best_acc:
            best_acc = acc
            if i > 80:
                torch.save(net.state_dict(), 'best_model.pth')
    '''可视化'''
    plt.plot(view_loss, label='Loss')
    plt.plot(view_acc, label='Acc')
    plt.legend()
    plt.savefig('loss_acc.jpg')
    plt.clf()
