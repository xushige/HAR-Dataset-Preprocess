import scipy.io as scio
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
'''数据读取'''
dir = './data'
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
    trainlen = int(cur_cate_len*0.7) # 训练集长度，默认训练：测试==7：3
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
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)



'''训练'''
print('\nSTART TRAIN')
X_train = torch.from_numpy(train_data).float()
X_test = torch.from_numpy(test_data).float()
Y_train = torch.from_numpy(train_label).long()
Y_test = torch.from_numpy(test_label).long()

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
category = len(set(Y_test.tolist()))
print(X_train.size(), X_test.size(), category)

train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((2, X_test.size(-1)))
        self.fc = nn.Linear(512*2*X_test.size(-1), category)
    def forward(self, x):
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
net = CNN().cuda()
print(net)
BS = 256
EP = 50

train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BS, shuffle=True)

lr = 5e-4
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

for i in range(EP):
    for img, label in train_loader:
        img, label = img.cuda(), label.cuda()
        net.train()
        out = net(img)
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    cor = 0
    for img, label in test_loader:
        img, label = img.cuda(), label.cuda()
        net.eval()
        out = net(img)
        _, pre = torch.max(out, 1)
        cor += (pre == label).sum()
    acc = cor.item()/len(test_data)
    print('epoch: %d, loss: %f, acc: %f' % (i, loss, acc))