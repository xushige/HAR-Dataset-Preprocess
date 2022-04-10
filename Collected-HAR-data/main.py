import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
import models_and_tools as t


X_train = torch.from_numpy(np.load('npy/train/X_train.npy'))
X_val = torch.from_numpy(np.load('npy/val/X_val.npy'))
X_test = torch.from_numpy(np.load('npy/test/X_test.npy'))

Y_train = torch.from_numpy(np.load('npy/train/Y_train.npy'))
Y_val = torch.from_numpy(np.load('npy/val/Y_val.npy'))
Y_test = torch.from_numpy(np.load('npy/test/Y_test.npy'))

train_data = data.TensorDataset(X_train, Y_train)
val_data = data.TensorDataset(X_val, Y_val)
test_data = data.TensorDataset(X_test, Y_test)

LR = 5e-4
EP = 300
B_S = 128

train_loader = data.DataLoader(train_data, batch_size=B_S, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=B_S, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=B_S, shuffle=True)

cnn = t.CNN().cuda()
rnn = t.RNN().cuda()
res = t.Resnet().cuda()

optim_dict = {
    torch.optim.Adagrad: 'Adagrad',
    torch.optim.Adam: 'Adam',
    torch.optim.SGD: 'SGD',
    torch.optim.RMSprop: 'RMS'
}

net_dict = {cnn: 'cnn_model.pth.tar',
            rnn: 'rnn_model.pth.tar',
            res: 'res_model.pth.tar'
}

def start(Net=cnn, OPTIMIZER=torch.optim.Adam, train=True, save_file=False):
    net = Net
    optimizer = OPTIMIZER(net.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    if train == True:
        list = []
        for i in range(EP):
            correct_num = 0
            for data, label in train_loader:
                data, label = data.cuda(), label.cuda()
                net.train()
                out = net(data)
                loss = loss_fn(out, label)
                list.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for data, label in val_loader:
                data, label = data.cuda(), label.cuda()
                net.eval()
                out = net(data)
                _, pre = torch.max(out, 1)
                correct_num += (pre == label).sum()
            acc = correct_num.cpu().numpy()/len(val_data)
            # list.append(acc)
            print('第%d次迭代结束，%s模型在%s迭代器下的loss为%f， 正确率为%f' % (i+1, net_dict[Net][:3], optim_dict[OPTIMIZER], loss, acc))
        plt.plot(list, label=net_dict[Net][:3]+'+'+optim_dict[OPTIMIZER])
        if save_file == True:
            save_path = os.path.join(net_dict[Net])
            torch.save({"state_dict": net.state_dict()}, save_path)
        return list

    else:
        load = torch.load(net_dict[Net])
        net.load_state_dict(load['state_dict'])
        correct_num = 0
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            net.eval()
            out = net(data)
            _, pre = torch.max(out, 1)
            correct_num += (pre == label).sum()
        acc = correct_num.cpu().numpy() / len(test_data)
        print('%s模型的正确率为%f' % (net_dict[Net][:3], acc))

def compare_net(epoch, net=False, optim=False):
    global EP
    EP = epoch
    if net != False:
        for each in net_dict.keys():
            start(Net=each, save_file=True)
    if optim != False:
        for each in optim_dict.keys():
            start(OPTIMIZER=each)

start()
