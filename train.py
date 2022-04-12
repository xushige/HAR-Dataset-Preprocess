import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append(r'..\HAR-Dataset-Prerocess')
from model import *
from Daily_and_Sports_Activities_dataset.dataproc import DASA
from UniMib_SHAR.dataproc import UNIMIB
from PAMAP2.dataproc import PAMAP
from UCI_HAR.dataproc import UCI
from USC_HAD.dataproc import USC
from WISDM.dataproc import WISDM

def parse_args():
    parser = argparse.ArgumentParser(description='Train a HAR task')
    parser.add_argument(
        '--dataset', 
        help='select dataset', 
        choices=['uci', 'unimib', 'usc', 'pamap', 'wisdm', 'dasa']
        )
    parser.add_argument('--datadir', help='the dir-path of the unpreprocessed data')
    parser.add_argument(
        '--model', 
        help='select network', 
        choices=['cnn', 'resnet', 'lstm'],
        default='cnn'
        )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    dataset_dict = {
        'uci': UCI,
        'unimib': UNIMIB,
        'pamap': PAMAP,
        'usc': USC,
        'dasa': DASA,
        'wisdm': WISDM
    }

    model_dict = {
        'cnn':CNN,
        'resnet': ResNet,
        'lstm': LSTM
    }
    GPU = torch.cuda.is_available()
    BS = 128
    EP = 50
    LR = 5e-4

    '''数据集加载'''
    print('\n==================================================【数据集预处理】===================================================\n')
    train_data, test_data, train_label, test_label = dataset_dict[args.dataset](dataset_dir=args.datadir)


    '''数据准备'''
    X_train = torch.from_numpy(train_data).float().unsqueeze(1)
    X_test = torch.from_numpy(test_data).float().unsqueeze(1)
    Y_train = torch.from_numpy(train_label).long()
    Y_test = torch.from_numpy(test_label).long()

    category = len(set(Y_test.tolist()))
    print('\n==================================================  【张量转换】  ===================================================\n')
    print('x_train_tensor shape: %s\nx_test_tensor shape: %s'%(X_train.shape, X_test.shape))
    print('Category num: %d'%(category))
    print('If GPU: 【%s】'%(GPU))


    '''模型加载'''
    print('\n==================================================  【模型加载】  ===================================================\n')
    net = model_dict[args.model](X_train.shape, category)
    if GPU:
        net.cuda()
    print(net)

    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BS, shuffle=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0001)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, EP//3, 0.5)
    loss_fn = nn.CrossEntropyLoss()

    '''训练'''
    print('\n==================================================   【训练】   ===================================================\n')
    for i in range(EP):
        net.train()
        for data, label in train_loader:
            if GPU:
                data, label = data.cuda(), label.cuda()
            out = net(data)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        cor = 0
        for data, label in test_loader:
            if GPU:
                data, label = data.cuda(), label.cuda()
            out = net(data)
            _, pre = torch.max(out, 1)
            cor += (pre == label).sum()
        acc = cor.item()/len(Y_test)
        print('epoch: %d, train-loss: %f, test-acc: %f' % (i, loss, acc))