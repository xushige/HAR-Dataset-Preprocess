import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append(r'../HAR-Dataset-Prerocess')
from model import *
from utils import *
from Daily_and_Sports_Activities_dataset.dataproc import DASA
from UniMib_SHAR.dataproc import UNIMIB
from PAMAP2.dataproc import PAMAP
from UCI_HAR.dataproc import UCI
from USC_HAD.dataproc import USC
from WISDM.dataproc import WISDM
from OPPORTUNITY.dataproc import OPPO

def parse_args():
    parser = argparse.ArgumentParser(description='Train a HAR task')
    parser.add_argument(
        '--dataset', 
        help='select dataset', 
        choices=dataset_dict.keys()
        )
    parser.add_argument('--datadir', help='the dir-path of the unpreprocessed data', default=None)
    parser.add_argument('--savepath', help='the dir-path of the .npy array for saving', default='')
    parser.add_argument(
        '--model', 
        help='select network', 
        choices=model_dict.keys(),
        default='cnn'
        )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    dataset_dict = {
        'uci': UCI,
        'unimib': UNIMIB,
        'pamap': PAMAP,
        'usc': USC,
        'dasa': DASA,
        'wisdm': WISDM,
        'oppo': OPPO
    }
    dir_dict = {
        'uci': 'UCI_HAR/UCI HAR Dataset',
        'unimib': 'UniMib_SHAR/UniMiB-SHAR/data',
        'pamap': 'PAMAP2/PAMAP2_Dataset/Protocol',
        'usc': 'USC_HAD/USC-HAD',
        'dasa': 'Daily_and_Sports_Activities_dataset/data',
        'wisdm': 'WISDM/WISDM_ar_v1.1',
        'oppo': 'OPPORTUNITY/OpportunityUCIDataset/dataset'
    }
    download_url_dict = {
        'uci': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
        'unimib': 'https://cvws.icloud-content.com.cn/B/ASdy4_6QGJ8i4pPMNoToriXa7nm4Ad2hwwtu0yNk4eNttEynQCd9X1EZ/UniMiB-SHAR.zip?o=AtP30jip98ZRj3bph_QGRSHp5K8hib0H88DkKagY41mW&v=1&x=3&a=CAog78Weas7GEz0BczfsnMZV1SI43oldlgaASpmlM8r_eLoSbRCe1N3S7TAYnrG51O0wIgEAUgTa7nm4WgR9X1EZaiYRDIJkgaw09Jutb9qZMQewIAJ1oECaxeg1R5I1BGCKB6X3Oko9w3ImBB7WRjWtdPvINsX8jxhjUZhVt6pwqW4PCiNoqJFmBIM70IJtFtM&e=1678704007&fl=&r=b9662625-0fac-4d32-8192-0a2527e122c8-1&k=xHD-RvY5fOZ1Of642JJmzw&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=213&s=8WB3xXc9yuzOZDqulVO0Ee1SyKo&+=037e2f38-61ef-4500-809a-9cb1dc2b4141',
        'pamap': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip',
        'usc': 'https://sipi.usc.edu/had/USC-HAD.zip',
        'dasa': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip',
        'wisdm': 'https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz',
        'oppo': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
    }
    model_dict = {
        'cnn':CNN,
        'resnet': ResNet,
        'lstm': LSTM,
        'vit': VisionTransformer
    }
    args = parse_args()
    GPU = torch.cuda.is_available()
    BS = 128
    EP = 50
    LR = 5e-4
    print('\n==================================================【HAR 训练任务开始】===================================================\n')
    print('Dataset_name: 【%s】\nRaw_data direction: 【%s】\nNumpy array save path: 【%s】\nModel: 【%s】' % (args.dataset, args.datadir, args.savepath, args.model))
    # 默认原始数据路径
    if args.datadir == None:
        print('\n未指定数据集读取路径，选取默认路径:【%s】' % (dir_dict[args.dataset]))
        args.datadir = dir_dict[args.dataset]
    # 下载数据集
    if not os.path.exists(args.datadir):
        download_dataset(
            dataset_name=args.dataset,
            file_url=download_url_dict[args.dataset],
            dir_path=args.datadir.split('/')[0]
        )
    '''数据集加载'''
    print('\n==================================================【数据集预处理】===================================================\n')
    # 获取训练与测试【数据，标签】
    train_data, test_data, train_label, test_label = dataset_dict[args.dataset](dataset_dir=args.datadir, SAVE_PATH=args.savepath)

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

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
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
        print('epoch: %d, train-loss: %f, val-acc: %f' % (i, loss, acc))
