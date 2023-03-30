import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
os.chdir(sys.path[0])
from models import cnn, resnet, res2net, resnext, sk_resnet, resnest, lstm, dilated_conv, depthwise_conv, shufflenet, vit, dcn, channel_attention, spatial_attention, swin
from Daily_and_Sports_Activities.dataproc import DASA
from UniMiB_SHAR.dataproc import UNIMIB
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
        choices=dataset_dict.keys(),
        default='wisdm'
        )
    parser.add_argument(
        '--model', 
        help='select network', 
        choices=model_dict.keys(),
        default='cnn'
        )
    parser.add_argument('--savepath', help='the dir-path of the .npy array for saving', default='../HAR-datasets') # 如无需保存npy形式数据集，将default置空字符串
    parser.add_argument('--batch', type=int, help='batch_size', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=100)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0005)
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
        'unimib': 'UniMiB_SHAR/UniMiB-SHAR/data',
        'pamap': 'PAMAP2/PAMAP2_Dataset/Protocol',
        'usc': 'USC_HAD/USC-HAD',
        'dasa': 'Daily_and_Sports_Activities/data',
        'wisdm': 'WISDM/WISDM_ar_v1.1',
        'oppo': 'OPPORTUNITY/OpportunityUCIDataset/dataset'
    }
    model_dict = {
        'cnn':cnn.CNN, 
        'resnet': resnet.ResNet,
        'res2net': res2net.Res2Net,
        'resnext': resnext.ResNext,
        'sknet': sk_resnet.SKResNet,
        'resnest': resnest.ResNeSt,
        'lstm': lstm.LSTM,
        'ca': channel_attention.ChannelAttentionNeuralNetwork,
        'sa': spatial_attention.SpatialAttentionNeuralNetwork,
        'dilation': dilated_conv.DilatedConv,
        'depthwise': depthwise_conv.DepthwiseConv,
        'shufflenet': shufflenet.ShuffleNet,
        'dcn': dcn.DeformableConvolutionalNetwork,
        'vit': vit.VisionTransformer,
        'swin': swin.SwinTransformer
    }
    args = parse_args()
    args.savepath = os.path.abspath(args.savepath) if args.savepath else '' # 如果有npy保存路径就转换为绝对路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU or CPU【若有多块GPU需要指定0号GPU，则可以将“cuda”改为“cuda:0”】
    BS = args.batch
    EP = args.epoch
    LR = args.lr
    
    print('\n==================================================【HAR 训练任务开始】===================================================\n')
    print('Dataset Name:【%s】\nModel:【%s】\nNumpy array save path:【%s】\nEpoch:【%d】\nBatch Size:【%d】\nInitial Learning Rate:【%f】\nDevice: 【%s】' % (args.dataset, args.model, args.savepath, args.epoch, args.batch, args.lr, device))
        
    '''数据集加载'''
    print('\n==================================================【数据集预处理】===================================================\n')
    dataset_name = dir_dict[args.dataset].split('/')[0]
    dataset_saved_path = os.path.join(args.savepath, dataset_name)

    # 获取训练与测试【数据，标签】
    if os.path.exists(dataset_saved_path): # npy数据集已存在则直接读取
        print('\n【%s】数据集在【%s】目录下已存在npy文件，直接读取...\n'%(dataset_name, dataset_saved_path))
        train_data, test_data, train_label, test_label = np.load('%s/x_train.npy'%(dataset_saved_path)), np.load('%s/x_test.npy'%(dataset_saved_path)), np.load('%s/y_train.npy'%(dataset_saved_path)), np.load('%s/y_test.npy'%(dataset_saved_path))
    else: # npy数据集不存在则进行数据预处理获取
        train_data, test_data, train_label, test_label = dataset_dict[args.dataset](dataset_dir=dir_dict[args.dataset], SAVE_PATH=args.savepath)

    '''npy数据tensor化'''
    X_train = torch.from_numpy(train_data).float().unsqueeze(1)
    X_test = torch.from_numpy(test_data).float().unsqueeze(1)
    Y_train = torch.from_numpy(train_label).long()
    Y_test = torch.from_numpy(test_label).long()

    category = len(set(Y_test.tolist()))
    print('\n==================================================  【张量转换】  ===================================================\n')
    print('x_train_tensor shape: %s\nx_test_tensor shape: %s'%(X_train.shape, X_test.shape))
    print('Category num: %d'%(category))

    '''模型加载'''
    print('\n==================================================  【模型加载】  ===================================================\n')
    net = model_dict[args.model](X_train.shape, category).to(device)
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
            data, label = data.to(device), label.to(device)
            out = net(data)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        cor = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = net(data)
            _, pre = torch.max(out, 1)
            cor += (pre == label).sum()
        acc = cor.item()/len(Y_test)
        print('epoch: %d, train-loss: %f, val-acc: %f' % (i, loss, acc))
