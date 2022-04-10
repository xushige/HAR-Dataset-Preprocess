import numpy as np
import pandas as pd
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flc = nn.Linear(1024, 5)
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        # self.flc = nn.Linear(x.size(1), 5)
        x = self.flc(x)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        self.flc = nn.Linear(512, 5)
    def forward(self, x):
        x, h = self.rnn(x, None)
        x = self.flc(x[:, -1, :])
        return x

class Residual_block(nn.Module):
    def __init__(self, inchannal, outchannal, stride=1):
        super(Residual_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inchannal, outchannal, 3, stride, 1),
            nn.BatchNorm1d(outchannal),
            nn.ReLU(),
            nn.Conv1d(outchannal, outchannal, 3, 1, 1),
            nn.BatchNorm1d(outchannal)
        )
        self.short = nn.Sequential()
        if stride != 1 or inchannal != outchannal:
            self.short = nn.Sequential(
                nn.Conv1d(inchannal, outchannal, 3, stride, 1),
                nn.BatchNorm1d(outchannal)
            )
    def forward(self, x):
        out = self.conv(x) + self.short(x)
        return nn.ReLU()(out)
class Resnet(nn.Module):
    def make_layer(self, inchannal, outchannal, stride, num):
        layer = []
        layer.append(Residual_block(inchannal, outchannal, stride))
        for i in range(1, num):
            layer.append(Residual_block(outchannal, outchannal))
        return nn.Sequential(*layer)
    def __init__(self):
        super(Resnet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(3, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.make_layer(64, 64, 2, 2),
            self.make_layer(64, 128, 2, 2),
            self.make_layer(128, 256, 2, 2),
            self.make_layer(256, 512, 2, 2),
            nn.AvgPool1d(4)
        )
        self.flc = nn.Linear(512*128//16//4, 5)
    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x



def column_to_row(filename):
    f = pd.read_excel(filename, skiprows=1)
    return f.to_numpy(dtype=np.float32).transpose(1, 0)

def window_slide(array, window_size, stride):
    times = (len(array) - window_size)//stride + 1
    temp = []
    for i in range(times):
        temp.append(array[i*stride: i*stride + window_size])
    return np.array(temp)

def unit_row(*array):
    temp = []
    for each in array:
        for eachline in each:
            temp.append(eachline)
    return np.array(temp)

def split_data(array, ratio=(7, 1, 2)):
    array = np.random.permutation(array)
    total = sum(ratio)
    train, test = len(array)*ratio[0]//total, len(array)*ratio[2]//total
    return array[:train], array[train:-test], array[-test:]

def sign_label(list, times, label):
    for i in range(times):
        list.append(label)
    return list

def z_score(array):
    return (array - array.mean())/array.var()

def get_accdata_and_save_label(file_dict, WINDOW_SIZE, STRIDE, AXIS, save_label = False):
    Train, Val, Test = np.array([]), np.array([]), np.array([])
    y_train, y_val, y_test = [], [], []
    for i in range(len(file_dict)):
        train, val, test = split_data(window_slide(z_score(column_to_row('excel/' + file_dict[i] + '.xls')[AXIS]), WINDOW_SIZE, STRIDE))
        Train, Val, Test = unit_row(Train, train), unit_row(Val, val), unit_row(Test, test)
        y_train, y_val, y_test = sign_label(y_train, len(train), i), sign_label(y_val, len(val), i), sign_label(y_test, len(test), i)
    if save_label == True:
        np.save('npy/train/Y_train.npy', np.array(y_train, dtype=np.int64))
        np.save('npy/val/Y_val.npy', np.array(y_val, dtype=np.int64))
        np.save('npy/test/Y_test.npy', np.array(y_test, dtype=np.int64))
    return Train, Val, Test

def unit_axis(*array):
    temp = []
    for each in array:
        temp.append(each)
    return np.array(temp).transpose(1, 0, 2)
