import os
import numpy as np

def txt_to_numpy(filename):
    with open(filename, 'r') as f:
        x = []
        for eachline in f:
            x.append(eachline.strip().split(','))
        print(np.array(x).shape)
        return x
def get_X():
    '''处理原始数据'''
    result_array = []
    move = os.listdir('data')
    os.chdir('data')
    print(move)
    for eachmove in move:
        people = os.listdir(eachmove)
        print(people)
        os.chdir(eachmove)
        for eachpeople in people:
            file = os.listdir(eachpeople)
            print(file)
            os.chdir(eachpeople)
            for eachfile in file:
                result_array.append(txt_to_numpy(eachfile))
            os.chdir('../')
        os.chdir('../')
    os.chdir('../')
    X = np.array(result_array, dtype=np.float32)
    print(X.shape)

    np.save('X', X)


def split_data(X, ratio=(8, 2)):
    '''数据集切分'''
    train_num = int(480*ratio[0]/sum(ratio))
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for i in range(19):
        x = np.random.permutation(X[480*i:480*(i+1)])
        for j in range(len(x)):
            if j < train_num:
                X_train.append(x[j])
                Y_train.append(i)
            else:
                X_test.append(x[j])
                Y_test.append(i)
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(Y_train, dtype=np.int64), np.array(Y_test, dtype=np.int64)

X = np.load('X.npy')
x_train, x_test, y_train, y_test = split_data(X)
np.save('x_train', x_train)
np.save('x_test', x_test)
np.save('y_train', y_train)
np.save('y_test', y_test)
