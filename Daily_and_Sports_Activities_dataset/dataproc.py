import os
import numpy as np

'''
WINDOW_SIZE=125 # int
OVERLAP_RATE=-- # float in [0，1）
SPLIT_RATE=(7,3) # tuple or list
'''

def DASA(SPLIT_RATE=(7,3), dataset_dir='data'):

    if not os.path.exists(dataset_dir):
        print('HAR-Dataset-Preprocess工程克隆不完整，请重新clone')
        quit()

    def txt_to_numpy(filename):
        with open(filename, 'r') as f:
            x = []
            for eachline in f:
                x.append(eachline.strip().split(','))
            return x
    def get_X():
        '''处理原始数据'''
        result_array = []
        move = os.listdir(dataset_dir)
        os.chdir(dataset_dir)
        print(move)
        for eachmove in move:
            print('======================================================\n         current activity sequence: 【%s】\n'%(eachmove))
            people = os.listdir(eachmove)
            os.chdir(eachmove)
            for eachpeople in people:
                file = os.listdir(eachpeople)
                os.chdir(eachpeople)
                for eachfile in file:
                    result_array.append(txt_to_numpy(eachfile))
                os.chdir('../')
            os.chdir('../')
        os.chdir('../')
        X = np.array(result_array, dtype=np.float32)

        return X


    def split_data(X, ratio):
        '''数据集切分'''
        train_num = int(480*ratio[0]/sum(ratio))
        X_train, X_test, Y_train, Y_test = [], [], [], []
        for i in range(19):
            x = np.random.permutation(X[480*i:480*(i+1)]).tolist()
            X_train += x[:train_num]
            Y_train += [i] * train_num
            X_test += x[train_num:]
            Y_test += [i] * (480-train_num)
        return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(Y_train, dtype=np.int64), np.array(Y_test, dtype=np.int64)

    xtrain, xtest, ytrain, ytest = split_data(get_X(), ratio=SPLIT_RATE)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':
    DASA()