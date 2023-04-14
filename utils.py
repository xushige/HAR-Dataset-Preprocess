import requests
import shutil
from clint.textui import progress
import os
import numpy as np
from collections import Counter

def download_dataset(dataset_name, file_url, dataset_dir):
    '''
        数据集下载
    '''
    # 检查是否存在源数据
    if os.path.exists(dataset_dir):
        return 

    print('\n==================================================【 %s 数据集下载】===================================================\n'%(dataset_name))
    print('url: %s\n'%(file_url))

    dir_path = dataset_dir.split('/')[0]
    if dataset_name == 'UniMiB-SHAR' and file_url[-4:] == '.git': # 由于unimib数据集无法直接访问下载，这里我把unimib数据集上传到github进行访问clone
        if os.path.exists(os.path.join(dir_path, dataset_name)):
            shutil.rmtree(os.path.join(dir_path, dataset_name))
        os.system('git clone %s %s/%s' % (file_url, dir_path, dataset_name))
   
    else: # 其他数据集
        r = requests.get(url=file_url, stream=True)
        with open(os.path.join(dir_path, 'dataset.zip'),mode='wb') as f:  # 需要用wb模式
            total_leng = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_leng / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        for format in ["zip", "tar", "gztar", "bztar", "xztar"]:
            try:
                shutil.unpack_archive(filename=os.path.join(dir_path, 'dataset.zip'),extract_dir=dir_path, format=format)
                break
            except:
                continue
        os.remove(os.path.join(dir_path, r'dataset.zip'))

    print()

    # 检查数据集是否下载完毕
    if not os.path.exists(dataset_dir):
        quit('数据集下载失败，请检查url与网络后重试')


def build_npydataset_readme(path):
    '''
        构建数据集readme
    '''
    datasets = os.listdir(path) 
    curdir = os.curdir # 记录当前地址
    os.chdir(path) # 进入所有npy数据集根目录
    with open('readme.md', 'w') as w:
        for dataset in datasets:
            if not os.path.isdir(dataset):
                continue
            x_train = np.load('%s/x_train.npy' % (dataset))
            x_test = np.load('%s/x_test.npy' % (dataset))
            y_train = np.load('%s/y_train.npy' % (dataset))
            y_test = np.load('%s/y_test.npy' % (dataset))
            category = len(set(y_test.tolist()))
            d = Counter(y_test)
            new_d = {} # 顺序字典
            for i in range(category):
                new_d[i] = d[i]
            log = '\n===============================================================\n%s\n   x_train shape: %s\n   x_test shape: %s\n   y_train shape: %s\n   y_test shape: %s\n\n共【%d】个类别\ny_test中每个类别的样本数为 %s\n' % (dataset, x_train.shape, x_test.shape, y_train.shape, y_test.shape, category, new_d)
            w.write(log)
    os.chdir(curdir) # 返回原始地址

def sliding_window(array, windowsize, overlaprate):
    '''
    array: 二维数据(n, m)，n为时序长度，m为模态轴数
    windowsize: 窗口尺寸
    overlaprate: 重叠率
    '''
    stride = int(windowsize * (1 - overlaprate)) # 计算stride
    times = (len(array)-windowsize)//stride + 1 # 滑窗次数，同时也是滑窗后数据长度
    res = []
    for i in range(times):
        x = array[i*stride : i*stride+windowsize]
        res.append(x)
    return res