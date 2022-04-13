## Human Activity Recogniton (HAR) 公开数据集预处理与网络搭建

### 包含数据集
* phyphox软件自采集excel数据
* Daily-and-Sports-Activities-dataset
* PAMAP2 dataset        http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
* UCI-HAR dataset       https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
* USC-HAD dataset
* UniMiB-SHAR dataset   http://www.sal.disco.unimib.it/technologies/unimib-shar/
* WISDM dataset         https://www.cis.fordham.edu/wisdm/dataset.php
* OPPORTUNITY dataset   http://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition


### 预处理代码运行样例
```
$ cd UniMiB-SHAR
$ python dataproc.py
```

### 预处理+模型训练代码运行样例
```
$ cd HAR-Dataset-Prerocess
$ python train.py --dataset uci --datadir ./UCI_HAR/UCI_HAR_Dataset --model resnet
```
