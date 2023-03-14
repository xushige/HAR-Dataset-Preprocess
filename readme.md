## Human Activity Recogniton (HAR) 公开数据集预处理与网络搭建
### 如有问题或者优化之处，欢迎留言交流或发起 merge request
### 包含数据集
* Daily-and-Sports-Activities-dataset   http://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities
* PAMAP2 dataset    http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
* UCI-HAR dataset   https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
* USC-HAD dataset   https://sipi.usc.edu/had/
* UniMiB-SHAR dataset   http://www.sal.disco.unimib.it/technologies/unimib-shar/
* WISDM dataset   https://www.cis.fordham.edu/wisdm/dataset.php
* OPPORTUNITY dataset   http://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

### 预处理代码运行样例
```
$ cd HAR-Dataset-Preprocess/UniMib_SHAR
$ python dataproc.py
```

### 预处理+模型训练代码运行样例
```
$ cd HAR-Dataset-Prerocess
$ python train.py --dataset wisdm --model vit
```
#### --dataset 【Required】 choose from 【'uci', 'unimib', 'usc', 'pamap', 'wisdm', 'dasa', 'oppo'】
#### --model  choose from 【'cnn', 'resnet', 'lstm', 'vit'】 
