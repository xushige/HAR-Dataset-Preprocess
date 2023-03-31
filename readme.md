## Human Activity Recogniton (HAR) 公开数据集预处理与网络搭建
### 下载工程
```
git clone https://gitcode.net/m0_52161961/HAR-Dataset-Preprocess.git --depth 1
```
### 安装库依赖
```
cd HAR-Dataset-Prerocess
pip3 install -r requirements.txt
```
### 模型训练代码运行样例【或者直接编译器运行train.py文件，在文件中修改参数:--dataset, --model】
```
python3 train.py --dataset wisdm --model vit
```
#### --dataset choose from 【'uci', 'unimib', 'usc', 'pamap', 'wisdm', 'dasa', 'oppo'】
#### --model choose from 【'cnn', 'resnet', 'res2net', 'resnext', 'sknet', 'resnest', 'lstm', 'ca', 'sa', 'dilation', 'depthwise', 'shufflenet', 'dcn', 'vit', 'swin'】 
<details open>
<summary>包含数据集</summary>

- [x] [Daily-and-Sports-Activities](http://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)
- [x] [PAMAP2](http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
- [x] [UCI-HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [x] [USC-HAD](https://sipi.usc.edu/had/)
- [x] [UniMiB-SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/)
- [x] [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
- [x] [OPPORTUNITY](http://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition)

</details>

<details open>
<summary>支持模型</summary>

- [x] ['cnn': cnn.CNN](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/cnn.py)
- [x] ['resnet': resnet.ResNet](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/resnet.py)
- [x] ['res2net': res2net.Res2Net](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/res2net.py)
- [x] ['resnext': resnext.ResNext](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/resnext.py)
- [x] ['sknet': sk_resnet.SKResNet](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/sk_resnet.py)
- [x] ['resnest': resnest.ResNeSt](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/resnest.py)
- [x] ['lstm': lstm.LSTM](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/lstm.py)
- [x] ['ca': channel_attention.ChannelAttentionNeuralNetwork](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/channel_attention.py)
- [x] ['sa': spatial_attention.SpatialAttentionNeuralNetwork](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/spatial_attention.py)
- [x] ['dilation': dilated_conv.DilatedConv](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/dilated_conv.py)
- [x] ['depthwise': depthwise_conv.DepthwiseConv](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/depthwise_conv.py)
- [x] ['shufflenet': shufflenet.ShuffleNet](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/shufflenet.py)
- [x] ['dcn': dcn.DeformableConvolutionalNetwork](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/dcn.py)
- [x] ['vit': vit.VisionTransformer](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/vit.py)
- [x] ['swin': swin.SwinTransformer](https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/swin.py)

</details>

### 仅预处理代码运行样例, 这里以WISDM数据集为例【或者直接编译器运行train.py文件，在文件中修改参数】
```
python3 WISDM/dataproc.py
```
