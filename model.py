import torch.nn as nn
import torch

'''Convolutional Neural Network'''
class CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(CNN, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (9, 1), (2, 1), (4, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512*train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

'''Resdual Neural Network'''
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 1, 1, 0),
            nn.BatchNorm2d(outchannel)
        )
        self.short = nn.Sequential()
        if (inchannel != outchannel or stride != 1):
            self.short = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        out = self.block(x) + self.short(x)
        return nn.ReLU()(out)
    
class ResNet(nn.Module):
    def __init__(self, train_shape, category):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer1 = self.make_layers(1, 64, 2, 2)
        self.layer2 = self.make_layers(64, 128, 2, 2)
        self.layer3 = self.make_layers(128, 256, 2, 2)
        self.layer4 = self.make_layers(256, 512, 2, 2)
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512*train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def make_layers(self, inchannel, outchannel, stride, blocks):
        layer = [Block(inchannel, outchannel, stride)]
        for i in range(1, blocks):
            layer.append(Block(outchannel, outchannel, 1))
        return nn.Sequential(*layer)

'''Long Short Term Memory Network'''
class LSTM(nn.Module):
    def __init__(self, train_shape, category):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.lstm = nn.LSTM(train_shape[-1], 512, 2, batch_first=True)
        self.fc = nn.Linear(512, category)
        
    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        x, _ = self.lstm(x.squeeze(1))
        x = x[:, -1, :]
        x = self.fc(x)
        return x

'''Multi Self-Attention: Transformer'''
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=4, att_size=64):
        super().__init__()
        '''
            input_dim: 输入维度, 即embedding维度
            output_dim: 输出维度
            head_num: 多头自注意力
            att_size: QKV矩阵维度
        '''
        self.head_num = head_num
        self.att_size = att_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.key = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.value = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.att_mlp = nn.Sequential(
            nn.Linear(head_num*att_size, input_dim),
            nn.LayerNorm(input_dim)
        ) # 恢复输入维度
        self.forward_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        ) # 变为输出维度
        
    def forward(self, x):
        '''
            x.shape: [batch, patch_num, input_dim]
        '''
        batch, patch_num, input_dim = x.shape
        # Q, K, V
        query = self.query(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 1, 3) # [batch, head_num, patch_num, att_size]
        key = self.key(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 3, 1) # [batch, head_num, att_size, patch_num]
        value = self.value(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 1, 3) # [batch, head_num, patch_num, att_size]
        # Multi Self-Attention Score
        z = torch.matmul(nn.Softmax(dim=-1)(torch.matmul(query, key) / (self.att_size ** 0.5)), value) # [batch, head_num, patch_num, att_size]
        z = z.permute(0, 2, 1, 3).reshape(batch, patch_num, -1) # [batch, patch_num, head_num*att_size]
        # Forward
        z = nn.ReLU()(x + self.att_mlp(z)) # [batch, patch_num, input_dim]
        out = nn.ReLU()(self.forward_mlp(z)) # [batch, patch_num, output_dim]
        return out

class Transformer(nn.Module):
    def __init__(self, train_shape, category, embedding_dim=512):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
            embedding_dim: embedding 维度
        '''
        # cut patch
        # 对于传感窗口数据来讲，模态轴的patch数应该等于轴数，而时序轴这里按算术平方根进行切patch
        # 例如 uci-har 数据集窗口尺寸为 [128, 9]，patch_num 应当为 [11, 9], 总patchs数为 11*9=99
        self.patch_num = (int(train_shape[-2]**0.5), train_shape[-1]) 
        self.all_patchs = self.patch_num[0] * self.patch_num[1]
        self.kernel_size = (train_shape[-2]//self.patch_num[0], train_shape[-1]//self.patch_num[1])
        self.stride = self.kernel_size
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0
        )
        # 位置信息
        self.position_embedding = nn.Parameter(torch.zeros(1, self.all_patchs, embedding_dim))
        # Multi Self-Attention Layer
        self.msa_layer = nn.Sequential(
            TransformerBlock(embedding_dim, embedding_dim//2), # 4层多头注意力层，每层输出维度下降 1/2
            TransformerBlock(embedding_dim//2, embedding_dim//4),
            TransformerBlock(embedding_dim//4, embedding_dim//8),
            TransformerBlock(embedding_dim//8, embedding_dim//16)
        )
        # classification
        self.dense_tower = nn.Sequential(
            nn.Linear(self.patch_num[0] * self.patch_num[1] * embedding_dim//16, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, category)
        )

    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        x = self.patch_conv(x) # [batch, embedding_dim, patch_num[0], patch_num[1]]
        x = self.position_embedding + x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # [batch, all_patchs, embedding_dim]
        x = self.msa_layer(x)
        x = nn.Flatten()(x)
        x = self.dense_tower(x)
        return x