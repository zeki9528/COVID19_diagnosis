import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ResNet
class ResNetModel(nn.Module):
    """
    实现通用的ResNet模块，可根据需要定义
    """
    def __init__(self, num_classes=1000, layer_num=[],bottleneck = False):
        super(ResNetModel, self).__init__()

        #conv1
        self.pre = nn.Sequential(
            #in 224*224*3
            nn.Conv2d(3,64,7,2,3,bias=False),   #输入通道3，输出通道64，卷积核7*7*64，步长2,根据以上计算出padding=3
            #out 112*112*64
            nn.BatchNorm2d(64),     #输入通道C = 64

            nn.ReLU(inplace=True),  #inplace=True, 进行覆盖操作
            # out 112*112*64
            nn.MaxPool2d(3,2,1),    #池化核3*3，步长2,计算得出padding=1;
            # out 56*56*64
        )

        if bottleneck:  #resnet50以上使用BottleNeckBlock
            self.residualBlocks1 = self.add_layers(64, 256, layer_num[0], 64, bottleneck=bottleneck)
            self.residualBlocks2 = self.add_layers(128, 512, layer_num[1], 256, 2,bottleneck)
            self.residualBlocks3 = self.add_layers(256, 1024, layer_num[2], 512, 2,bottleneck)
            self.residualBlocks4 = self.add_layers(512, 2048, layer_num[3], 1024, 2,bottleneck)

            self.fc = nn.Linear(2048, num_classes)
        else:           #resnet34使用普通ResidualBlock
            self.residualBlocks1 = self.add_layers(64,64,layer_num[0])
            self.residualBlocks2 = self.add_layers(64,128,layer_num[1])
            self.residualBlocks3 = self.add_layers(128,256,layer_num[2])
            self.residualBlocks4 = self.add_layers(256,512,layer_num[3])
            self.fc = nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.reset_parameters()

    def add_layers(self, inchannel, outchannel, nums, pre_channel=64, stride=1, bottleneck=False):
        layers = []
        if bottleneck is False:

            #添加大模块首层, 首层需要判断inchannel == outchannel ?
            #跨维度需要stride=2，shortcut也需要1*1卷积扩维

            layers.append(ResidualBlock(inchannel,outchannel))

            #添加剩余nums-1层
            for i in range(1,nums):
                layers.append(ResidualBlock(outchannel,outchannel))
            return nn.Sequential(*layers)
        else:   #resnet50使用bottleneck
            #传递每个block的shortcut，shortcut可以根据是否传递pre_channel进行推断

            #添加首层,首层需要传递上一批blocks的channel
            layers.append(BottleNeckBlock(inchannel,outchannel,pre_channel,stride))
            for i in range(1,nums): #添加n-1个剩余blocks，正常通道转换，不传递pre_channel
                layers.append(BottleNeckBlock(inchannel,outchannel))
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.residualBlocks1(x)
        x = self.residualBlocks2(x)
        x = self.residualBlocks3(x)
        x = self.residualBlocks4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    '''
    定义普通残差模块
    resnet34为普通残差块，resnet50为瓶颈结构
    '''
    def __init__(self, inchannel, outchannel, stride=1, padding=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        #resblock的首层，首层如果跨维度，卷积stride=2，shortcut需要1*1卷积扩维
        if inchannel != outchannel:
            stride= 2
            shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )

        # 定义残差块的左部分
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, 3, 1, padding, bias=False),
            nn.BatchNorm2d(outchannel),

        )

        #定义右部分
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return F.relu(out)

class BottleNeckBlock(nn.Module):
    '''
    定义resnet50的瓶颈结构
    '''
    def __init__(self,inchannel,outchannel, pre_channel=None, stride=1,shortcut=None):
        super(BottleNeckBlock, self).__init__()
        #首个bottleneck需要承接上一批blocks的输出channel
        if pre_channel is None:     #为空则表示不是首个bottleneck，
            pre_channel = outchannel    #正常通道转换


        else:   # 传递了pre_channel,表示为首个block，需要shortcut
            shortcut = nn.Sequential(
                nn.Conv2d(pre_channel,outchannel,1,stride,0,bias=False),
                nn.BatchNorm2d(outchannel)
            )

        self.left = nn.Sequential(
            #1*1,inchannel
            nn.Conv2d(pre_channel, inchannel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            #3*3,inchannel
            nn.Conv2d(inchannel,inchannel,3,1,1,bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            #1*1,outchannel
            nn.Conv2d(inchannel,outchannel,1,1,0,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        return F.relu(out+residual)


# GoogleNet
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=2):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.conv3 = nn.Conv2d(32,120,kernel_size=5)
        self.drop2d = nn.Dropout2d()
        self.fc1 = nn.Linear(120,64)
        self.fc2 = nn.Linear(64,2)
        self.apply(self.init_weights)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop2d(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),25))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, mean=0, std=0.5)
        elif type(layer) == nn.Linear:
            nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
            nn.init.constant_(layer.bias, 0.1)

def get_model(opt):
    # channel_nums = [64,128,256,512,1024,2048]
    num_classes = 2
    #layers = 18, 34, 50, 101, 152
    layer_nums = [[2,2,2,2],[3,4,6,3],[3,4,6,3],[3,4,23,3],[3,8,36,3]]
    if opt.model_name == 'Resnet_18':
        #选择resnet版本：resnet18 ——0；resnet34——1,resnet-50——2,resnet-101——3,resnet-152——4
        i = 0
        bottleneck = i >= 2   #i<2, false,使用普通的ResidualBlock; i>=2，true,使用BottleNeckBlock
        model = ResNetModel(num_classes,layer_nums[i],bottleneck)
    elif opt.model_name == 'Resnet_34':
        #选择resnet版本：resnet18 ——0；resnet34——1,resnet-50——2,resnet-101——3,resnet-152——4
        i = 1
        bottleneck = i >= 2   #i<2, false,使用普通的ResidualBlock; i>=2，true,使用BottleNeckBlock
        model = ResNetModel(num_classes,layer_nums[i],bottleneck)
    elif opt.model_name == 'Resnet_50':
        #选择resnet版本：resnet18 ——0；resnet34——1,resnet-50——2,resnet-101——3,resnet-152——4
        i = 2
        bottleneck = i >= 2   #i<2, false,使用普通的ResidualBlock; i>=2，true,使用BottleNeckBlock
        model = ResNetModel(num_classes,layer_nums[i],bottleneck)
    elif opt.model_name == 'Resnet_101':
        #选择resnet版本：resnet18 ——0；resnet34——1,resnet-50——2,resnet-101——3,resnet-152——4
        i = 3
        bottleneck = i >= 2   #i<2, false,使用普通的ResidualBlock; i>=2，true,使用BottleNeckBlock
        model = ResNetModel(num_classes,layer_nums[i],bottleneck)
    elif opt.model_name == 'Resnet_152':
        #选择resnet版本：resnet18 ——0；resnet34——1,resnet-50——2,resnet-101——3,resnet-152——4
        i = 4
        bottleneck = i >= 2   #i<2, false,使用普通的ResidualBlock; i>=2，true,使用BottleNeckBlock
        model = ResNetModel(num_classes,layer_nums[i],bottleneck)
    elif opt.model_name == 'GoogleNet':
        model = GoogleNet(num_class=num_classes)
        
        return model
    