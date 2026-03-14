import torch 
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self , in_channels ,out_channels,downsample=None , stride=1 ):
        super(Block, self).__init__()
        self.expansion=1

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(in_channels , out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample

    def forward(self , x):

        identity=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)

        if self.downsample is not None:
            identity=self.downsample(x)

        x+=identity

        return self.relu(x)

class BottleNeck(nn.Module):
    def __init__(self, in_channels , out_channels , kernel_size=3, stride=1, downsample=None):

        super(BottleNeck, self).__init__()
        self.expansion=4

        base_width=64

        width=int(out_channels*(base_width/64))*1

        self.conv1=nn.Conv2d(in_channels, width , kernel_size=1, stride=stride , padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(num_features=width)
        self.conv2=nn.Conv2d(in_channels=width, out_channels=width , kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(num_features=width)
        self.conv3=nn.Conv2d(in_channels=width , out_channels=width*self.expansion, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn3=nn.BatchNorm2d(num_features=width*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, x):
        identity=x

        x=self.bn1(self.conv1(x))
        x=self.bn2(self.conv2(x))
        x=self.bn3(self.conv3(x))

        if downsample is not None:
            identity=self.downsample(x)
        
        x+=identity

        return self.relu(x)