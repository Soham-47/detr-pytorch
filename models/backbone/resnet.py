import torch 
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels , out_channels, kernel_size=3, padding=1, stride=1, bias=False)
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
            identity=self.downsample(identity)

        x+=identity

        return self.relu(x)

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):

        super(BottleNeck, self).__init__()

        base_width=64

        width=int(out_channels*(base_width/64))*1

        self.conv1=nn.Conv2d(in_channels, width , kernel_size=1, stride=1 , padding=0, bias=False)
        self.bn1=nn.BatchNorm2d(num_features=width)
        self.conv2=nn.Conv2d(in_channels=width, out_channels=width , kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(num_features=width)
        self.conv3=nn.Conv2d(in_channels=width , out_channels=width*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3=nn.BatchNorm2d(num_features=width*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, x):
        identity=x

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.bn3(x)

        if self.downsample is not None:
            identity=self.downsample(identity)
        
        x+=identity

        return self.relu(x)

class Resnet(nn.Module):

    def __init__(self , block , layers, num_classes):

        super().__init__()
        self.in_channels=64

        self.conv1=nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.adppool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, downsample, stride))
        self.in_channels = out_channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.adppool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)

def resnet50(num_classes=1000):
    return Resnet(BottleNeck, [3, 4, 6, 3], num_classes)

if __name__ == "__main__":
    model = resnet50()
    test_data = torch.randn(1, 3, 224, 224)
    output = model(test_data)
    print(f"Output shape: {output.shape}")
        
        