import torch 
import torch.nn as nn
from .resnet import resnet50
from ..positional_encoding.sine_2d import Sine2DPositionalEncoding

class Backbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = resnet50()
        
        self.body = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        
        self.input_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
        self.pos_encoding = Sine2DPositionalEncoding(embed_dim)

    def forward(self, x):
        features = self.body(x)
        proj_features = self.input_proj(features)
        pos = self.pos_encoding(proj_features)
        return proj_features, pos