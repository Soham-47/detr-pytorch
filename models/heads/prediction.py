import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self , input_dim , hidden_dim ,output_dim , num_layers):
        super().__init__()
        self.layers = num_layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self , x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self , embed_dim , num_classes , num_heads):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.class_embed = nn.Linear(embed_dim , num_classes + 1)
        self.bbox_embed = MLP(embed_dim , embed_dim , 4 , 3)

    def forward(self , tgt):
        class_pred = self.class_embed(tgt)
        bbox_pred = self.bbox_embed(tgt).sigmoid()
        return class_pred , bbox_pred
