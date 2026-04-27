import torch 
import torch.nn as nn
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self , embed_dim , num_heads , dim_feedforward):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim , num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim , dim_feedforward),
            nn.ReLU() , 
            nn.Linear(dim_feedforward , embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self , src , pos=None):
        if pos is not None:
            src = src + pos
        q = k = src
        src2 = self.self_attn (q , k , v = src)
        src = src + src2
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self , embed_dim , num_heads , dim_feedforward , num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim , num_heads , dim_feedforward) for _ in range(num_layers)])

    def forward(self , src , pos=None):
        for layer in self.layers:
            src = layer(src , pos)
        return src