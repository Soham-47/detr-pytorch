import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers):
        super().__init__()
        self.encoder = Encoder(embed_dim, num_heads, dim_feedforward, num_layers)
        self.decoder = Decoder(embed_dim, num_heads, dim_feedforward, num_layers)

    def forward(self, src, query_pos, pos):
        B, C, H, W = src.shape
        
        src = src.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)
        memory = self.encoder(src, pos=pos)
        tgt = torch.zeros_like(query_pos)
        hs = self.decoder(tgt, memory, query_pos=query_pos, pos=pos)
        
        return hs