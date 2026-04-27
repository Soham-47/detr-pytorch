import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self ,embed_dim , num_heads , dim_feedforward ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim , num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim , num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim , dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward , embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self , tgt , memory , query_pos = None , pos = None ):

        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q , k , v= tgt)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        q = tgt if query_pos is None else tgt + query_pos
        k= memory if pos is None else (memory+pos)
        v= memory
        tgt2 = self.cross_attn(q , k , v)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt
        
        
class Decoder(nn.Module):
    def __init__(self , embed_dim , num_heads , dim_feedforward , num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim , num_heads , dim_feedforward) for _ in range(num_layers)])

    def forward(self , tgt , memory , query_pos = None , pos = None ):
        for layer in self.layers:
            tgt = layer(tgt , memory , query_pos , pos)
        return tgt