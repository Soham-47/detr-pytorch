import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    This module is responsible for the 'Self-Attention' mechanism. 
    In DETR, this allows object queries to 'talk' to each other and to the image features.
    """
    def __init__(self, embed_dim, num_heads):
        if embed_dim<=0 or num_heads<=0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads!=0:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__()

        self.query = nn.Linear(embed_dim ,embed_dim )
        self.key = nn.Linear(embed_dim ,embed_dim)
        self.values = nn.Linear(embed_dim , embed_dim)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.output = nn.Linear(embed_dim , embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.query(q)
        K = self.key(k)
        V = self.values(v)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        Q = Q.permute(0 , 2 , 1 , 3)

        K = K.view (batch_size, -1 , self.num_heads, self.head_dim)
        K = K.permute(0 , 2 , 1 , 3)

        V = V.view (batch_size , -1 , self.num_heads , self.head_dim)
        V = V.permute ( 0 , 2 , 1 ,3)

        scores = torch.matmul(Q , K.transpose(-2, -1))
        scores = scores /math.sqrt(self.head_dim)
        
        weights = torch.softmax (scores , dim=-1)    

        weighted_sum = torch.matmul(weights, V)
        
        x = weighted_sum.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, self.embed_dim)
        return self.output(x)

