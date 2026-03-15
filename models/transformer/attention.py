import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    This module is responsible for the 'Self-Attention' mechanism. 
    In DETR, this allows object queries to 'talk' to each other and to the image features.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads==0

        self.query = nn.Linear(d_model ,d_model )
        self.key = nn.Linear(d_model ,d_model)
        self.values = nn.Linear(d_model , d_model)

        self.num_heads = num_heads
        self.d_model = d_model

        self.output = nn.Linear(d_model , d_model)

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
        x = x.contiguous().view(batch_size, -1, self.d_model)
        return self.output(x)

