import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
 
class MultiHeadAttention(nn.Module):
 
    def __init__(self, query_dim, key_dim, num_units, num_heads):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
 
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
 
    def forward(self, query, key, mask=None):
        querys = self.W_query(query) 
        keys = self.W_key(key)  
        values = self.W_value(key)
 
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  

        scores = torch.matmul(querys, keys.transpose(2, 3)) 
        scores = scores / (self.key_dim ** 0.5)
 
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
 
        return out,scores