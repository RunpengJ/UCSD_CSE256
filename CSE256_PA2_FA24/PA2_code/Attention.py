import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=2):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = self.d_model / num_heads

        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.out = nn.Linear(self.d_model, self.d_model)

        self._initialization()

    def _initialization(self):
        None


    def forward(self, query, key, value, mask=None):
        batch_size, seq_length = key.size(0), key.size(1)

        seq_length_query = query.size(1)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length_query, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = q.transpose(1,2)    # batch_size x num_heads x seq_lenth x head_dim
        k = k.transpose(1,2) 
        v = v.transpose(1,2)    

        k = k.transpose(-1, -2)    # batch_size x num_heads x head_dim x seq_lenth
        product = q @ k

        if mask is not None:
            product = product.mask_filled(mask == 0, float("-inf"))

        product = product / math.sqrt(self.head_dim)

        scores = F.softmax(product, dim=-1)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        out = self.out(concat)

        return out
        