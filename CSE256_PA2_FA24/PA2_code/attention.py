import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=2):
        super().__init__()

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        self._initialization()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // num_heads


    def _initialization(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(batch_size, seq_length_query, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = query.transpose(1,2)    # batch_size x num_heads x seq_lenth x head_dim
        k = key.transpose(1,2) 
        v = value.transpose(1,2)    

        k_t = k.transpose(-1, -2)    # batch_size x num_heads x head_dim x seq_lenth
        scaled_scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scaled_scores, dim=-1)


        attention = torch.matmul(attention_weights, v)

        attention = attention.transpose(1, 2).contiguous()
        concat = attention.view(batch_size, seq_length, self.d_model)   # batch_size x 

        out = self.out(concat)

        return out, attention_weights
        