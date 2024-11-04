import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // num_heads


    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # batch_size, num_heads, seq_lenth, head_dim
        query = query.view(batch_size, seq_length_query, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        k_t = key.transpose(-1, -2)    # batch_size, num_heads, head_dim, seq_lenth
        scaled_scores = torch.matmul(query, k_t) / math.sqrt(self.head_dim)     # batch_size, num_heads, seq_lenth, seq_lenth

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scaled_scores, dim=-1)


        attention = torch.matmul(attention_weights, value)

        attention = attention.transpose(1, 2).contiguous()
        concat = attention.view(batch_size, seq_length, self.d_model) 

        out = self.out(concat)

        return out, attention_weights
        