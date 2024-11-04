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
    

# Part 3 starts here
class MultiHeadAttentionALiBi(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.n_head = num_heads
        self.head_size = d_model // num_heads

        # Projections for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # ALiBi slopes
        m = torch.tensor([2 ** (-(8 / num_heads) * i) for i in range(num_heads)])
        self.register_buffer("slopes", m.view(1, num_heads, 1, 1))
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Get Q, K, V
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add ALiBi bias
        position_ids = torch.arange(T, device=x.device)
        distance = position_ids.view(1, 1, 1, T) - position_ids.view(1, 1, T, 1)
        alibi_bias = self.slopes * distance
        scores = scores + alibi_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), attn
    
class MultiHeadLocalAttention(nn.Module):
    def __init__(self, n_embed, n_head, window_size=16):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.window_size = window_size
        
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Get Q, K, V
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Compute attention scores with local window
        scale = 1.0 / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Create local attention mask
        local_mask = torch.ones(T, T, device=x.device).triu(-self.window_size).tril(0)
        scores = scores.masked_fill(local_mask == 0, float('-inf'))
        
        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), attn