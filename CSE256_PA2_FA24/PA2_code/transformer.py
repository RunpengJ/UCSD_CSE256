# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn

from attneiton import MultiHeadAttention
from feed_forward import FeedForward
from embedding import InputEmbedding, PositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, num_heads=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads

        self.multi_head_attention = MultiHeadAttention(self.d_model, self.num_heads)
        self.ff = FeedForward(self.d_model, d_ff=self.d_ff)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attentions = self.multi_head_attention(query, key, value)
        attentions_res1 = attentions + query
        attentions_norm1 = self.norm1(attentions_res1)
        attentions_drop1 = self.dropout1(attentions_norm1)

        attentions_ff = self.ff(attentions_drop1)
        attentions_res2 = attentions_drop1 + attentions_ff
        attentions_norm2 = self.norm2(attentions_res2)
        attentions_drop2 = self.dropout2(attentions_norm2)

        return attentions_drop2


class TransformerEncoder(nn.Module):
    def __init__(self, seq_lenth, vocab_size, d_model, num_encoders=2, d_ff=None, num_heads=2):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(seq_lenth, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads) for i in range(num_encoders)])

    def forward(self, x):
        out = self.input_embedding(x)
        out = self.pos_embedding(out)
        for layer in self.layers:
            out = layer(out,out,out)

        return out