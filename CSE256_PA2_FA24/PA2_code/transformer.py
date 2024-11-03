# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from embedding import WordEmbedding, PositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, d_out=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):  # Added mask parameter
        # Attention
        attentions, attention_weights = self.self_attention(query, key, value, mask)
        attentions = self.dropout1(attentions)  # Apply dropout before residual
        attentions_res1 = attentions + query
        attentions_norm1 = self.norm1(attentions_res1)

        # Feed forward
        attentions_ff = self.ff(attentions_norm1)
        attentions_ff = self.dropout2(attentions_ff)  # Apply dropout before residual
        attentions_res2 = attentions_ff + attentions_norm1
        attentions_norm2 = self.norm2(attentions_res2)

        return attentions_norm2, attention_weights


class Encoder(nn.Module):
    def __init__(self, seq_lenth, vocab_size, d_model, d_ff, num_layers=4, num_heads=2):
        super().__init__()

        self.input_embedding = WordEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(seq_lenth, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)])


    def forward(self, x):
        att_maps = []
        out = self.input_embedding(x)
        out = self.pos_embedding(out)
        for layer in self.layers:
            out, att_map = layer(out,out,out)
            att_maps.append(att_map)

        return out, att_maps
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, d_out=d_model, activate_fn="relu")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):  # Added mask parameter
        # Attention
        self_att, self_att_weights = self.self_attention(query, key, value, mask)
        self_att = self.dropout1(self_att)  # Apply dropout before residual
        self_att = self.norm1(self_att + query)

        # Feed forward
        ff_out = self.ff(self_att)
        ff_out = self.dropout2(ff_out)  # Apply dropout before residual
        ff_out = self.norm2(ff_out + self_att)

        return ff_out, self_att_weights


class Decoder(nn.Module):
    def __init__(self, seq_lenth, vocab_size, d_model, d_ff, num_layers=4, num_heads=2):
        super().__init__()

        self.output_embedding = WordEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(seq_lenth, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_out, mask=None):

        if mask is None:
            seq_length = x.size(1)
            # Create causal mask (lower triangular)
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask = mask.to(x.device)

        out = self.output_embedding(x)
        out = self.pos_embedding(out)

        attn_maps = []
        for layer in self.layers:
            out, attn_map = layer(out, encoder_out, self_attention_mask=None, cross_attention_mask=None)
            attn_maps.append(attn_map)

        out = self.output_projection(out)

        return out, attn_maps


