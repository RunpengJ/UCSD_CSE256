# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn

from attention import MultiHeadAttention, MultiHeadAttentionALiBi, MultiHeadLocalAttention
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


    def forward(self, x, mask=None):
        # Pre-norm architecture
        x_norm = self.norm1(x)
        self_att, self_att_weights = self.self_attention(x_norm, x_norm, x_norm, mask)
        self_att = self_att + x  # Residual connection with original input
        
        # Pre-norm for feed forward
        ff_norm = self.norm2(self_att)
        ff_out = self.ff(ff_norm)
        ff_out = ff_out + self_att  # Residual connection
        
        return ff_out, self_att_weights


class Decoder(nn.Module):
    def __init__(self, seq_lenth, vocab_size, d_model, d_ff, num_layers=4, num_heads=2):
        super().__init__()

        self.token_embedding = WordEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(seq_lenth, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, num_heads) for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)


    def forward(self, x, mask=None):
        if mask is None:
            seq_length = x.size(1)
            mask = ~torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask = mask.to(x.device)

        out = self.token_embedding(x)
        out = self.pos_embedding(out)

        attn_maps = []
        for layer in self.layers:
            out, attn_map = layer(out, mask=mask)
            attn_maps.append(attn_map)

        out = self.ln_f(out)
        logits = self.lm_head(out)

        return logits, attn_maps

""" Part 3 starts here """
class DecoderBlockPart3(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = MultiHeadLocalAttention(n_embed, n_head)
        self.ff = FeedForward(n_embed, 100, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights
    

class DecoderPart3(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        
        # Token embeddings only (no positional embeddings needed with ALiBi)
        self.token_embedding = WordEmbedding(vocab_size=vocab_size, d_model=n_embed)
        
        # Decoder blocks with ALiBi attention
        self.blocks = nn.ModuleList([
            DecoderBlockPart3(n_embed, n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        if mask is None:
            seq_length = x.size(1)
            mask = ~torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask = mask.to(x.device)

        out = self.token_embedding(x)

        attn_maps = []
        for layer in self.blocks:
            out, attn_map = layer(out, mask=mask)
            attn_maps.append(attn_map)

        out = self.ln_f(out)
        logits = self.lm_head(out)

        return logits, attn_maps