import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_embeddings()
        

    def _init_embeddings(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.2)


    def forward(self, x):
        out = self.token_embedding(x)
        out = self.dropout(out)

        return out


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding. The positional embedding are added to the token embeddings before being fed into the encoder or decoder
    """

    def __init__(self, max_seq_len, d_model, dropout=0.1):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_embeddings()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        

    def _init_embeddings(self):
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.2)


    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = token_embeddings.size()

        # Make sure sequence length doesn't exceed the maximum length
        if seq_length > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_length} exceeds maximum length {self.max_seq_len}")
        
        positions = torch.arange(seq_length, device=token_embeddings.device)
        pos_embeddings = self.pos_embedding(positions)
        out = token_embeddings + pos_embeddings.unsqueeze(0)
        out = self.dropout(out)

        return out
    