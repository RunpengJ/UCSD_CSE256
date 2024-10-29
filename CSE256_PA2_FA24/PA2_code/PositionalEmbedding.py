import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding. The positional embedding are added to the token embeddings before being fed into the encoder or decoder
    """

    def __init__(self, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.embed_dim = embed_dim
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

        return self.dropout(out)



        

    