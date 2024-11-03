import torch
import torch.nn as nn

from transformer import Decoder

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, d_ff, num_layers=4, num_heads=2):
        self.decoder = Decoder(seq_lenth=seq_length, vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads)

    def forward(self, x, mask=None):
        out, attn_maps = self.decoder(x, mask)

        return out, attn_maps
    
def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100, device="cpu"):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity
