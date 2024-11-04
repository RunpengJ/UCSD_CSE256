import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, x, targets=None):
        logits, attn_maps = self.decoder(x)

        if targets is None:
            return logits, attn_maps

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return loss

    
def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100, device="cpu"):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
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

def experiment_LM(model, train_loader, test_loader, device, max_iters, eval_interval, eval_iters, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    iter_count = 0
    train_losses = []
    test_perplexities = []

    for iter_num, (xb, yb) in enumerate(train_loader):
        if iter_num > max_iters:
            break

        xb, yb = xb.to(device), yb.to(device)
        
        loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluation
        if iter_num % eval_interval == 0:
            train_perplexity = compute_perplexity(model, train_loader, eval_iters=eval_iters, device=device)
            perplexity = compute_perplexity(model, test_loader, eval_iters=eval_iters, device=device)
            test_perplexities.append(perplexity)
            print(f"Iteration {iter_num}: Train Loss = {loss.item():.4f}, Train Perplexity = {train_perplexity:.2f}, Test Perplexity = {perplexity:.2f}")

        iter_count += 1

    return train_losses, test_perplexities

    