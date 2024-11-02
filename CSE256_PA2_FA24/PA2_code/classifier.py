import torch
import torch.nn as nn
import torch.nn.functional as F
from feed_forward import FeedForward


class Classifier(nn.Module):
    def __init__(self, d_model, d_hidden, d_out):
        super().__init__()

        self.ff = FeedForward(d_model=d_model, d_ff=d_hidden, d_out=d_out, activate_fn="relu")


    def forward(self, x):
        out = torch.mean(x, dim=1)
        out = self.ff(out)

        return out
    

class SpeechClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier


    def forward(self, x):
        out, att_maps = self.encoder(x)
        out = self.classifier(out)

        return out, att_maps