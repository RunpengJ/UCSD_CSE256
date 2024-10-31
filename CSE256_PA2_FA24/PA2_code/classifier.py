import torch
import torch.nn as nn

class SpeechClassifier(nn.Module):
    def __init__(self, d_model, d_out) -> None:
        super().__init__()

        self.linear = nn.Linear(d_model. d_out)
        self.softmax = nn.Softmax(d_out)

        self.d_model = d_model
        self.d_out = d_out

    def forward(self, x):

        out = self.linear(x)

        out = self.softmax(out)
        
        return out