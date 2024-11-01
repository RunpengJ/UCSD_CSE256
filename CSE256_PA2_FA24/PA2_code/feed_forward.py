import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, d_out, activate_fn="gelu", dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_out)

        self.dropout = nn.Dropout(p=dropout)
        self.activate = nn.GELU() if activate_fn == "gelu" else nn.ReLU
        self._initialization(activate_fn)

        self.d_model = d_model
        self.d_ff = d_ff 
        self.d_out = d_out

    def _initialization(self, activate_fn):
        if activate_fn == "gelu":
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            nn.init.kaiming_normal_(self.linear1.weight)
            nn.init.kaiming_normal_(self.linear2.weight)        

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        

    def forward(self, x):
        
        out = self.activate(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)

        return out
