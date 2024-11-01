import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, d_out, activate_fn="relu", dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_out)

        self.dropout = nn.Dropout(p=dropout)

        if activate_fn == "relu":
            self.activate = F.relu()
            self._initialization("kaiming")
        else:
            self.activate = F.gelu()
            self._initialization()

        self.d_model = d_model
        self.d_ff = d_ff 
        self.d_out = d_out

    def _initialization(self, initialize="xavier"):
        if initialize == "kaiming":
            nn.init.kaiming_normal_(self.linear1.weight)
            nn.init.kaiming_normal_(self.linear2.weight)
        else:
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        

    def forward(self, x):
        
        out = self.activate(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)

        return out
