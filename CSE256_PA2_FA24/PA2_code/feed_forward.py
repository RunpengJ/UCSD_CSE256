import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model 

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)

        self.dropout = nn.Dropout(p=dropout)

        self._initialization()

    def _initialization(self):

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        

    def forward(self, x):
        
        out = F.gelu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        # out = self.dropout(out)

        return out
