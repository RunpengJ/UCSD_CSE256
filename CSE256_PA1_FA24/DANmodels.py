# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from utils import *

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embed):
        self.examples = read_sentiment_examples(infile)
        self.word_embed = word_embed
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)
        self.indices = [torch.tensor([self.word_embed.word_indexer.index_of(w) if self.word_embed.word_indexer.index_of(w) != -1 
                                      else self.word_embed.word_indexer.index_of("UNK")
                                      for w in ex.words], dtype=torch.long) for ex in self.examples]
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]
    
    def collate_fn(self, batch):
        indices, labels = zip(*batch)
        indices_padded = pad_sequence(indices, batch_first=True, padding_value=0)  # PAD index assumed to be 0
        labels = torch.tensor(labels, dtype=torch.long)
        return indices_padded, labels
        

class DAN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size=None, word_embed=None, dropout=0.2, frozen=True):
        super().__init__()
        # Using pretrained initialization
        
        if word_embed:
            self.embeddings = word_embed.get_initialized_embedding_layer(frozen=frozen)
        else:
            # Initialization without pretrained
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
            init.kaiming_uniform_(self.embeddings.weight)

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output size for binary classification
        self.dropout = nn.Dropout(dropout) # dropout
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x): 
        x = self.embeddings(x)
        x = x.mean(dim=1)  # Average embeddings to get a fixed-size vector
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Remove ReLU here before log softmax
        x = self.log_softmax(x)
        return x
    