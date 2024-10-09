# models.py

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings, WordEmbeddings
from torch.utils.data import Dataset

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embed_file):
        self.examples = read_sentiment_examples(infile)
        self.word_embeddings = read_word_embeddings(embed_file)

        self.sentences = []
        # append averaged embedding
        for ex in self.examples:
            sentence = [self.word_embeddings.get_embedding(w) for w in ex.words]
            self.sentences.append(sum(sentence) / len(sentence))

        self.labels = [ex.label for ex in self.examples]

        self.embeddings = torch.tensor(np.array(self.sentences), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    