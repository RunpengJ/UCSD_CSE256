import torch
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import *
import re, collections
import time

class Byte_Pair_Encoding(Dataset):
    def __init__(self, infile, num_of_vocab):
        print("###### Loading dataset ... #####")
        self.examples = read_sentiment_examples(infile)
        self.indexer = Indexer()
        self.indexer.add_and_get_index('</w>')

        print("###### Building vocabulary ... ######")
        self.vocab = self.build_vocab()
        print("##### Computing merge operations #####")
        self.merge_ops = self.compute_merge_ops(num_of_vocab)
        
        print("###### Start encoding ... ###### ")
        self.encode() 

        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def build_vocab(self):
        vocab = collections.defaultdict(int)
        for ex in self.examples:
            for w in ex.words:
                word = ' '.join(list(w)) + " </w>"
                vocab[word] += 1
                for c in list(w):
                    self.indexer.add_and_get_index(c)
        return vocab
    
    def compute_merge_ops(self, num_of_vocab):
        num_of_merge = num_of_vocab - self.indexer.__len__()
        merge_ops = []

        for i in range(num_of_merge):
            pairs = self.get_stats(self.vocab)
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            merge_ops.append(best)
            self.indexer.add_and_get_index(''.join(best))
        return merge_ops

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = collections.defaultdict(int)
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    # given an example, return list of integers (the token)
    def encode(self):
        self.indices = []
        for ex in self.examples:
            tokens = [c for w in ex.words for c in list(w) + ["</w>"]]
            merged = self.merge_word(tokens)
            self.indices.append(torch.tensor([self.indexer.index_of(w) for w in merged]))

    def merge_word(self, tokens):
        sentence = ' '.join(tokens)
        for pair in self.merge_ops:
            if ' '.join(pair) in sentence:
                sentence = re.sub(re.escape(' '.join(pair)), ''.join(pair), sentence)
        return sentence.split()
                

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]
    
    def collate_fn(self, batch):
        indices, labels = zip(*batch)
        indices_padded = pad_sequence(indices, batch_first=True, padding_value=0)  # PAD index assumed to be 0
        labels = torch.tensor(labels, dtype=torch.long)
        return indices_padded, labels

