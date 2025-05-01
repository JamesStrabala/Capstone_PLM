import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, sequences, vocab, max_len=512):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for seq in sequences:
            tokens = [vocab[aa] for aa in seq if aa in vocab]
            if len(tokens) >= 2:  # need at least 2 tokens for input -> target
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        input_seq += [0] * (self.max_len - len(input_seq))
        target_seq += [0] * (self.max_len - len(target_seq))

        return torch.tensor(input_seq), torch.tensor(target_seq)