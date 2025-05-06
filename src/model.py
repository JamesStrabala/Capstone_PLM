import torch.nn as nn

class ProteinLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to (batch_size, seq_len, embed_dim)
        logits = self.fc_out(x)
        return logits
