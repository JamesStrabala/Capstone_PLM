import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from setup import load_fasta_sequences, build_vocab
from dataset import ProteinDataset
from model import ProteinLanguageModel

def main():
    # Config
    batch_size = 32
    epochs = 1
    max_len = 512
    embed_dim = 64
    lr = 1e-3

    # Load data
    print("Loading up")
    sequences = load_fasta_sequences("Sequences")
    vocab = build_vocab()

    dataset = ProteinDataset(sequences, vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ProteinLanguageModel(vocab_size=len(vocab)+1, embed_dim=embed_dim, max_len=max_len)
    model.load_state_dict(torch.load("output/model_epoch_1.pt", map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Training loop
    print ("Begining training loop")
    for epoch in range(epochs):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))  # flatten
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # Save model checkpoint
        torch.save(model.state_dict(), f"output/model_epoch_{epoch+2}.pt")
        print('epoch ' + str(epoch+1)+" done")

if __name__ == "__main__":
    main()
