import torch
import torch.nn.functional as F
from model import ProteinLanguageModel
from setup import build_vocab

def generate_sequence(model_path, start_seq="M", max_length=512, temperature=1.0):
    vocab = build_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}  # id -> amino acid

    model = ProteinLanguageModel(vocab_size=len(vocab)+1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model good and loaded")
    input_ids = [vocab.get(aa, 0) for aa in start_seq]
    generated = input_ids.copy()

    for i in range(max_length - len(start_seq)):
        input_tensor = torch.tensor(generated[-512:], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        next_token_logits = outputs[0, -1, :]  # (vocab_size,)

        # Apply temperature
        next_token_logits = next_token_logits / temperature
        probs = F.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == 0:  # Padding token, skip
            break

        generated.append(next_token_id)

    # Convert token ids back to amino acids
    generated_seq = ''.join([inv_vocab.get(idx, 'X') for idx in generated])
    print("Sequence converted")
    return generated_seq

if __name__ == "__main__":
    model_path = "output/model_epoch_10.pt"  
    inputSeq="M"
    amino_acids = "RHKDESTNQCUGPAVILMFYW"
    While inputSeq !='0':
        valid=True
        For t in inputSeq:
            if t not in amino_acids:
                valid=False
        if valid and len(inputSeq>0):
            print("Enter a prompt containing only Amino Acid Abbreviations (ACDEFGHIKLMNPQRSTVWY) or '0' to quit:")
            new_seq = generate_sequence(model_path, start_seq=inputSeq, max_length=512, temperature=1.0)
            print(f"Generated Unprompted sequence: \n{new_seq}")
            inputSeq =input("Enter a prompt containing only Amino Acid Abbreviations (ACDEFGHIKLMNPQRSTVWY) or '0' to quit:")
        else:
            inputSeq=input("Invalid character(s) detected. Please check your prompt and try again.")
    print("Quitting")
