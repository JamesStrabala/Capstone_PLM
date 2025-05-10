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
    return generated_seq
#
if __name__ == "__main__":
    model_path = "output/model_epoch_10.pt"  # 
    for i in range(10):
        new_seq = generate_sequence(model_path, start_seq="M", max_length=512, temperature=1.0)
        print(f"Generated unprompted sequence {i+1}: \n{new_seq}")
    print("Generating insulin variant")
    new_seq = generate_sequence(model_path, start_seq="MALWMRLLPLLALLALWGPD", max_length=512, temperature=1.0)
    print(f"Generated insulin variant: \n{new_seq}")
    print("Generating holin variant")
    new_seq = generate_sequence(model_path, start_seq="WLGVAERALKTAAQTALASI", max_length=512, temperature=1.0)
    print(f"Generated holin variant: \n{new_seq}")
    print("Generating channel variant")
    new_seq = generate_sequence(model_path, start_seq="MEQTEKSKVYAENGLLEKIK", max_length=512, temperature=1.0)
    print(f"Generated channel variant: \n{new_seq}")
    print("Generating cytochrome variant")
    new_seq = generate_sequence(model_path, start_seq="MEQTEVLKPRTLADLIRILH", max_length=512, temperature=1.0)
    print(f"Generated cytochrome variant: \n{new_seq}")
