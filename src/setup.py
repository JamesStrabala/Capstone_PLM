import os
def load_fasta_sequences(path):
    max_len=512
    sequences = []
    for filename in os.listdir(path):
        if filename.endswith(".fasta"):
            file_path = os.path.join(path, filename)
            with open(file_path, "r") as f:
                seq = ""
                for line in f:
                    if line.startswith(">"):
                        if seq and len(sec)<=max_len:
                            sequences.append(seq)
                            seq = ""
                    else:
                        seq += line.strip()
                if seq and len (sec)<=max_len:
                    sequences.append(seq)
    print(str(len(sequences))+" sequences loaded.")
    return sequences

def build_vocab():
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    vocab = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # 0 reserved for padding
    return vocab
#Sequences\Voltage 588K.fasta
