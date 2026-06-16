import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        maxLen = max((len(seq) for seq in seqs), default=0)
    else:
        maxLen = max_len

    for idx in range(len(seqs)):
        seqs[idx] = seqs[idx][:maxLen] + [pad_value] * max(0, maxLen - len(seqs[idx]))

    return np.array(seqs)