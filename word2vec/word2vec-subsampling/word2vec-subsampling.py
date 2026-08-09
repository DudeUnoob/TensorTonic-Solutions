import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    N = sum(counts)

    frequencies = []
    keep_probability = []

    for i in counts:
        frequencies.append(i / N)

    print(frequencies)

    for i in frequencies:
        keep_probability.append(min(1, torch.sqrt(t / i)))

    return torch.tensor(keep_probability)
    pass
