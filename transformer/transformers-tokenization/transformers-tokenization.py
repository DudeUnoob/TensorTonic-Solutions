import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Collect all words (lowercased) from list of strings
        words = []
        for t in texts:
            words.extend(t.lower().split())
    
        # Special tokens with fixed IDs
        specials = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
    
        # Reset vocab dicts
        self.word_to_id = {}
        self.id_to_word = {}
    
        # Add special tokens
        for tok, idx in specials.items():
            self.word_to_id[tok] = idx
            self.id_to_word[idx] = tok
    
        # Add unique non-special words in sorted order
        unique_words = sorted(set(words))
        next_id = len(specials)
    
        for w in unique_words:
            if w in self.word_to_id:
                continue
            self.word_to_id[w] = next_id
            self.id_to_word[next_id] = w
            next_id += 1
    
        # Set vocab size
        self.vocab_size = next_id
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        text = text.lower()
        tokens = text.split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(tok, unk_id) for tok in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        unk = self.unk_token
        return " ".join(self.id_to_word.get(i, unk) for i in ids)