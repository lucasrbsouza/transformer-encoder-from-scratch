import numpy as np
from typing import List

class EmbeddingTable:
    def __init__(self, vocab_size: int, d_model: int):
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("Tamanho do vocabulário e d_model devem ser maiores que zero.")
        
        self.d_model = d_model
        np.random.seed(42) 
        self._weights = np.random.randn(vocab_size, d_model)

    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        if not token_ids:
            raise ValueError("A lista de IDs de tokens não pode estar vazia.")
        
        sequence_embeddings = self._weights[token_ids]
        
        return np.expand_dims(sequence_embeddings, axis=0)