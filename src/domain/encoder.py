import numpy as np
from domain.encoder_block import EncoderBlock

class TransformerEncoder:
    def __init__(self, num_layers: int, d_model: int, d_ff: int):
        if num_layers <= 0:
            raise ValueError("O número de camadas deve ser no mínimo 1.")
            
        # Instancia N blocos independentes (cada um terá seus próprios pesos inicializados)
        self.layers = [EncoderBlock(d_model, d_ff) for _ in range(num_layers)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        # O tensor X atravessa as camadas sequencialmente
        for layer in self.layers:
            x = layer.forward(x)
            
        return x