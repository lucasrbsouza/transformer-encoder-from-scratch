import numpy as np
from domain.attention import ScaledDotProductAttention
from domain.layer_norm import LayerNorm
from domain.feed_forward import FeedForwardNetwork

class EncoderBlock:
    def __init__(self, d_model: int, d_ff: int):
        self.attention = ScaledDotProductAttention(d_model)
        self.norm1 = LayerNorm()
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNorm()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Etapa 1: Self-Attention e Conexão Residual + LayerNorm
        x_att = self.attention.forward(x)
        x_norm1 = self.norm1.forward(x + x_att)
        
        # Etapa 2: Feed-Forward Network e Conexão Residual + LayerNorm
        x_ffn = self.ffn.forward(x_norm1)
        x_out = self.norm2.forward(x_norm1 + x_ffn)
        
        return x_out