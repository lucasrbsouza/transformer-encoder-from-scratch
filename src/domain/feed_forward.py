import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int):
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("As dimensões devem ser estritamente positivas.")
            
        np.random.seed(42)
        self.w1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        
        self.w2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden_layer = np.maximum(0, x @ self.w1 + self.b1)
        
        return hidden_layer @ self.w2 + self.b2