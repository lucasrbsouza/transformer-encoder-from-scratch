import numpy as np

class LayerNorm:
    def __init__(self, epsilon: float = 1e-6):
        if epsilon <= 0:
            raise ValueError("Epsilon deve ser um valor positivo estritamente maior que zero.")
        
        self.epsilon = epsilon

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        return (x - mean) / np.sqrt(variance + self.epsilon)