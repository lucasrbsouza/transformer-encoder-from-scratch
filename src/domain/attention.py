import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_model: int):
        if d_model <= 0:
            raise ValueError("d_model deve ser maior que zero.")
            
        self.d_model = d_model
        
        # Inicializando os pesos com valores pequenos para evitar saturação no softmax inicial
        np.random.seed(42)
        self.w_q = np.random.randn(d_model, d_model) * 0.01
        self.w_k = np.random.randn(d_model, d_model) * 0.01
        self.w_v = np.random.randn(d_model, d_model) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Subtrair o máximo estabiliza a exponencial numericamente (evita overflow)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # Transpor apenas as duas últimas dimensões (SeqLen, d_model) -> (d_model, SeqLen)
        k_t = np.swapaxes(k, -2, -1)
        
        scores = (q @ k_t) / np.sqrt(self.d_model)
        attention_weights = self._softmax(scores)
        
        return attention_weights @ v