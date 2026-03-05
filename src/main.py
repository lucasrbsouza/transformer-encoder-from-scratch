from domain.vocabulary import Vocabulary
from infrastructure.embeddings import EmbeddingTable
from domain.attention import ScaledDotProductAttention
from domain.layer_norm import LayerNorm

def main():
    token_map = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
    vocab = Vocabulary(token_map)
    
    d_model = 64
    embeddings = EmbeddingTable(vocab_size=vocab.size, d_model=d_model)
    attention = ScaledDotProductAttention(d_model=d_model)
    layer_norm1 = LayerNorm()
    
    phrase = "o banco bloqueou cartao"
    token_ids = vocab.encode(phrase)
    
    X = embeddings.get_embeddings(token_ids)
    
    X_att = attention.forward(X)
    
    X_res = X + X_att
    X_norm1 = layer_norm1.forward(X_res)
    
    print(f"Frase original: '{phrase}'")
    print(f"Shape de Entrada (X): {X.shape}")
    print(f"Shape da Atenção (X_att): {X_att.shape}")
    print(f"Shape após Add & Norm (X_norm1): {X_norm1.shape}")

if __name__ == "__main__":
    main()