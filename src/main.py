from domain.vocabulary import Vocabulary
from infrastructure.embeddings import EmbeddingTable
from domain.attention import ScaledDotProductAttention
from domain.layer_norm import LayerNorm
from domain.feed_forward import FeedForwardNetwork

def main():
    token_map = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
    vocab = Vocabulary(token_map)
    
    d_model = 64
    d_ff = d_model * 4
    
    embeddings = EmbeddingTable(vocab_size=vocab.size, d_model=d_model)
    attention = ScaledDotProductAttention(d_model=d_model)
    layer_norm1 = LayerNorm()
    
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
    layer_norm2 = LayerNorm()
    
    phrase = "o banco bloqueou cartao"
    token_ids = vocab.encode(phrase)
    
    X = embeddings.get_embeddings(token_ids)
    
    X_att = attention.forward(X)
    X_norm1 = layer_norm1.forward(X + X_att)
    
    X_ffn = ffn.forward(X_norm1)
    X_out = layer_norm2.forward(X_norm1 + X_ffn)
    
    print(f"Frase original: '{phrase}'")
    print(f"Shape de Entrada (X): {X.shape}")
    print(f"Shape após Attention e Norm 1 (X_norm1): {X_norm1.shape}")
    print(f"Shape após FFN e Norm 2 (X_out): {X_out.shape}")

if __name__ == "__main__":
    main()