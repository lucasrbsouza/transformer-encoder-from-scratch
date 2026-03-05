from domain.vocabulary import Vocabulary
from infrastructure.embeddings import EmbeddingTable

def main():
    token_map = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
    vocab = Vocabulary(token_map)
    
    d_model = 64
    embeddings = EmbeddingTable(vocab_size=vocab.size, d_model=d_model)
    
    phrase = "o banco bloqueou cartao"
    token_ids = vocab.encode(phrase)
    
    X = embeddings.get_embeddings(token_ids)
    
    print(f"Frase original: '{phrase}'")
    print(f"IDs dos tokens: {token_ids}")
    print(f"Shape do tensor de entrada (X): {X.shape}")

if __name__ == "__main__":
    main()