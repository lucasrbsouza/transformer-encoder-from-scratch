from domain.vocabulary import Vocabulary
from infrastructure.embeddings import EmbeddingTable
from domain.encoder import TransformerEncoder

def main():
    token_map = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
    vocab = Vocabulary(token_map)
    
    # Parâmetros do modelo (usando d_model=64 para simulação em CPU)
    d_model = 64
    d_ff = d_model * 4
    num_layers = 6
    
    # Inicialização dos componentes
    embeddings = EmbeddingTable(vocab_size=vocab.size, d_model=d_model)
    encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, d_ff=d_ff)
    
    # Preparação da entrada
    phrase = "o banco bloqueou cartao"
    token_ids = vocab.encode(phrase)
    
    # Fluxo de Dados (Forward Pass)
    X = embeddings.get_embeddings(token_ids)
    Z = encoder.forward(X)
    
    # Validação de Sanidade exigida no roteiro
    print(f"Frase original: '{phrase}'")
    print(f"Camadas processadas: {num_layers}")
    print(f"Shape de Entrada (X): {X.shape}")
    print(f"Shape de Saída (Z): {Z.shape}")
    print(f"\nValidação de Sanidade Passou? {'Sim' if X.shape == Z.shape else 'Não'}")

if __name__ == "__main__":
    main()