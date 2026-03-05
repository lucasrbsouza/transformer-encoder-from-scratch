# Transformer Encoder "From Scratch"

Implementação educacional da passagem direta (Forward Pass) de um bloco Encoder do Transformer, baseado no artigo *"Attention Is All You Need"* (Vaswani et al., 2017).

## 📋 Descrição

Este projeto demonstra os mecanismos internos de um Encoder Transformer implementados inteiramente em **Python e NumPy**, sem dependências de frameworks de Deep Learning como PyTorch ou TensorFlow.

O foco é na compreensão prática de:
- Multiplicações de matrizes em redes neurais
- Projeções lineares (Query, Key, Value)
- Mecanismo de atenção (Attention)
- Normalização de camadas (Layer Normalization)

## 🔧 Requisitos

- Python 3.10+
- NumPy
- Pandas

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/lucasrbsouza/transformer-encoder-from-scratch.git
cd transformer-encoder-from-scratch
```

### 2. Instale as dependências

```bash
pip install numpy pandas
```

### 3. Execute o programa

```bash
python src/main.py
```

## 📁 Estrutura do Projeto

```
transformer-encoder-from-scratch/
├── README.md
├── src/
│   ├── main.py                          # Ponto de entrada principal
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── vocabulary.py                # Gerenciamento de vocabulário
│   │   ├── attention.py                 # Mecanismo de Atenção
│   │   └── layer_norm.py                # Normalização de Camadas
│   └── infrastructure/
│       ├── __init__.py
│       └── embeddings.py                # Implementação de Embeddings
```

## 📚 Conceitos Implementados

- **Multi-Head Attention**: Mecanismo de atenção com múltiplas cabeças
- **Feed Forward Network**: Rede completamente conectada com ativação ReLU
- **Layer Normalization**: Normalização de camadas para estabilidade
- **Positional Encoding**: Codificação posicional das sequências

## 🔧 Componentes Principais

### `ScaledDotProductAttention` (domain/attention.py)

Implementa o mecanismo de atenção em escala com produto ponderado:

```
Attention(Q, K, V) = softmax(QK^T / √d_model) * V
```

**Características:**
- Projeções lineares para Query, Key e Value
- Estabilização numérica do softmax (subtração de máximo)
- Inicialização de pesos com distribuição normal pequena

### `LayerNorm` (domain/layer_norm.py)

Normalização de camadas que padroniza as ativações:

```
LayerNorm(x) = (x - mean(x)) / √(variance(x) + epsilon)
```

**Características:**
- Normalização por features (aplicada em `axis=-1`)
- Parâmetro epsilon configurável para estabilidade numérica
- Validação de entrada para valores positivos

### `EmbeddingTable` (infrastructure/embeddings.py)

Gerencia as representações vetoriais dos tokens, transformando IDs em embeddings densos.

### `Vocabulary` (domain/vocabulary.py)

Gerencia o mapeamento entre tokens e índices, facilitando a codificação/decodificação de frases.

## 🔄 Fluxo de Execução

1. **Tokenização**: Frase é convertida em IDs usando `Vocabulary`
2. **Embedding**: IDs são transformados em vetores densos via `EmbeddingTable`
3. **Atenção**: Vetores passam pela `ScaledDotProductAttention`
4. **Residual + Normalização**: Resultado é somado à entrada original e normalizado com `LayerNorm`

```
Entrada Textual → Tokenização → Embeddings → Atenção → Add & Norm → Saída
```

## � Exemplo de Uso

```python
from domain.vocabulary import Vocabulary
from infrastructure.embeddings import EmbeddingTable
from domain.attention import ScaledDotProductAttention
from domain.layer_norm import LayerNorm

# Configurar vocabulário
token_map = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
vocab = Vocabulary(token_map)

# Instanciar componentes
d_model = 64
embeddings = EmbeddingTable(vocab_size=vocab.size, d_model=d_model)
attention = ScaledDotProductAttention(d_model=d_model)
layer_norm = LayerNorm()

# Processar frase
phrase = "o banco bloqueou cartao"
token_ids = vocab.encode(phrase)
X = embeddings.get_embeddings(token_ids)

# Forward pass
X_att = attention.forward(X)
X_residual = X + X_att
X_normalized = layer_norm.forward(X_residual)

print(f"Shape final: {X_normalized.shape}")  # (4, 64)
```

## �📝 Nota de Integridade e Créditos

Desenvolvido para a disciplina de **Tópicos em Inteligência Artificial**.

Ferramentas de IA foram consultadas exclusivamente para:
- Revisão de arquitetura de software
- Boas práticas de código Python
- Revisão de README

A lógica matemática e a implementação são autorais.

**Autor**: Lucas Souza
