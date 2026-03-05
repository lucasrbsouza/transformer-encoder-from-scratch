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
│   ├── main.py                 # Ponto de entrada principal
│   ├── domain/
│   │   ├── __init__.py
│   │   └── vocabulary.py       # Gerenciamento de vocabulário
│   └── infrastructure/
│       ├── __init__.py
│       └── embeddings.py       # Implementação de embeddings
```

## 📚 Conceitos Implementados

- **Multi-Head Attention**: Mecanismo de atenção com múltiplas cabeças
- **Feed Forward Network**: Rede completamente conectada com ativação ReLU
- **Layer Normalization**: Normalização de camadas para estabilidade
- **Positional Encoding**: Codificação posicional das sequências

## 📝 Nota de Integridade e Créditos

Desenvolvido para a disciplina de **Tópicos em Inteligência Artificial**.

Ferramentas de IA foram consultadas exclusivamente para:
- Revisão de arquitetura de software
- Boas práticas de código Python
- Revisão de README

A lógica matemática e a implementação são autorais.

**Autor**: Lucas Souza
