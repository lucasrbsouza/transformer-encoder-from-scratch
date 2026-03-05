# Construindo o Transformer Encoder "From Scratch"

**Instituição:** iCEV - Instituto de Ensino Superior  
**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Aluno:** José Lucas Silva Souza

---

## 🎯 Objetivo do Laboratório

Este repositório contém a implementação da passagem direta (*Forward Pass*) de um bloco Encoder completo do modelo Transformer, baseado no artigo *"Attention Is All You Need"* (Vaswani et al., 2017).

O objetivo principal é demonstrar o funcionamento interno das:
- Multiplicações de matrizes em redes neurais
- Projeções (*Query*, *Key*, *Value*)
- Atenção (*Scaled Dot-Product Attention*)
- Normalizações (*Layer Normalization*)

Para focar no motor matemático, o projeto foi desenvolvido **estritamente com** `Python 3.11`, `NumPy` e `Pandas`, **sem** utilização de frameworks de Deep Learning como PyTorch, TensorFlow ou Keras.

---

## 🏗️ Arquitetura do Projeto

O código foi estruturado seguindo princípios de **Clean Architecture**, **SOLID** e **Domain-Driven Design (DDD)**, separando claramente as lógicas de domínio das operações de infraestrutura:

- **`src/domain/`** - Contém as regras de negócio e a matemática pura do modelo (Atenção, FFN, LayerNorm, etc.)
- **`src/infrastructure/`** - Lida com a simulação da tabela de Embeddings e inicialização de pesos
- **`src/main.py`** - Ponto de entrada que orquestra o fluxo de dados e valida os tensores

---

## 📋 Requisitos

- Python 3.8+
- NumPy
- Pandas

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/lucasrbsouza/transformer-encoder-from-scratch.git
cd transformer-encoder-from-scratch
```

### 2. Crie um ambiente virtual (Opcional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install numpy pandas
```

### 4. Execute o programa

```bash
python src/main.py
```

### ✅ Saída Esperada

Ao executar o código, o sistema processará a frase `"o banco bloqueou cartao"` através de N=6 camadas do Encoder. O console exibirá a validação de que as dimensões do tensor se mantêm consistentes `(Batch, Tokens, d_model)` do início ao fim do processamento:

```
Frase original: 'o banco bloqueou cartao'
Camadas processadas: 6
Shape de Entrada (X): (1, 4, 64)
Shape de Saída (Z): (1, 4, 64)

Validação de Sanidade Passou? Sim
```

---

## 📝 Nota de Integridade e Créditos

Em conformidade com o Contrato Pedagógico da disciplina, declaro que este projeto foi desenvolvido utilizando os conceitos lecionados em sala.

Ferramentas de Inteligência Artificial foram consultadas **exclusivamente** como assistentes para:
- Revisão de arquitetura de software
- Boas práticas de estruturação de código Python (*Clean Code*)
- Revisão do README
- Revisões de Texto

A **autoria e a implementação lógica/matemática originais** foram mantidas conforme exigido no roteiro do laboratório.

---

**Desenvolvido para iCEV - 2026.1**

**Author: lucasrbsouza**
