# Árvore de Decisão — Algoritmo ID3

Este repositório contém a implementação em Python puro do algoritmo de aprendizado de máquina **ID3 (Iterative Dichotomiser 3)** para construção de árvores de decisão.

---

## 👤 Autor

- **Nome:** Thiago Tafuri Santos
- **Matrícula:** 2024101346
- **Disciplina:** APS — Inteligência Artificial

---

## 📌 Sobre o Projeto

O script `id3_aps.py` treina uma Árvore de Decisão a partir de um conjunto de dados categóricos de análise de crédito/empréstimo.

### Conceitos Utilizados:
- **Entropia $H(S)$**: Medida de incerteza ou desordem dos dados.
- **Ganho de Informação $Ganho(S, A)$**: Critério de seleção do melhor atributo a cada nó da árvore.
- **Construção Recursiva**: Divisão do dataset em subconjuntos até encontrar folhas puras ou esgotar atributos.
- **Predição e Casos de Teste**: Classificação de novos clientes (não presentes no conjunto de treino).

---

## 🚀 Como Executar

### Pré-requisitos
Apenas o **Python 3** instalado (o script utiliza apenas bibliotecas nativas como `math`, `collections` e `unittest`).

### Passo a Passo

1. **Clone ou Baixe o Repositório:**
   ```bash
   git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
   cd SEU-REPOSITORIO
   ```

2. **Execute o Script:**
   ```bash
   python id3_aps.py
   ```

---

## 📊 O que o Script Exibe?

Ao rodar o script, você verá no terminal:
1. O valor da **Entropia Inicial** $H(S)$ do dataset.
2. A **Estrutura Visual da Árvore de Decisão** aprendida (exibindo atributos, ramos e decisões).
3. A **Demonstração com Clientes Novos** (classificação de novas instâncias fora do treino).
4. A execução dos **Testes Unitários** validando os cálculos, acurácia e predições do modelo.
