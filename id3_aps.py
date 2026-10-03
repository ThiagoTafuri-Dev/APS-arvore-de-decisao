"""
APS: Inteligência Artificial
Thiago Tafuri Santos
Matrícula: 2024101346

"""

import math
import unittest
from collections import Counter


# Teoria da Informação 

def entropia(rotulos):
    """Calcula H(S) = -sum(p * log2(p)), assumindo 0 * log2(0) = 0."""
    n = len(rotulos)
    probs = [c / n for c in Counter(rotulos).values()]
    return -sum(p * math.log2(p) if p > 0 else 0.0 for p in probs)


def agrupar_por(dados, atributo):
    """Separa os pares (x, y) pelo valor do atributo especificado: {valor: [pares]}."""
    grupos = {}
    for x, y in dados:
        grupos.setdefault(x[atributo], []).append((x, y))
    return grupos


def ganho_informacao(dados, atributo):
    """Calcula Ganho(S, A) = H(S) - sum(|Sv|/|S| * H(Sv))."""
    n = len(dados)
    resto = sum(len(g) / n * entropia([y for _, y in g])
                for g in agrupar_por(dados, atributo).values())
    return entropia([y for _, y in dados]) - resto


# Estrutura da Árvore de Decisão 

class Node:
    def __init__(self, attribute=None, branches=None, label=None, default=None):
        self.attribute = attribute    # atributo testado neste nó (None se for folha)
        self.branches = branches or {}  # {valor_do_atributo: próximo nó}
        self.label = label            # classe final (apenas se for folha)
        self.default = default        # classe majoritária (usada p/ valores inéditos)


class DecisionTreeID3:
    def fit(self, X, y):
        """Treina a árvore ID3 a partir de X (lista de dicts) e y (lista de rótulos)."""
        self.root = self._construir(list(zip(X, y)), list(X[0]))
        return self

    def _construir(self, dados, atributos):
        rotulos = [y for _, y in dados]
        maioria = Counter(rotulos).most_common(1)[0][0]

        if len(set(rotulos)) == 1 or not atributos:
            return Node(label=maioria, default=maioria)

        melhor_atributo = max(atributos, key=lambda a: ganho_informacao(dados, a))
        atributos_restantes = [a for a in atributos if a != melhor_atributo]
        
        no = Node(attribute=melhor_atributo, default=maioria)
        for valor, subconjunto in agrupar_por(dados, melhor_atributo).items():
            no.branches[valor] = self._construir(subconjunto, atributos_restantes)
            
        return no

    def predict(self, X_test):
        """Retorna a classe prevista para cada instância de X_test."""
        predicoes = []
        for x in X_test:
            no = self.root
            while no.branches and x.get(no.attribute) in no.branches:
                no = no.branches[x[no.attribute]]
            # Se parou num nó interno, o valor era inédito: usa a classe majoritária
            predicoes.append(no.label if not no.branches else no.default)
        return predicoes


def imprimir_arvore(no, recuo=""):
    """Exibe visualmente a árvore de decisão no terminal."""
    if not no.branches:
        print(f"{recuo}→ Decisão: {no.label}")
        return
    print(f"{recuo}[Atributo: {no.attribute}]")
    for valor, filho in no.branches.items():
        print(f"{recuo}  {valor}:")
        imprimir_arvore(filho, recuo + "    ")


# Conjunto de Dados de Exemplo

DATASET = [
    {"montante": "médio", "idade": "sênior", "salário": "baixo", "conta": "sim", "empréstimo": "não"},
    {"montante": "médio", "idade": "sênior", "salário": "baixo", "conta": "não", "empréstimo": "não"},
    {"montante": "baixo", "idade": "sênior", "salário": "baixo", "conta": "sim", "empréstimo": "sim"},
    {"montante": "alto",  "idade": "média",  "salário": "baixo", "conta": "sim", "empréstimo": "sim"},
    {"montante": "alto",  "idade": "jovem",  "salário": "alto",  "conta": "sim", "empréstimo": "sim"},
    {"montante": "alto",  "idade": "jovem",  "salário": "alto",  "conta": "não", "empréstimo": "não"},
    {"montante": "baixo", "idade": "jovem",  "salário": "alto",  "conta": "não", "empréstimo": "sim"},
    {"montante": "médio", "idade": "média",  "salário": "baixo", "conta": "sim", "empréstimo": "não"},
    {"montante": "médio", "idade": "jovem",  "salário": "alto",  "conta": "sim", "empréstimo": "sim"},
    {"montante": "alto",  "idade": "média",  "salário": "alto",  "conta": "sim", "empréstimo": "sim"},
    {"montante": "médio", "idade": "média",  "salário": "alto",  "conta": "não", "empréstimo": "sim"},
    {"montante": "baixo", "idade": "jovem",  "salário": "baixo", "conta": "não", "empréstimo": "sim"},
    {"montante": "baixo", "idade": "sênior", "salário": "alto",  "conta": "sim", "empréstimo": "sim"},
    {"montante": "alto",  "idade": "média",  "salário": "baixo", "conta": "não", "empréstimo": "não"},
]

X = [{k: v for k, v in linha.items() if k != "empréstimo"} for linha in DATASET]
y = [linha["empréstimo"] for linha in DATASET]


# Casos de Demonstração (nenhum aparece no conjunto de treino)
# Cada item: (cliente novo, decisão esperada)

CASOS_DEMO = [
    ({"montante": "baixo",   "idade": "média",  "salário": "alto",  "conta": "não"}, "sim"),  # montante -> baixo
    ({"montante": "médio",   "idade": "jovem",  "salário": "baixo", "conta": "sim"}, "não"),  # montante -> médio -> salário
    ({"montante": "alto",    "idade": "sênior", "salário": "alto",  "conta": "não"}, "não"),  # montante -> alto -> conta
    ({"montante": "inédito", "idade": "jovem",  "salário": "alto",  "conta": "sim"}, "sim"),  # valor desconhecido -> maioria
]


def demonstrar_casos(modelo):
    """Classifica os clientes novos e mostra a decisão de cada um."""
    clientes = [cliente for cliente, _ in CASOS_DEMO]
    for i, (cliente, decisao) in enumerate(zip(clientes, modelo.predict(clientes)), start=1):
        print(f"Caso {i}: {cliente}")
        print(f"  → Empréstimo: {decisao}")


# Testes Unitários 

class TesteID3(unittest.TestCase):
    def test_entropia_inicial(self):
        self.assertAlmostEqual(entropia(y), 0.940, places=3)

    def test_raiz_eh_montante(self):
        self.assertAlmostEqual(ganho_informacao(list(zip(X, y)), "montante"), 0.247, delta=0.002)
        self.assertEqual(DecisionTreeID3().fit(X, y).root.attribute, "montante")

    def test_acuracia_treinamento(self):
        modelo = DecisionTreeID3().fit(X, y)
        predicoes = modelo.predict(X)
        acuracia = sum(p == t for p, t in zip(predicoes, y)) / len(y)
        self.assertEqual(acuracia, 1.0)

    def test_casos_demonstracao(self):
        modelo = DecisionTreeID3().fit(X, y)
        predicoes = modelo.predict([cliente for cliente, _ in CASOS_DEMO])
        self.assertEqual(predicoes, [esperado for _, esperado in CASOS_DEMO])


if __name__ == "__main__":
    modelo = DecisionTreeID3().fit(X, y)
    print(f"Entropia inicial H(S): {entropia(y):.4f}\n")
    print("=== ESTRUTURA DA ÁRVORE APRENDIDA ===")
    imprimir_arvore(modelo.root)
    print("\n=== DEMONSTRAÇÃO COM CLIENTES NOVOS ===")
    demonstrar_casos(modelo)
    print("\n=== EXECUTANDO TESTES UNITÁRIOS ===")
    unittest.main(argv=["ignorado"], exit=False, verbosity=2)