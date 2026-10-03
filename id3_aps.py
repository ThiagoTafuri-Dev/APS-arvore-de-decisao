"""ID3 em Python puro — versão em Português para a APS."""

import math
import unittest
from collections import Counter


# --- Teoria da Informação ---------------------------------------------------
# Os dados são uma lista de pares (x, y): x = dict de atributos, y = classe/rótulo.

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


# --- Estrutura da Árvore de Decisão -------------------------------------------

class No:
    def __init__(self, atributo=None, ramos=None, rotulo=None, padrao=None):
        self.atributo = atributo      # Atributo testado no nó (None caso seja folha)
        self.ramos = ramos or {}      # {valor_do_atributo: próximo_nó}
        self.rotulo = rotulo          # Classe final predita (apenas em nós folha)
        self.padrao = padrao          # Classe majoritária (para valores não vistos no treino)


class ArvoreDecisaoID3:
    def ajustar(self, X, y):
        """Treina a árvore de decisão ID3 (equivalente ao método fit)."""
        self.raiz = self._construir(list(zip(X, y)), list(X[0]))
        return self

    def _construir(self, dados, atributos):
        rotulos = [y for _, y in dados]
        maioria = Counter(rotulos).most_common(1)[0][0]

        # Condição de parada (Nó Folha): dados puros ou sem atributos restantes
        if len(set(rotulos)) == 1 or not atributos:
            return No(rotulo=maioria, padrao=maioria)

        # Escolhe o atributo de maior Ganho de Informação
        melhor_atributo = max(atributos, key=lambda a: ganho_informacao(dados, a))
        atributos_restantes = [a for a in atributos if a != melhor_atributo]
        
        no = No(atributo=melhor_atributo, padrao=maioria)
        for valor, subconjunto in agrupar_por(dados, melhor_atributo).items():
            no.ramos[valor] = self._construir(subconjunto, atributos_restantes)
            
        return no

    def predizer(self, X_teste):
        """Realiza predições para novas instâncias (equivalente ao método predict)."""
        predicoes = []
        for x in X_teste:
            no = self.raiz
            while no.ramos and x.get(no.atributo) in no.ramos:
                no = no.ramos[x[no.atributo]]
            # Se a folha tiver nó interno ou valor inédito, utiliza a classe padrão (majoritária)
            predicoes.append(no.rotulo if not no.ramos else no.padrao)
        return predicoes

    # Mantemos aliases fit e predict para compatibilidade com padrão scikit-learn
    fit = ajustar
    predict = predizer


def imprimir_arvore(no, recuo=""):
    """Exibe visualmente a árvore de decisão no terminal."""
    if not no.ramos:
        print(f"{recuo}→ Decisão: {no.rotulo}")
        return
    print(f"{recuo}[Atributo: {no.atributo}]")
    for valor, filho in no.ramos.items():
        print(f"{recuo}  {valor}:")
        imprimir_arvore(filho, recuo + "    ")


# --- Conjunto de Dados de Exemplo (Aula / Tema 4) -----------------------------

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


# --- Testes Unitários ---------------------------------------------------------

class TesteID3(unittest.TestCase):
    def test_entropia_inicial(self):
        self.assertAlmostEqual(entropia(y), 0.940, places=3)

    def test_raiz_eh_montante(self):
        self.assertAlmostEqual(ganho_informacao(list(zip(X, y)), "montante"), 0.247, delta=0.002)
        self.assertEqual(ArvoreDecisaoID3().ajustar(X, y).raiz.atributo, "montante")

    def test_acuracia_treinamento(self):
        modelo = ArvoreDecisaoID3().ajustar(X, y)
        predicoes = modelo.predizer(X)
        acuracia = sum(p == t for p, t in zip(predicoes, y)) / len(y)
        self.assertEqual(acuracia, 1.0)


if __name__ == "__main__":
    modelo = ArvoreDecisaoID3().ajustar(X, y)
    print(f"Entropia inicial H(S): {entropia(y):.4f}\n")
    print("=== ESTRUTURA DA ÁRVORE APRENDIDA ===")
    imprimir_arvore(modelo.raiz)
    print("\n=== EXECUTANDO TESTES UNITÁRIOS ===")
    unittest.main(argv=["ignorado"], exit=False, verbosity=2)