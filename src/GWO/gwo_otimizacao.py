import numpy as np
import sys
from pathlib import path

sys.path.insert(0, str(path(__file__).parent.parent.parent))
from src.framework.interface import AlgoritmoOtimizacao


class GwoOtimizacao(AlgoritmoOtimizacao):
    def iniciar(self,funcao,bounds,tamanho_populacao = 50,tamanho_problema = 12,Num_Interacoes   = 1000):
        self.funcao = funcao
        self.bounds = bounds
        self.tamanho_populacao = tamanho_populacao
        self.tamanho_problema = tamanho_problema
        self.Num_Interacoes = Num_Interacoes
        self.iteracao_atual = 0
        self.melhor_global = float('inf')
        self.historico_fitness = []
        v1 ,v2 = bounds
        self.populacao = np.uniform(v1,v2,tamanho_populacao,tamanho_problema)
        self.fitness = np.array([self.funcao(i) for i in self.populacao])




