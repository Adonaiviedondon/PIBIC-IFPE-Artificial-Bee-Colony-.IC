import numpy as np 
import sys
import random
from pathlib import path

sys.path.insert(0, str(path(__file__).parent.parent.parent))
from src.framework.interface import AlgoritmoOtimizacao

class AbcOtimizacao(AlgoritmoOtimizacao):
    def iniciar(self,bounds,funcao,tamanho_populacao = 50,tamanho_problema = 12):
        self.bounds = bounds
        self.funcao = funcao
        self.tamanho_populacao = tamanho_populacao
        self.tamanho_problema = tamanho_problema
        self.Num_Falhas = 50
        self.melhor_global = float('inf')
        v1 , v2 = bounds

        self.populacao = np.random.uniform(v1,v2,(tamanho_populacao,tamanho_problema))

        fitness_lista = []
        for s in self.populacao:
            valor = self.funcao(s)
            fitness_lista.append(valor)
            
        self.fitness = np.array(fitness_lista)
        self.contadores = np.zeros(tamanho_populacao)

    def explorar(self,index):
        x,y = self.populacao.shape

        dimensao = random.randint(0,y - 1)

        solucao_vizinhas = []
        for i in range(n):
            if i != index:
                solucao_vizinhas()
        solucao_vizinha_escolhida = random.choice(solucao_vizinhas)

        solucao_Atual = np.copy(self.populacao(self))

        pertubacaoPhi = (random.random() - 0.5) * 2

        solucao_Atual += pertubacaoPhi * (self.populacao[index][dimensao] -self.populacao[solucao_vizinha_escolhida][dimensao])

        v1, v2 = self.bounds
        solucao_Atual = np.clip(solucao_Atual,v1,v2)

        fitness_solucao_atual = self.funcao(solucao_Atual)

        if fitness_solucao_atual < self.fitness[index]:
            self.populacao[index] = solucao_Atual
            self.fitness[index] = fitness_solucao_atual
            self.contadores[index]=0
        else:
            self.contadores[index] += 1

          

