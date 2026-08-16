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
        for i in range():
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

    def executar_ciclo(self):
            n = self.tamanho_populacao

            for i in range(n):
                self.explorar(i) 
            valor = 1.0 / (1.0 + self.fitness)
            prob  = valor / np.sum(valor)

            for i in range(n):
                idx = int(np.random.choice(range(n), p=prob))
                self._explorar(idx)
            v1, v2 = self.bounds
            for i in range(n):
                if self.contadores[i] > self.num_falhas:
                    self.populacao[i]  = np.random.uniform(v1, v2, self.tamanho_problema)
                    self.fitness[i]    = self.funcao(self.populacao[i])
                    self.contadores[i] = 0  

                    melhor_atual       = float(np.min(self.fitness))
            self.melhor_global = min(self.melhor_global, melhor_atual)
            self.historico_fitness.append(self.melhor_global)
 
            return self.melhor_global
        

    def obter_melhor_fitness(self):
        return self.melhor_global  

    def obter_melhor_solucao(self):
        self.populacao[int(np.argmin(self.fitness))]

    def obter_taxa_melhoria(self):
        if len(self.historico_fitness) < 12:
            return 1.0 
        ultimos = self.historico_fitness[-12:]
        return abs(ultimos[0] - ultimos[-1]) / (abs(ultimos[0]) + 1e-10)


    def obter_estado(self):
            return{
                'melhor_fitness' : self.melhor_global,
                'media_fitness'  : float(np.mean(self.fitness)),
                'desvio_padrao'  : float(np.std(self.fitness)),
                'diversidade': float(np.mean(np.std(self.posicoes, axis=0))),
                'taxa_melhoria' : self.taxa_melhoria(),
        }    

