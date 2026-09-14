# src/ABC/abc_otimizacao.py
import numpy as np
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.framework.interface import AlgoritmoOtimizacao


class AbcOtimizacao(AlgoritmoOtimizacao):

    def iniciar(self, funcao, bounds, tamanho_populacao=50, tamanho_problema=12, **kwargs):
        self.funcao            = funcao
        self.bounds            = bounds
        self.tamanho_populacao = tamanho_populacao
        self.tamanho_problema  = tamanho_problema
        self.Num_Falhas        = 50
        self.melhor_global     = float('inf')
        self.historico_fitness = []                            

        v1, v2 = bounds
        self.populacao = np.random.uniform(
                            v1, v2, (tamanho_populacao, tamanho_problema))
        self.fitness   = np.array([self.funcao(s) for s in self.populacao])
        self.contadores = np.zeros(tamanho_populacao)

    def explorar(self, index):
        x, y = self.populacao.shape
        dimensao = random.randint(0, y - 1)

        
        solucao_vizinhas = [i for i in range(x) if i != index]
        solucao_vizinha_escolhida = random.choice(solucao_vizinhas)

        
        solucao_Atual = np.copy(self.populacao[index])

        pertubacaoPhi = (random.random() - 0.5) * 2

        
        solucao_Atual[dimensao] += pertubacaoPhi * (
            self.populacao[index][dimensao] -
            self.populacao[solucao_vizinha_escolhida][dimensao]
        )

        v1, v2 = self.bounds
        solucao_Atual = np.clip(solucao_Atual, v1, v2)

        fitness_solucao_atual = self.funcao(solucao_Atual)

        if fitness_solucao_atual < self.fitness[index]:
            self.populacao[index]  = solucao_Atual
            self.fitness[index]    = fitness_solucao_atual
            self.contadores[index] = 0
        else:
            self.contadores[index] += 1

    def passo(self):                                          
        return self.executar_ciclo()

    def executar_ciclo(self):
        n = self.tamanho_populacao

        # Fase operárias
        for i in range(n):
            self.explorar(i)

        # Fase observadoras
        valor = 1.0 / (1.0 + self.fitness)
        prob  = valor / np.sum(valor)
        for i in range(n):
            idx = int(np.random.choice(range(n), p=prob))
            self.explorar(idx)                              

        # Fase exploradoras
        v1, v2 = self.bounds
        melhor_atual = float('inf')                           
        for i in range(n):
            if self.contadores[i] > self.Num_Falhas:          
                self.populacao[i]  = np.random.uniform(v1, v2, self.tamanho_problema)
                self.fitness[i]    = self.funcao(self.populacao[i])
                self.contadores[i] = 0

        melhor_atual       = float(np.min(self.fitness))       
        self.melhor_global = min(self.melhor_global, melhor_atual)
        self.historico_fitness.append(self.melhor_global)

        return self.melhor_global

    def obter_melhor(self):                                   
        idx = int(np.argmin(self.fitness))
        return self.populacao[idx], self.melhor_global

    def obter_melhor_fitness(self):
        return self.melhor_global

    def obter_melhor_solucao(self):
        return self.populacao[int(np.argmin(self.fitness))]    

    def _taxa_melhoria(self):
        if len(self.historico_fitness) < 12:
            return 1.0
        ultimos = self.historico_fitness[-12:]
        return abs(ultimos[0] - ultimos[-1]) / (abs(ultimos[0]) + 1e-10)

    def obter_estado(self):
        return {
            'melhor_fitness': self.melhor_global,
            'media_fitness' : float(np.mean(self.fitness)),
            'desvio_padrao' : float(np.std(self.fitness)),
            'diversidade'   : float(np.mean(              
                                np.std(self.populacao, axis=0))),
            'taxa_melhoria' : self._taxa_melhoria(),    
        }

    def aplicar_acao(self, acao: dict):                    
        """Agente RL pode ajustar o limite de abandono."""
        if 'Num_Falhas' in acao:
            self.Num_Falhas = int(acao['Num_Falhas'])