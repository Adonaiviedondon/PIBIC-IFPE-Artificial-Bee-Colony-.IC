
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.framework.interface import AlgoritmoOtimizacao


class GwoOtimizacao(AlgoritmoOtimizacao):

    def iniciar(self, funcao, bounds, tamanho_populacao=50,tamanho_problema=12, Num_Interacoes=1000, **kwargs):
        self.funcao            = funcao
        self.bounds            = bounds
        self.tamanho_populacao = tamanho_populacao
        self.tamanho_problema  = tamanho_problema
        self.Num_Interacoes    = Num_Interacoes
        self.iteracao_atual    = 0
        self.melhor_global     = float('inf')
        self.historico_fitness = []

        v1, v2 = bounds
        self.populacao = np.random.uniform(
                            v1, v2, (tamanho_populacao, tamanho_problema))
        self.fitness   = np.array([self.funcao(i) for i in self.populacao])
        self._atualizar_lideres()

    def _atualizar_lideres(self):
        ind = np.argsort(self.fitness)
        self.alpha_pos     = np.copy(self.populacao[ind[0]])
        self.alpha_fitness = float(self.fitness[ind[0]])
        self.beta_pos      = np.copy(self.populacao[ind[1]])
        self.delta_pos     = np.copy(self.populacao[ind[2]])

    def executar_ciclo(self):         
        v1, v2 = self.bounds
        a = 2 - 2 * (self.iteracao_atual / self.Num_Interacoes)

        for i in range(self.tamanho_populacao):
            posicao_nova = np.zeros(self.tamanho_problema)

            for lider in [self.alpha_pos, self.beta_pos, self.delta_pos]:
                r1 = np.random.rand(self.tamanho_problema)  
                r2 = np.random.rand(self.tamanho_problema)  

                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * lider - self.populacao[i])
                posicao_nova += lider - A * D

            posicao_nova = np.clip(posicao_nova / 3.0, v1, v2)
            novo_fitness = self.funcao(posicao_nova)

            if novo_fitness < self.fitness[i]:
                self.populacao[i] = posicao_nova
                self.fitness[i]   = novo_fitness

        self._atualizar_lideres()
        self.iteracao_atual += 1

        self.melhor_global = min(self.melhor_global, self.alpha_fitness)
        self.historico_fitness.append(self.melhor_global)

        return self.melhor_global

    def obter_melhor_fitness(self):
        return self.melhor_global

    def obter_melhor_solucao(self):  
        return self.alpha_pos

    def _taxa_melhoria(self):
        if len(self.historico_fitness) < 12:
            return 1.0
        ultimos = self.historico_fitness[-12:]
        return abs(ultimos[0] - ultimos[-1]) / (abs(ultimos[0]) + 1e-10)

    def obter_estado(self):
        return {
            'melhor_fitness': self.melhor_global,              
            'desvio_padrao' : float(np.std(self.fitness)),
            'diversidade'   : float(np.mean(                  
                                np.std(self.populacao, axis=0))),
            'media_fitness' : float(np.mean(self.fitness)),
            'taxa_melhoria' : self._taxa_melhoria(),           
        }