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
        self.atualizar_lideres()

    def atualizar_lideres(self):
        ind_global = np.argsort(self.fitness)
        self.alpha_pos     = np.copy(self.populacao[ind_global[0]])
        self.alpha_fitness = float(self.fitness[ind_global[0]])

        self.beta_pos  = np.copy(self.populacao[ind_global[1]])
        self.delta_pos = np.copy(self.populacao[ind_global[2]])
    def executar_ciclo(self):
        v1 ,v2 = self.bounds 

        a =  2 - 2 *(self.iteracao_atual/self.Num_Interacoes)

        for i in range(self.tamanho_populacao):
            posicao_nova =  np.zeros(self.tamanho_problema)
            for lider in [self.alpha_pos,self.beta_pos,self.delta_pos]:
                r1 = np.random.randint(self.tamanho_problema)
                r2 = np.random.randint(self.tamanho_problema)

                A= a * 2  * r1 - a
                C = r2 * 2
                D = np.abs(lider * C - self.populacao[i])
                posicao_nova +=lider - D * A
            posicao_nova = np.clip(posicao_nova/3.0,v1,v2)
            selecao_gulosa = self.funcao(posicao_nova)
            if selecao_gulosa < self.fitness[i]:
                self.populacao[i]=posicao_nova
                self.fitness[i]=selecao_gulosa
        self.atualizar_lideres()
        self.iteracao_atual+=1

        self.melhor_global = min(self.melhor_global)
        self.melhor_global = min(self.melhor_global,self.alpha_fitness)
        self.historico_fitness.append(self.melhor_global)

        return self.melhor_global

    def obter_melhor_fitness(self):
        return self.melhor_global

    def obter_melhor_solucao(self):
        return self.alpha_pos


    def obter_taxa_melhoria(self):
        if len(self.historico_fitness)<12:
            return 1.0
        ultimos_valores = self.historico_fitness[-12:]
        return abs(ultimos_valores[0] - ultimos_valores[-1]) / (abs(ultimos_valores[0]) + 1e-10)


    def obter_estado(self):
        return {
            'melhor fitness':self.melhor_global,
            'desvio padrao':float(np.std(self.fitness)),
            'diversificacao':float(np.mean(np.std(self.populacao, axis=0))),
            'media fitness':float(np.mean(self.fitness)),
            'taxa de melhoria':self.obter_taxa_melhoria(),
            
        }
            
            

        
        







