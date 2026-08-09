import numpy as np
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent.parent))

def iniciar(self,funcao,bounds,tamanho_populacao = 50,tamanho_problema = 12):
    self.funcao = funcao
    self.bounds = bounds
    self.tamanho_populacao = tamanho_populacao
    self.tamanho_problema = tamanho_problema
    self.melhor_global = float('inf')
    self.historico.fitness = []

    self.w = 0.8#peso da inercia 
    self.c1 = 1.7#coeficiente cognitivo
    self.c2 = 1.7#coeficiente social

    v1,v2 = bounds
    velocidade_maxima = (v2 - v1) * 0.1 #largura maxima da busca

    self.posicoes = np.random.uniform(v1,v2,(tamanho_populacao,tamanho_problema))
    self.velocidades = np.random.uniform(-velocidade_maxima,velocidade_maxima,(tamanho_populacao,tamanho_problema))
    self.fitness = np.array([self.funcao(i) for i in self.posicoes])
    
    self.individual_best_position = np.copy(self.posicoes)
    self.individual_best_fitness = np.copy(self.fitness)

    index_global = int(np.argmin(self.fitness))#pontuação minima para melhor resultado possivel
    self.global_best_fitness = np.copy(self.fitness[index_global])#posiçao e valor exato da melhor particula
    self.global_best_position = np.copy(self.posicoes[index_global])#posiçao e valor exato da melhor particula

def execucao_ciclo(self):
    #extrainod os limites e numerom de particulas
    v1 ,v2 = self.bounds
    n = self.tamanho_populaçao
    r1 =np.random.rand(n,self.tamanho_problema)
    r2 =np.random.rand(n,self.tamanho_problema)

    self.velocidades = (self.w  * self.velocidades
            + self.c1 * r1 * (self.individual_best_position - self.posicoes)
            + self.c2 * r2 * (self.global_best_position  - self.posicoes))



