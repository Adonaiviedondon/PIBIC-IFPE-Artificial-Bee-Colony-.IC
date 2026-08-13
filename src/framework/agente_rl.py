import numpy as np 
import json 
from pathlib import Path

class AgenteRL:
    AlgoritmoAcao ={0:"Abc" ,1:"PSO",2:"GWO"}

    def __init__(self,alpha = 0.15,gamma = 0.85,epsilon = 0.35):
        self.alpha = alpha
        self.alpha = gamma
        self.alpha = epsilon

        self.tabela_Q= np.zeros(6,3)

    def discretizar_estado(self,estado):
        taxa = estado.get('taxa_melhoria',0)
        diversidade = estado.get('diversidade',0)

        if taxa < 0.015:
            taxa_index = 0
        elif taxa < 0.15:
            taxa_index = 1
        else:
            taxa_index = 2

        if diversidade < 10:
            diversidade_index = 0
        else:
            diversidade_index = 1

        return taxa_index * 2 + diversidade_index

    def selecionar_acao(self,estado):
        index_estado = self.discretizar_estado(estado)

        if np.random.rand() < self.epsilon:
            acao = np.random.rand(0,3) 
        else:
            acao = int(np.argmax(self.tabela_Q[index_estado]))

    def diminuir_epsilon(self,fator =0.99,minimo=0.05):
        self.epsilon = max(minimo,self.epsilon*fator)

    def calcular_recompensa(self,fitness_anterior,fitness_posterior):
        melhoria = fitness_anterior - fitness_posterior
        if melhoria > 0:
            return float(np.log1p(melhoria))
        else:
            return -0.02
    
    def atualizar(self, estado_anterior, acao, recompensa, estado_atual:
        s  = self.discretizar_estado(estado_anterior)
        s2 = self.discretizar_estado(estado_atual)

        q_atual = self.tabela_Q[s,acao]
        q_futuro = np.max(self.tabela_Q[s2])

        self.tabela_Q[s,acao] += self.alpha * (recompensa + self.gamma * q_futuro - q_atual)



