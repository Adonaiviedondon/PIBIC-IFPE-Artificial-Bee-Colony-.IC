from src.ABC.abc_otimizacao import AbcOtimizador
from src.PSO.pso_otimizacao import PsoOtimizador
from src.GWO.gwo_otimizacao import GwoOtimizador
from src.framework.agente_rl import AgenteRL
import numpy as np 

class AmbienteOtimizacao:
    def __init__(self,funcao,bounds,tamanho_populacao,tamanho_problema,num_iteracoes,verbose):
        self.funcao           = funcao
        self.bounds           = bounds
        self.num_iteracoes    = num_iteracoes
        self.verbose          = verbose
        self.tamanho_populacao = tamanho_populacao
        self.tamanho_problema = tamanho_problema
        self.melhor_global    = float('inf')
        self.melhor_solucao   = None

        self.algoritmos = {0:AbcOtimizador,1:PsoOtimizador,2:GwoOtimizador}
        self.nomes = {0:"ABC",1:"PSO",2:"GWO"}
        self.agente = AgenteRL(alpha = 0.15,gamma = 0.85,epsilon = 0.35)

        self.historico = {
            "melhor_fitness":[],
            "algotritmo_selecionado":[],
            "recompensa":[],
        }
    def iniciarAlgoritmo(self):
        for algoritmo in  self.algoritmos.values():
            if isinstance(algoritmo,GwoOtimizador):
                algoritmo.iniciar(self.funcao,self.bounds,self.tamanho_problema,self.tamanho_populacao,self.num_iteracoes)
            else:
                algoritmo.iniciar(self.funcao,self.bounds,self.tamanho_problema,self.tamanho_populacao)

    def obterEstadoGlobal(self):
        estados=[algoritmo.obter_estado() for algoritmo in self.algoritmos.values]
        return{
            "melhor fitness":self.melhor_global,
            "diversidade":np.mean([e['diversidade'] for e in estados]),
            "taxa melhoria":np.mean([e['taxa_melhoria'] for e in estados])
        }

    def executarCiclo(self):
        self.iniciarAlgoritmo()

        for iteracao in range(self.num_iteracoes):
            estadoAnterior = self.obterEstadoGlobal()
            acao      = self.agente.selecionar_acao(estadoAnterior)
            algoritmo = self.algoritmos[acao]

            fitness_anterior = self.melhor_global
            fitness_posterior  = algoritmo.executar_ciclo()

            if fitness_posterior < self.melhor_global:
                self.melhor_global  = fitness_posterior
                self.melhor_solucao = algoritmo.obter_melhor_solucao()

            estado_depois = self.obterEstadoGlobal()
            recompensa    = self.agente.calcular_recompensa(fitness_anterior, self.melhor_global)
            self.agente.atualizar(estado_depois, acao, recompensa, estado_depois)
            self.agente.reduzir_epsilon()

            self.historico['melhor_fitness'].append(self.melhor_global)
            self.historico['algoritmo_usado'].append(self.nomes[acao])
            self.historico['recompensas'].append(recompensa)

        return self.melhor_solucao, self.melhor_global, self.historico
