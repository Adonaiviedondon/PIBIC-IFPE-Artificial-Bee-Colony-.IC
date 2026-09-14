from src.ABC.abc_otimizacao import AbcOtimizacao
from src.PSO.pso_otimizacao import PsoOtimizacao
from src.GWO.gwo_otimizacao import GwoOtimizacao
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

        self.abc = AbcOtimizacao()
        self.pso = PsoOtimizacao()
        self.gwo = GwoOtimizacao()

        self.algoritmos = {0:AbcOtimizacao,1:PsoOtimizacao,2:GwoOtimizacao}
        self.nomes = {0:"ABC",1:"PSO",2:"GWO"}
        self.agente = AgenteRL(alpha = 0.15,gamma = 0.85,epsilon = 0.35)

        self.historico = {
            'melhor_fitness' : [],
            'algoritmo_usado': [],   
            'recompensas'    : [],   
        }
    def iniciarAlgoritmo(self):
        for algoritmo in self.algoritmos.values():
            if isinstance(algoritmo, GwoOtimizacao):
                algoritmo.iniciar(
                self.funcao ,
                self.bounds,
                tamanho_populacao = self.tamanho_populacao,  # ← argumentos nomeados
                tamanho_problema  = self.tamanho_problema,   # ← evita confusão de ordem
                Num_Interacoes    = self.num_iteracoes
            )
            else:
                algoritmo.iniciar(
                self.funcao,
                self.bounds,
                tamanho_populacao = self.tamanho_populacao,  # ← argumentos nomeados
                tamanho_problema  = self.tamanho_problema
            )

    def obterEstadoGlobal(self):
        estados = [alg.obter_estado() for alg in self.algoritmos.values()]
        return {
            'melhor_fitness': self.melhor_global,
            'diversidade'   : float(np.mean([
                                e.get('diversidade', 0) for e in estados])),
            'taxa_melhoria' : float(np.mean([
                                e.get('taxa_melhoria', 1.0) for e in estados])),
        }

    def executar(self):
        
        self._iniciar_algoritmos()
 
        for iteracao in range(self.num_iteracoes):
            estado_antes  = self.obter_estado_global()
            acao          = self.agente.selecionar_acao(estado_antes)
            algoritmo     = self.algoritmos[acao]
 
            fitness_antes = self.melhor_global
 
            
            fitness_atual = algoritmo.executar_ciclo()
 
            if fitness_atual < self.melhor_global:
                self.melhor_global  = fitness_atual
                self.melhor_solucao = algoritmo.obter_melhor_solucao()
 
            estado_depois = self.obter_estado_global()
            recompensa    = self.agente.calcular_recompensa(
                                fitness_antes, self.melhor_global)
 
            
            self.agente.atualizar(estado_antes, acao, recompensa, estado_depois)
            self.agente.reduzir_epsilon()
 
            self.historico['melhor_fitness'].append(self.melhor_global)
            self.historico['algoritmo_usado'].append(self.nomes[acao])
            self.historico['recompensas'].append(recompensa)
 
            if self.verbose:
                print(f"  Iter {iteracao+1:>4}/{self.num_iteracoes} | "
                      f"Alg: {self.nomes[acao]:<4} | "
                      f"Melhor: {self.melhor_global:.6f}")
 
        return self.melhor_solucao, self.melhor_global, self.historico

    def imprimir_uso_algoritmos(self):
        usado = self.historico['algoritmo_usado']
        total = len(usado)
        if total == 0:
            return
        print("\n  Uso dos algoritmos pelo agente RL:")
        for nome in ['ABC', 'PSO', 'GWO']:
            count = usado.count(nome)
            pct   = count / total * 100
            barra = '█' * int(pct / 2.5) + '░' * (40 - int(pct / 2.5))
            print(f"  {nome}: |{barra}| {pct:.1f}% ({count} iterações)")
 
