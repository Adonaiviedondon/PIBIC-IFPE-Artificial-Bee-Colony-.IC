from src.ABC.abc_otimizacao import AbcOtimizador
from src.PSO.pso_otimizacao import PsoOtimizador
from src.GWO.gwo_otimizacao import GwoOtimizador
from src.framework.agente_rl import AgenteRL

class AmbienteOtimizacao:
    def __init__(self,funcao,bounds,tamanho_populacao,tamanho_problema,num_iteracoes,verbose):
        self.funcao           = funcao
        self.bounds           = bounds
        self.num_iteracoes    = num_iteracoes
        self.verbose          = verbose
        self.melhor_global    = float('inf')
        self.melhor_solucao   = None

        self.algoritmos = {0:AbcOtimizador,1:PsoOtimizador,2:GwoOtimizador}
        self.nomes = {0:"ABC",1:"PSO",2:"GWO"}
        