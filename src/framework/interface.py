import numpy as np
from abc import ABC , abstractmethod

class AlgoritmoOtimizacao(ABC):
    @abstractmethod
    def iniciar(self,funcao,bounds,tamanho_populacao,tamanho_problema):
        pass


    @abstractmethod
    def executar_ciclo(self):
        pass

    @abstractmethod
    def obter_melhor_solucao(self):
        pass

    @abstractmethod
    def obter_estado(self):
        pass

    @abstractmethod
    def obter_melhor_fitness(self):
        pass



