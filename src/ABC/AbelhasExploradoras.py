import numpy as np

class AbelhasExploradoras:
    def __init__(self,EscolhasSolucao,valorEficiencia,contadorInteacoesSemMelhoria,eficiencia_fn,LimiteInteacoesSemMelhoria,bounds = (-100,100)):
        self.EscolhasSolucao = EscolhasSolucao
        self.valorEficiencia = valorEficiencia
        self.contadorInteacoesSemMelhoria = contadorInteacoesSemMelhoria
        self.eficiencia_fn = eficiencia_fn
        self.LimiteInteacoesSemMelhoria = LimiteInteacoesSemMelhoria
        self.bounds =bounds
    def atualiza(self , index):
       tamanhoProblema =  self.EscolhasSolucao.shape[1]

       valor1, valor2 = self.bounds

       SolucaoAtual = np.random.uniform(valor1,valor2,tamanhoProblema)
       self.EscolhasSolucao[index] = SolucaoAtual
       self.valorEficiencia[index] = self.eficiencia_fn(SolucaoAtual)
       self.contadorInteacoesSemMelhoria[index] = 0