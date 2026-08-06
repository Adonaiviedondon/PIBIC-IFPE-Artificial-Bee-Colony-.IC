import numpy as np 
import random 

class abelhasObservadoras:
    def __init__(self,EscolhasSolucao,valorEficiencia,contadorInteacoesSemMelhoria,eficiencia_fn,bounds=(-100,100)):
        self.EscolhasSolucao = EscolhasSolucao
        self.valorEficiencia = valorEficiencia
        self.contadorInteacoesSemMelhoria = contadorInteacoesSemMelhoria
        self.eficiencia_fn = eficiencia_fn
        self.bounds=bounds

    def escolher_fontes(self, probabilidades):
        return int(np.random.choice(range(len(self.EscolhasSolucao)),p=probabilidades))
    def explore(self,index):
        numeroSoluçoes,tamanhoProblema = self.EscolhasSolucao.shape
        proporcao = random.randint(0,tamanhoProblema - 1)

        operarioAleatorio = random.choice([i for i in range(numeroSoluçoes) if i != index])

        mutante = np.copy(self.EscolhasSolucao[index])
        phi = (random.random() - 0.5) * 2
        mutante[proporcao] += phi *(self.EscolhasSolucao[index][proporcao] - self.EscolhasSolucao[operarioAleatorio][proporcao]) 

        valor1,valor2 = self.bounds
        mutante = np.clip(mutante,valor1,valor2)

        Eficiencia = self.eficiencia_fn(mutante)

        if Eficiencia < self.valorEficiencia[index]:
            self.EscolhasSolucao[index] = mutante
            self.valorEficiencia[index] = Eficiencia
            self.contadorInteacoesSemMelhoria[index] = 0 
        else:
            self.contadorInteacoesSemMelhoria[index] += 1 








    