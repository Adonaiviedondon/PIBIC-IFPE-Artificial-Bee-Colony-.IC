import numpy as np 
import random

class AbelhasOperarias:
    def __init__(self,funçoesSolucao,valorEficiencia,contadorInteacoesSemMelhoria,eficiencia_fn,bounds=(-100,100)):
        self.funçoesSolucao = funçoesSolucao  
        self.valorEficiencia = valorEficiencia
        self.contadorInteacoesSemMelhoria = contadorInteacoesSemMelhoria
        self.eficiencia_fn = eficiencia_fn
        self.bounds=bounds
    def explore(self, index: int):
        numeroSoluçoes,tamanhoProblema = self.funçoesSolucao.shape
        dimensao = random.randint(0,tamanhoProblema - 1)

        operarioAleatorio = random.choice([i for i in range(numeroSoluçoes) if i !=index])

        mutante = np.copy(self.funçoesSolucao[index])
        phi = (random.random() - 0.5) * 2
        mutante[dimensao] += phi * (
            self.funçoesSolucao[index][dimensao] -
            self.funçoesSolucao[operarioAleatorio][dimensao])

        valor1,valor2 = self.bounds
        mutante = np.clip(mutante,valor1,valor2)

        eficiencia_Do_Mutante = self.eficiencia_fn(mutante)

        if eficiencia_Do_Mutante < self.valorEficiencia[index]:
            self.funçoesSolucao[index]  = mutante
            self.valorEficiencia[index] = eficiencia_Do_Mutante 
            self.contadorInteacoesSemMelhoria[index] = 0
        else:
            self.contadorInteacoesSemMelhoria[index] += 1 