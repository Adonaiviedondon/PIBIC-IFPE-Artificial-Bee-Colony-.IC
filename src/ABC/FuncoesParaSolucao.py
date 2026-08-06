import numpy as np 

def Esfera(x):
    return np.sum(pow(x, 2))

def Rastringin(x):
    y = 10
    z = len(x)
    return y * z + np.sum(x**2 - y * np.cos(2 * np.pi * x))
def BananaRosenBrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def DimensaoVetorAckley(x):
    n = len(x)
    soma1=np.sum(x**2)
    soma2=np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(soma1 / n)) - np.exp(soma2 / n) + 20 + np.e

def OtimizacaoGlobalGriewank(x):
    parteSoma = np.sum(pow(x,2)) /4000
    parteProduto = np.prod(np.cos(x / np.sqrt(np.arange(1,len(x) + 1))))
    return parteSoma - parteProduto + 1

def OtimizaçaoZakharov(x):
    VetorIndice = np.arange(1,len(x)+1)
    soma1 = np.sum(pow(x,2))
    soma2 = np.sum(x * VetorIndice * 0.5)
    return soma1 + pow(soma2,2) + pow(soma2,4)



