import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ABC.controlador import AbcControlador
from src.ABC.FuncoesParaSolucao import (
    Esfera,
    Rastringin,
    BananaRosenBrock,
    DimensaoVetorAckley,
    OtimizacaoGlobalGriewank,
    OtimizaçaoZakharov
)
from src.utils.analisador import Analisador
from src.utils.plot import plot
from src.utils.ajudantes import (
    DataDadosArquivos,
    LoggingExperimentos,
    BarraProgresso,
    printHeader,
    printSecao,
)

FUNCOES = {
     'Esfera':      Esfera,
    'Rastrigin':   Rastringin,
    'Rosenbrock':  BananaRosenBrock,
    'Ackley':      DimensaoVetorAckley,
    'Griewank':    OtimizacaoGlobalGriewank,
    'Zakharov':    OtimizaçaoZakharov,
}


NUM_EXECUCOES    = 30
NUM_ITERACOES    = 1000
TAMANHO_PROBLEMA = 10
NUM_OPERARIAS    = 45
NUM_OBSERVADORAS = 45
NUM_FALHAS       = 50
BOUNDS           = (-100, 100)
PASTA_RESULTADOS = 'data/results'

def testeSimples():
    printHeader("ABC — Teste Simples (Esfera)")

    abc = AbcControlador(
        eficiencia_fn = Esfera,
        FunçoesParaSolução= None,
        num_operarias    = 45,
        num_observadoras = 45,
        tamanhoProblema     = 10,
        Num_Interacoes   = 1000,
        Num_Falhas      = 20,
        bounds           = (-100, 100),
        verbose          = True,
    )

    melhor_solucao, melhor_fitness, historico = abc.run()
    print(f"\nMelhor fitness que foi encontrado:{melhor_fitness:.6f}")

    plotter = plot(diretorioGraficos=PASTA_RESULTADOS)
    plotter.PlotConvergenciaAlgoritmo(
         historia= historico,
         title= "convergencia da Esfera",
         filename="teste da esfera",
         show=False

    )

    


def rodarExperimentos():
    printHeader("Resultado dos Experimentos")

    # NUM_EXECUCOES    = 32
    # NUM_ITERACOES    = 110
    # TAMANHO_PROBLEMA = 12
    # NUM_FALHAS       = 40

    logger   = LoggingExperimentos(verbose=True)
    plotter = plot(diretorioGraficos=PASTA_RESULTADOS)

    historicos_por_funcao = {}
    melhores_por_funcao ={}

    for nome, fn in FUNCOES.items():
        printSecao(f"Função: {nome}")

        melhores   = []
        historicos = []
        barra      = BarraProgresso(NUM_EXECUCOES, nome)
              

        
        for execucao in range(NUM_EXECUCOES):
            abc = AbcControlador(
                eficiencia_fn = fn,
                FunçoesParaSolução = fn,
                num_operarias    = NUM_OPERARIAS,
                num_observadoras = NUM_OBSERVADORAS,
                tamanhoProblema    = TAMANHO_PROBLEMA,
                Num_Interacoes   = NUM_ITERACOES,
                Num_Falhas       = NUM_FALHAS,
                bounds           = BOUNDS,
                verbose          = False
            )

            melhor_solucao, melhor_fitness, historico = abc.run()
            melhores.append(melhor_fitness)
            historicos.append(historico)
            barra.atualizar(execucao + 1)
            logger.info(f"{nome} | execução {execucao+1}/{NUM_EXECUCOES}" f"| fitness: {melhor_fitness:.6f}")

        
        analisador = Analisador(nome, melhores, historicos)
        analisador.salvar(PASTA_RESULTADOS)
        plotter.PlotConvergenciaMedia(
                historicos = historicos,
                nome_funcao = nome,
                filename = f'convergencia_{nome}.png',
                show = False
            )
        historicos_por_funcao[nome] = historicos[-1]
        melhores_por_funcao[nome]   = melhores

        printSecao(f"Resultados — {nome}")
        analisador.imprimir_resumo()

        plotter.PlotConvergenciaMultipla(
        historicos = historicos_por_funcao,
        filename   = 'comparativo_convergencias.png',
        show       = False
    )
    plotter.PlotBoxplot(
        resultados = melhores_por_funcao,
        filename   = 'boxplot_resultados.png',
        show       = False
    )

    logger.tempo_passado()
    logger.sucesso("Experimentos concluídos. Resultados em data/results/")

if __name__ == '__main__':
    
    testeSimples()
    rodarExperimentos()