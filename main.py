import sys
import json
import numpy as np
from pathlib import Path

# garante que a raiz do projeto está no PATH do Python
sys.path.insert(0, str(Path(__file__).parent))

# funções benchmark — as mesmas que já existiam no projeto
from src.ABC.FuncoesParaSolucao import (
    Esfera,
    Rastringin,
    BananaRosenBrock,
    DimensaoVetorAckley,
    OtimizacaoGlobalGriewank,
    OtimizacaoZakharov,
)

# framework híbrido — novo
from src.framework.ambiente import AmbienteSimulacao

# utilitários — os mesmos que já existiam
from src.utils.ajudantes import (
    printHeader,
    printSecao,
    LoggingExperimentos,
    BarraProgresso,
)

# ── configurações do experimento ─────────────────────────────────────────────
FUNCOES = {
    'Esfera'     : Esfera,
    'Rastrigin'  : Rastringin,
    'Rosenbrock' : BananaRosenBrock,
    'Ackley'     : DimensaoVetorAckley,
    'Griewank'   : OtimizacaoGlobalGriewank,
    'Zakharov'   : OtimizacaoZakharov,
}

CONFIG = {
    'tamanho_populacao' : 30,
    'tamanho_problema'  : 10,
    'num_iteracoes'     : 500,
    'num_execucoes'     : 10,
    'bounds'            : (-100, 100),
    'pasta_resultados'  : 'data/results',
}


def testeSimples():
    printHeader("TESTE SIMPLES — Esfera (1 execução, framework híbrido)")

    ambiente = AmbienteSimulacao(
        funcao            = Esfera,
        bounds            = CONFIG['bounds'],
        tamanho_populacao = CONFIG['tamanho_populacao'],
        tamanho_problema  = CONFIG['tamanho_problema'],
        num_iteracoes     = 100,
        verbose           = True,
    )

    melhor_solucao, melhor_fitness, historico = ambiente.executar()

    print(f"\n  Melhor fitness encontrado: {melhor_fitness:.6f}")
    ambiente.imprimir_uso_algoritmos()
    ambiente.agente.imprimir_qtable()


def rodarExperimentos():
    printHeader("FRAMEWORK HÍBRIDO ABC + PSO + GWO — Experimentos Completos")

    logger = LoggingExperimentos(verbose=True)
    pasta  = Path(CONFIG['pasta_resultados'])
    pasta.mkdir(parents=True, exist_ok=True)

    todos_resultados = {}

    for nome, fn in FUNCOES.items():
        printSecao(f"Função: {nome}")

        melhores       = []
        uso_algoritmos = {'ABC': 0, 'PSO': 0, 'GWO': 0}
        barra          = BarraProgresso(CONFIG['num_execucoes'], nome)

        for execucao in range(CONFIG['num_execucoes']):

            ambiente = AmbienteSimulacao(
                funcao            = fn,
                bounds            = CONFIG['bounds'],
                tamanho_populacao = CONFIG['tamanho_populacao'],
                tamanho_problema  = CONFIG['tamanho_problema'],
                num_iteracoes     = CONFIG['num_iteracoes'],
                verbose           = False,
            )

            melhor_solucao, melhor_fitness, historico = ambiente.executar()
            melhores.append(melhor_fitness)

            for alg, count in _contar_uso(historico['algoritmo_usado']).items():
                uso_algoritmos[alg] += count

            barra.atualizar()
            logger.info(
                f"{nome} | execução {execucao+1}/{CONFIG['num_execucoes']} "
                f"| fitness: {melhor_fitness:.6f}"
            )

        resultado = {
            'funcao'           : nome,
            'melhor'           : float(np.min(melhores)),
            'pior'             : float(np.max(melhores)),
            'media'            : float(np.mean(melhores)),
            'desvio_padrao'    : float(np.std(melhores)),
            'mediana'          : float(np.median(melhores)),
            'melhores_por_exec': melhores,
            'uso_algoritmos'   : uso_algoritmos,
        }

        caminho = pasta / f'hibrido_{nome}.json'
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

        todos_resultados[nome] = resultado

        printSecao(f"Resultados — {nome}")
        print(f"  Melhor   : {resultado['melhor']:.6f}")
        print(f"  Pior     : {resultado['pior']:.6f}")
        print(f"  Média    : {resultado['media']:.6f}")
        print(f"  Desvio   : {resultado['desvio_padrao']:.6f}")
        print(f"  Uso ABC  : {uso_algoritmos['ABC']}")
        print(f"  Uso PSO  : {uso_algoritmos['PSO']}")
        print(f"  Uso GWO  : {uso_algoritmos['GWO']}")

    caminho_geral = pasta / 'resumo_hibrido.json'
    with open(caminho_geral, 'w', encoding='utf-8') as f:
        json.dump(todos_resultados, f, indent=2, ensure_ascii=False)

    logger.tempo_passado()
    logger.sucesso(f"Resultados salvos em {CONFIG['pasta_resultados']}/")


def _contar_uso(lista_algoritmos):
    return {
        'ABC': lista_algoritmos.count('ABC'),
        'PSO': lista_algoritmos.count('PSO'),
        'GWO': lista_algoritmos.count('GWO'),
    }


if __name__ == '__main__':
    testeSimples()
    rodarExperimentos()