# utils/analisador.py
import numpy as np
import json
import os

class Analisador:

    def __init__(self, nome_funcao, melhores_por_execucao, historicos):
        
        self.nome_funcao           = nome_funcao
        self.melhores_por_execucao = np.array(melhores_por_execucao)
        self.historicos            = historicos


    def melhor_global(self):
        return float(np.min(self.melhores_por_execucao))

    def pior_global(self):
        return float(np.max(self.melhores_por_execucao))

    def media_global(self):
        return float(np.mean(self.melhores_por_execucao))

    def desvio_padrao(self):
        return float(np.std(self.melhores_por_execucao))

    def mediana(self):
        """Mediana dos melhores fitness entre todas as execuções."""
        return float(np.median(self.melhores_por_execucao))

    

    def taxa_convergencia(self):
        curva_media = self._curva_media_convergencia()
        diferencas  = np.diff(curva_media)
        return float(np.mean(diferencas))

    def iteracao_convergencia(self, tolerancia=1e-6):
        curva_media = self._curva_media_convergencia()
        diferencas  = np.abs(np.diff(curva_media))

        for i, d in enumerate(diferencas):
            if d < tolerancia:
                return i + 1

        return len(curva_media)   # não convergiu dentro do limite

    def _curva_media_convergencia(self):
        num_iteracoes = min(len(h['best_fitness']) for h in self.historicos)
        return np.mean(
        [[h['best_fitness'][i] for i in range(num_iteracoes)]
         for h in self.historicos],
        axis=0
        )

    
    def indice(self):
        return {
            'funcao'            : self.nome_funcao,
            'melhor'            : self.melhor_global(),
            'pior'              : self.pior_global(),
            'media'             : self.media_global(),
            'mediana'           : self.mediana(),
            'desvio_padrao'     : self.desvio_padrao(),
            'taxa_convergencia' : self.taxa_convergencia(),
            'iter_convergencia' : self.iteracao_convergencia(),
            'num_execucoes'     : len(self.melhores_por_execucao),
            'melhores_por_exec' : self.melhores_por_execucao.tolist()
        }

    

    def salvar(self, pasta='data/results'):
        os.makedirs(pasta, exist_ok=True)
        caminho = os.path.join(pasta, f'resultados_{self.nome_funcao}.json')

        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(self.indice(), f, indent=4, ensure_ascii=False)

        print(f"  Resultados salvos em: {caminho}")


    def imprimir_resumo(self):
        separador = "-" * 40
        print(separador)
        print(f"  Função        : {self.nome_funcao}")
        print(f"  Execuções     : {len(self.melhores_por_execucao)}")
        print(separador)
        print(f"  Melhor        : {self.melhor_global():.6f}")
        print(f"  Pior          : {self.pior_global():.6f}")
        print(f"  Média         : {self.media_global():.6f}")
        print(f"  Mediana       : {self.mediana():.6f}")
        print(f"  Desvio padrão : {self.desvio_padrao():.6f}")
        print(f"  Taxa conv.    : {self.taxa_convergencia():.6f}")
        print(f"  Iter. conv.   : {self.iteracao_convergencia()}")
        print(separador)