import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from pathlib import Path

class plot:
    def __init__(self,diretorioGraficos='data/results'):
        self.diretorioGraficos=Path(diretorioGraficos)
        self.diretorioGraficos.mkdir(parents=True, exist_ok=True)


    def _salvar(self,filename,show):
        if filename:
            caminho = self.diretorioGraficos / filename
            plt.savefig(caminho, dpi=400, bbox_inches='tight')
            print(f"  Gráfico salvo em: {caminho}")
        if show:
            plt.show()
        else:
            plt.close()
    def PlotConvergenciaAlgoritmo(self,historia,title="convergencia ABC",filename=None,show=True):
        
        plt.figure(figsize=(12,8))

        interacoes = list(range(len(historia['best_fitness'])))
        
        plt.plot(interacoes, historia['best_fitness'], label='Melhor', linewidth=3,color='blue')
        plt.plot(interacoes, historia['mean_fitness'], label='medio', linewidth=3,color='yellow')
        plt.plot(interacoes, historia['worst_fitness'], label='pior', linewidth=3,color='red')
        

        plt.xlabel('Iteracao', fontsize=14)
        plt.ylabel('performance', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4)
        plt.yscale('log')  
        plt.tight_layout()

        self._salvar(filename,show)

    def PlotConvergenciaMedia(self, historicos, nome_funcao='',filename=None,show=True):
        num_iter = min(len(h['best_fitness']) for h in historicos)
        curvas   = np.array([[h['best_fitness'][i] for i in range(num_iter)]
                              for h in historicos])
        media  = curvas.mean(axis=0)
        desvio = curvas.std(axis=0)
 
        plt.figure(figsize=(12, 8))
        iters = range(num_iter)
        plt.plot(iters, media, linewidth=2, label='Média')
        plt.fill_between(iters, media - desvio, media + desvio,
                         alpha=0.2, label='±1 desvio padrão')
 
        plt.xlabel('Iteração',fontsize=14)
        plt.ylabel('Fitness',fontsize=14)
        plt.title(f'Convergência Média — {nome_funcao}', fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4)
        plt.yscale('log')
        plt.tight_layout()
 
        self._salvar(filename, show)
    def PlotConvergenciaMultipla(self,historicos,title='comparativo das convergencias',filename=None,show=True):
        plt.figure(figsize=(15,12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(historicos)))

        for i, (name, history) in enumerate(historicos.items()):
            iterations = range(len(history['best_fitness']))
            plt.plot(iterations, history['best_fitness'], 
                    label=name, linewidth=4, color=colors[i])
        
        plt.xlabel('Iteração', fontsize=14)
        plt.ylabel('Melhor Fitness', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()

        self._salvar(filename,show)

    def PlotBoxplot(self,resultados,filename=None,show =True):
        nomes = list(resultados.keys())
        dados = [resultados[n] for n in nomes]

        fig, ax = plt.subplots(figsize=(max(8, len(nomes) * 2), 6))
        bp = ax.boxplot(dados, patch_artist=True, labels=nomes)
 
        cores = plt.cm.tab10.colors
        for patch, cor in zip(bp['boxes'], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
 
        ax.set_title('Distribuição dos Melhores Fitness por Função',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Função',        fontsize=14)
        ax.set_ylabel('Melhor Fitness', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
 
        self._salvar(filename, show)

    def CriarTabelaEstatistica(self,estatistica,filename=None,show=True):

        if estatistica.empty:
            raise ValueError("dataframe sem dados estatisticos")
        

        tamanho_altura = max(2, len(estatistica) * 0.7)
        tamanho_largura = max(8, len(estatistica.columns) * 1.6)

        tamanho, areaTabela = plt.subplots(figsize=(tamanho_largura,tamanho_altura))

        areaTabela.axis('tight')
        areaTabela.axis('off')

        valores_formatados = estatistica.map(
            lambda i: f"{i:.3f}" if isinstance(i,float) else i).values
        
        tabela = areaTabela.table(
            cellText=valores_formatados,
            colLabels=estatistica.columns,
            rowLabels=estatistica.index,
            cellLoc='center',
            loc='center'
        )

        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        tabela.scale(1,1.6)   

        for (linha,coluna), celula in tabela.get_celld().items():
            if linha == 0:
                celula.set_facecolor('#4CAF54')
                celula.set_text_props(weight='bold',color='blue')
        
        plt.title('estatistica dos experimentos',fontsize =15,fontweight='bold',pad=22)
        
 
        self._salvar(filename, show)
 


        
