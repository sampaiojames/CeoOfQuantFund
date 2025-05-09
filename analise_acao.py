import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt

class AnaliseAcao:
    def __init__(self, ticker, inicio, fim):
        self.ticker = ticker
        self.inicio = inicio
        self.fim = fim
        self.dados = None
        self.dados_tratados = None

    def baixar_dados(self):
        print(f"Baixando dados de {self.ticker}...")
        self.dados = yf.download(self.ticker, start=self.inicio, end=self.fim, threads=False, multi_level_index=False)
        return self.dados

    def tratar_dados(self):
        if self.dados is None:
            raise ValueError("Dados não baixados ainda. Use baixar_dados() primeiro.")

        print("Tratando dados...")
        self.dados_tratados = self.dados.dropna()
        self.dados_tratados['Retorno'] = self.dados_tratados['Close'].pct_change()
        return self.dados_tratados

    def plotar_dados(self):
        if self.dados_tratados is None:
            raise ValueError("Dados não tratados ainda. Use tratar_dados() primeiro.")
        print("Plotando dados...")
        plt.figure(figsize=(10,5))
        plt.plot(self.dados_tratados.index, self.dados_tratados['Close'], label= f'Preço {self.ticker}')
        plt.title(f'Preço de {self.ticker}')
        plt.xlabel('Data')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def executar_backtest(self):
        if self.dados_tratados is None:
            raise ValueError("Dados não tratados ainda. Use tratar_dados() primeiro.")
        print("Executando backtest com estratégia simples (Golden Cross)...")
        close = self.dados_tratados['Close']
        fast_ma = close.rolling(window=9).mean()
        slow_ma = close.rolling(window=21).mean()
        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            freq='1D',
            fees=0.001
        )
        pf.plot().show()
        return pf.stats()

    def executar_analise(self):
        self.baixar_dados()
        self.tratar_dados()
        self.plotar_dados()
        return self.dados_tratados



