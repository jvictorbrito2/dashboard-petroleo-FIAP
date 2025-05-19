import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates

# Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def carregar_dados_processados():
    """
    Carrega os dados processados do petróleo Brent.
    
    Returns:
        DataFrame com os dados processados.
    """
    csv_path = os.path.join('dados_processados', 'petroleo_brent_processado.csv')
    
    if not os.path.exists(csv_path):
        print(f"Arquivo {csv_path} não encontrado.")
        return None
    
    # Carregar os dados
    df = pd.read_csv(csv_path)
    
    # Converter a coluna de data para datetime
    df['Data'] = pd.to_datetime(df['Data'])
    
    return df

def identificar_eventos_importantes():
    """
    Identifica eventos históricos importantes que afetaram o preço do petróleo.
    
    Returns:
        DataFrame com os eventos importantes.
    """
    # Lista de eventos importantes que afetaram o preço do petróleo
    eventos = [
        {'data': '1990-08-02', 'evento': 'Invasão do Kuwait pelo Iraque', 'descricao': 'Início da Guerra do Golfo, causando aumento nos preços do petróleo.'},
        {'data': '1997-07-01', 'evento': 'Crise Financeira Asiática', 'descricao': 'Crise econômica que afetou vários países asiáticos, reduzindo a demanda por petróleo.'},
        {'data': '1998-12-01', 'evento': 'Baixa Histórica de Preços', 'descricao': 'Preços do petróleo atingiram níveis muito baixos devido à crise asiática e excesso de oferta.'},
        {'data': '2001-09-11', 'evento': 'Ataques Terroristas de 11 de Setembro', 'descricao': 'Impacto nos mercados globais e na demanda por petróleo.'},
        {'data': '2003-03-20', 'evento': 'Invasão do Iraque pelos EUA', 'descricao': 'Segunda Guerra do Golfo, gerando incertezas no mercado de petróleo.'},
        {'data': '2008-07-11', 'evento': 'Pico de Preço Pré-Crise', 'descricao': 'Petróleo atingiu valor recorde antes da crise financeira global.'},
        {'data': '2008-09-15', 'evento': 'Falência do Lehman Brothers', 'descricao': 'Marco da crise financeira global que levou à queda abrupta nos preços do petróleo.'},
        {'data': '2011-01-25', 'evento': 'Primavera Árabe', 'descricao': 'Série de protestos e revoluções no Oriente Médio e Norte da África, afetando a produção de petróleo.'},
        {'data': '2014-06-01', 'evento': 'Início da Queda de Preços', 'descricao': 'Início de um período de queda nos preços devido ao excesso de oferta e desaceleração econômica.'},
        {'data': '2016-01-20', 'evento': 'Mínima Pós-2003', 'descricao': 'Preços atingiram os níveis mais baixos desde 2003 devido ao excesso de oferta.'},
        {'data': '2020-03-09', 'evento': 'Guerra de Preços Rússia-Arábia Saudita', 'descricao': 'Disputa entre produtores levou a uma queda acentuada nos preços.'},
        {'data': '2020-04-20', 'evento': 'Preços Negativos do WTI', 'descricao': 'Pela primeira vez na história, os preços do petróleo WTI ficaram negativos devido à pandemia de COVID-19.'},
        {'data': '2022-02-24', 'evento': 'Invasão da Ucrânia pela Rússia', 'descricao': 'Conflito geopolítico que elevou os preços do petróleo devido a sanções contra a Rússia.'},
        {'data': '2023-10-07', 'evento': 'Conflito Israel-Hamas', 'descricao': 'Escalada de tensões no Oriente Médio, gerando preocupações sobre a oferta de petróleo.'}
    ]
    
    # Criar DataFrame com os eventos
    df_eventos = pd.DataFrame(eventos)
    df_eventos['data'] = pd.to_datetime(df_eventos['data'])
    
    return df_eventos

def analisar_periodos_importantes(df, df_eventos):
    """
    Analisa períodos importantes na série histórica do preço do petróleo.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
        df_eventos: DataFrame com os eventos importantes.
    """
    print("=== Análise de Períodos Importantes ===")
    
    # Criar diretório para salvar as visualizações
    vis_dir = os.path.join(os.getcwd(), 'visualizacoes')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Gráfico da série temporal completa com eventos marcados
    plt.figure(figsize=(16, 10))
    plt.plot(df['Data'], df['Preco'], linewidth=1.5)
    
    # Adicionar linhas verticais e anotações para os eventos importantes
    for i, evento in df_eventos.iterrows():
        if evento['data'] >= df['Data'].min() and evento['data'] <= df['Data'].max():
            plt.axvline(x=evento['data'], color='r', linestyle='--', alpha=0.5)
            plt.annotate(evento['evento'], 
                         xy=(mdates.date2num(evento['data']), 
                             df.loc[df['Data'] <= evento['data'], 'Preco'].iloc[-1]),
                         xytext=(10, 0), 
                         textcoords='offset points',
                         rotation=90,
                         fontsize=8,
                         ha='left',
                         va='bottom')
    
    plt.title('Preço do Petróleo Brent e Eventos Históricos Importantes (1987-2025)')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '08_serie_temporal_eventos.png'), dpi=300)
    plt.close()
    
    # 2. Análise de períodos específicos
    periodos = [
        {'nome': 'Guerra do Golfo', 'inicio': '1990-07-01', 'fim': '1991-03-01'},
        {'nome': 'Crise Asiática', 'inicio': '1997-06-01', 'fim': '1998-12-31'},
        {'nome': 'Pré e Pós 11 de Setembro', 'inicio': '2001-06-01', 'fim': '2001-12-31'},
        {'nome': 'Boom de Commodities', 'inicio': '2007-01-01', 'fim': '2008-07-31'},
        {'nome': 'Crise Financeira Global', 'inicio': '2008-08-01', 'fim': '2009-06-30'},
        {'nome': 'Primavera Árabe', 'inicio': '2011-01-01', 'fim': '2011-12-31'},
        {'nome': 'Queda de 2014-2016', 'inicio': '2014-06-01', 'fim': '2016-12-31'},
        {'nome': 'Pandemia COVID-19', 'inicio': '2020-01-01', 'fim': '2020-12-31'},
        {'nome': 'Conflito Rússia-Ucrânia', 'inicio': '2022-01-01', 'fim': '2022-12-31'}
    ]
    
    # Criar um gráfico para cada período
    for periodo in periodos:
        inicio = pd.to_datetime(periodo['inicio'])
        fim = pd.to_datetime(periodo['fim'])
        
        # Filtrar os dados para o período
        df_periodo = df[(df['Data'] >= inicio) & (df['Data'] <= fim)]
        
        if len(df_periodo) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(df_periodo['Data'], df_periodo['Preco'], linewidth=1.5)
            
            # Adicionar eventos que ocorreram neste período
            eventos_periodo = df_eventos[(df_eventos['data'] >= inicio) & (df_eventos['data'] <= fim)]
            for i, evento in eventos_periodo.iterrows():
                plt.axvline(x=evento['data'], color='r', linestyle='--', alpha=0.7)
                plt.annotate(evento['evento'], 
                             xy=(mdates.date2num(evento['data']), 
                                 df_periodo.loc[df_periodo['Data'] <= evento['data'], 'Preco'].iloc[-1]),
                             xytext=(10, 0), 
                             textcoords='offset points',
                             fontsize=10,
                             ha='left',
                             va='bottom')
            
            plt.title(f'Preço do Petróleo Brent durante {periodo["nome"]} ({inicio.strftime("%m/%Y")} - {fim.strftime("%m/%Y")})')
            plt.xlabel('Data')
            plt.ylabel('Preço (USD)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'09_periodo_{periodo["nome"].replace(" ", "_").lower()}.png'))
            plt.close()
            
            # Calcular estatísticas do período
            preco_inicial = df_periodo['Preco'].iloc[0]
            preco_final = df_periodo['Preco'].iloc[-1]
            variacao_abs = preco_final - preco_inicial
            variacao_pct = (variacao_abs / preco_inicial) * 100
            
            print(f"\nPeríodo: {periodo['nome']} ({inicio.strftime('%m/%Y')} - {fim.strftime('%m/%Y')})")
            print(f"Preço inicial: ${preco_inicial:.2f}")
            print(f"Preço final: ${preco_final:.2f}")
            print(f"Variação absoluta: ${variacao_abs:.2f}")
            print(f"Variação percentual: {variacao_pct:.2f}%")
            print(f"Preço médio: ${df_periodo['Preco'].mean():.2f}")
            print(f"Preço máximo: ${df_periodo['Preco'].max():.2f}")
            print(f"Preço mínimo: ${df_periodo['Preco'].min():.2f}")
            print(f"Volatilidade (desvio padrão): ${df_periodo['Preco'].std():.2f}")
    
    # 3. Análise de volatilidade por década
    df['Decada'] = (df['Ano'] // 10) * 10
    volatilidade_decada = df.groupby('Decada')['Variacao'].std().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(volatilidade_decada['Decada'].astype(str), volatilidade_decada['Variacao'])
    plt.title('Volatilidade do Preço do Petróleo Brent por Década')
    plt.xlabel('Década')
    plt.ylabel('Volatilidade (Desvio Padrão da Variação Diária %)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '10_volatilidade_decada.png'))
    plt.close()
    
    print("\nVolatilidade por Década:")
    for i, row in volatilidade_decada.iterrows():
        print(f"Década de {int(row['Decada'])}: {row['Variacao']:.2f}%")
    
    # 4. Análise de correlação entre eventos e variações extremas
    df['Variacao_Abs'] = df['Variacao'].abs()
    
    # Identificar os dias com maiores variações
    top_variacoes = df.nlargest(20, 'Variacao_Abs')
    
    print("\nDias com Maiores Variações de Preço:")
    for i, row in top_variacoes.iterrows():
        print(f"Data: {row['Data'].strftime('%d/%m/%Y')}, Variação: {row['Variacao']:.2f}%, Preço: ${row['Preco']:.2f}")
        
        # Verificar se há eventos próximos (7 dias antes ou depois)
        data_inicio = row['Data'] - pd.Timedelta(days=7)
        data_fim = row['Data'] + pd.Timedelta(days=7)
        eventos_proximos = df_eventos[(df_eventos['data'] >= data_inicio) & (df_eventos['data'] <= data_fim)]
        
        if len(eventos_proximos) > 0:
            for j, evento in eventos_proximos.iterrows():
                dias_diff = (evento['data'] - row['Data']).days
                print(f"  Evento próximo ({dias_diff:+d} dias): {evento['evento']} - {evento['descricao']}")
    
    # 5. Análise de tendências de longo prazo
    # Calcular médias móveis de longo prazo (90 dias, 180 dias, 365 dias)
    df['MM90'] = df['Preco'].rolling(window=90).mean()
    df['MM180'] = df['Preco'].rolling(window=180).mean()
    df['MM365'] = df['Preco'].rolling(window=365).mean()
    
    plt.figure(figsize=(16, 8))
    plt.plot(df['Data'], df['Preco'], linewidth=1, alpha=0.7, label='Preço Diário')
    plt.plot(df['Data'], df['MM90'], linewidth=2, label='Média Móvel 90 dias')
    plt.plot(df['Data'], df['MM180'], linewidth=2, label='Média Móvel 180 dias')
    plt.plot(df['Data'], df['MM365'], linewidth=2, label='Média Móvel 365 dias')
    
    plt.title('Tendências de Longo Prazo do Preço do Petróleo Brent')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '11_tendencias_longo_prazo.png'))
    plt.close()
    
    # 6. Análise de sazonalidade anual
    sazonalidade_mensal = df.groupby(['Ano', 'Mes'])['Preco'].mean().reset_index()
    sazonalidade_pivot = sazonalidade_mensal.pivot(index='Ano', columns='Mes', values='Preco')
    
    # Calcular a média de cada mês ao longo dos anos
    media_mensal = sazonalidade_pivot.mean()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, 13), media_mensal)
    plt.title('Sazonalidade Mensal do Preço do Petróleo Brent')
    plt.xlabel('Mês')
    plt.ylabel('Preço Médio (USD)')
    plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '12_sazonalidade_mensal.png'))
    plt.close()
    
    print("\nSazonalidade Mensal (Média Histórica):")
    for mes, preco in enumerate(media_mensal, 1):
        print(f"Mês {mes}: ${preco:.2f}")

def salvar_eventos_importantes(df_eventos):
    """
    Salva os eventos importantes em um arquivo CSV.
    
    Args:
        df_eventos: DataFrame com os eventos importantes.
    """
    # Criar diretório de dados processados se não existir
    data_dir = os.path.join(os.getcwd(), 'dados_processados')
    os.makedirs(data_dir, exist_ok=True)
    
    # Salvar os eventos
    csv_path = os.path.join(data_dir, 'eventos_importantes.csv')
    df_eventos.to_csv(csv_path, index=False)
    
    print(f"\nEventos importantes salvos em: {csv_path}")
    print(f"Total de eventos: {len(df_eventos)}")

def main():
    # Carregar os dados processados
    df = carregar_dados_processados()
    
    if df is None:
        print("Não foi possível carregar os dados processados. Encerrando.")
        return
    
    # Identificar eventos importantes
    df_eventos = identificar_eventos_importantes()
    
    # Analisar períodos importantes
    analisar_periodos_importantes(df, df_eventos)
    
    # Salvar eventos importantes
    salvar_eventos_importantes(df_eventos)
    
    print("\nExploração de variações e eventos importantes concluída com sucesso!")

if __name__ == "__main__":
    main()
