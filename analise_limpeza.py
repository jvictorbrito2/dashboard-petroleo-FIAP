import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def carregar_dados():
    """
    Carrega os dados do petróleo Brent do arquivo CSV.
    
    Returns:
        DataFrame com os dados do petróleo Brent.
    """
    csv_path = os.path.join('dados', 'petroleo_brent.csv')
    
    if not os.path.exists(csv_path):
        print(f"Arquivo {csv_path} não encontrado.")
        return None
    
    # Carregar os dados
    df = pd.read_csv(csv_path)
    
    # Converter a coluna de data para datetime
    df['Data'] = pd.to_datetime(df['Data'])
    
    return df

def analisar_dados(df):
    """
    Realiza uma análise inicial dos dados.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
    """
    print("=== Análise Inicial dos Dados ===")
    
    # Informações gerais
    print("\nInformações gerais:")
    print(f"Número de registros: {len(df)}")
    print(f"Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
    
    # Estatísticas descritivas
    print("\nEstatísticas descritivas do preço:")
    print(df['Preco'].describe())
    
    # Verificar valores ausentes
    print("\nValores ausentes:")
    print(df.isnull().sum())
    
    # Verificar a frequência dos dados
    df = df.sort_values('Data')
    df['dias_entre'] = df['Data'].diff().dt.days
    
    print("\nFrequência dos dados:")
    print(df['dias_entre'].value_counts().head(10))
    
    media_dias = df['dias_entre'].mean()
    print(f"Média de dias entre registros: {media_dias:.2f}")
    
    # Remover a coluna auxiliar
    df = df.drop('dias_entre', axis=1)
    
    return df

def identificar_outliers(df):
    """
    Identifica possíveis outliers nos dados de preço.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
        
    Returns:
        DataFrame com os dados tratados.
    """
    print("\n=== Identificação de Outliers ===")
    
    # Calcular limites para outliers usando o método IQR
    Q1 = df['Preco'].quantile(0.25)
    Q3 = df['Preco'].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    print(f"Limite inferior para outliers: ${limite_inferior:.2f}")
    print(f"Limite superior para outliers: ${limite_superior:.2f}")
    
    # Identificar outliers
    outliers = df[(df['Preco'] < limite_inferior) | (df['Preco'] > limite_superior)]
    
    print(f"Número de possíveis outliers: {len(outliers)}")
    
    if len(outliers) > 0:
        print("\nExemplos de outliers:")
        print(outliers.head(10))
    
    # Não vamos remover os outliers, pois podem representar eventos importantes
    # como crises ou choques de oferta/demanda
    print("\nNota: Não removeremos os outliers, pois podem representar eventos importantes na série histórica.")
    
    return df

def criar_features_temporais(df):
    """
    Cria features baseadas em tempo para análise e modelagem.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
        
    Returns:
        DataFrame com as novas features.
    """
    print("\n=== Criação de Features Temporais ===")
    
    # Criar colunas de ano, mês, dia da semana
    df['Ano'] = df['Data'].dt.year
    df['Mes'] = df['Data'].dt.month
    df['DiaSemana'] = df['Data'].dt.dayofweek
    
    # Criar coluna de trimestre
    df['Trimestre'] = df['Data'].dt.quarter
    
    # Criar coluna para indicar se é fim de mês
    df['FimMes'] = df['Data'].dt.is_month_end
    
    # Criar coluna para variação diária
    df = df.sort_values('Data')
    df['Variacao'] = df['Preco'].pct_change() * 100
    
    # Criar coluna para média móvel de 7 e 30 dias
    df['MM7'] = df['Preco'].rolling(window=7).mean()
    df['MM30'] = df['Preco'].rolling(window=30).mean()
    
    print("Features temporais criadas:")
    print(df.columns.tolist())
    
    return df

def visualizar_dados(df):
    """
    Cria visualizações preliminares dos dados.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
    """
    print("\n=== Criando Visualizações Preliminares ===")
    
    # Criar diretório para salvar as visualizações
    vis_dir = os.path.join(os.getcwd(), 'visualizacoes')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Série temporal completa
    plt.figure(figsize=(14, 7))
    plt.plot(df['Data'], df['Preco'], linewidth=1)
    plt.title('Preço do Petróleo Brent (1987-2025)')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '01_serie_temporal_completa.png'))
    plt.close()
    
    # 2. Histograma de preços
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Preco'], bins=50, kde=True)
    plt.title('Distribuição dos Preços do Petróleo Brent')
    plt.xlabel('Preço (USD)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '02_histograma_precos.png'))
    plt.close()
    
    # 3. Boxplot por década
    df['Decada'] = (df['Ano'] // 10) * 10
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Decada', y='Preco', data=df)
    plt.title('Distribuição de Preços por Década')
    plt.xlabel('Década')
    plt.ylabel('Preço (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '03_boxplot_decada.png'))
    plt.close()
    
    # 4. Preço médio por ano
    preco_anual = df.groupby('Ano')['Preco'].mean().reset_index()
    plt.figure(figsize=(14, 7))
    plt.bar(preco_anual['Ano'], preco_anual['Preco'])
    plt.title('Preço Médio Anual do Petróleo Brent')
    plt.xlabel('Ano')
    plt.ylabel('Preço Médio (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '04_preco_medio_anual.png'))
    plt.close()
    
    # 5. Variação diária
    plt.figure(figsize=(14, 7))
    plt.plot(df['Data'], df['Variacao'], linewidth=0.8)
    plt.title('Variação Diária do Preço do Petróleo Brent (%)')
    plt.xlabel('Data')
    plt.ylabel('Variação (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '05_variacao_diaria.png'))
    plt.close()
    
    # 6. Preço com médias móveis
    plt.figure(figsize=(14, 7))
    plt.plot(df['Data'], df['Preco'], linewidth=1, label='Preço Diário')
    plt.plot(df['Data'], df['MM7'], linewidth=1.5, label='Média Móvel 7 dias')
    plt.plot(df['Data'], df['MM30'], linewidth=1.5, label='Média Móvel 30 dias')
    plt.title('Preço do Petróleo Brent com Médias Móveis')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '06_preco_medias_moveis.png'))
    plt.close()
    
    # 7. Sazonalidade mensal
    preco_mensal = df.groupby('Mes')['Preco'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.bar(preco_mensal['Mes'], preco_mensal['Preco'])
    plt.title('Preço Médio Mensal do Petróleo Brent')
    plt.xlabel('Mês')
    plt.ylabel('Preço Médio (USD)')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '07_sazonalidade_mensal.png'))
    plt.close()
    
    print(f"Visualizações salvas no diretório: {vis_dir}")

def salvar_dados_limpos(df):
    """
    Salva os dados limpos e processados em um novo arquivo CSV.
    
    Args:
        df: DataFrame com os dados processados.
    """
    # Criar diretório de dados processados se não existir
    data_dir = os.path.join(os.getcwd(), 'dados_processados')
    os.makedirs(data_dir, exist_ok=True)
    
    # Salvar os dados processados
    csv_path = os.path.join(data_dir, 'petroleo_brent_processado.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nDados processados salvos em: {csv_path}")
    print(f"Total de registros: {len(df)}")

def main():
    # Carregar os dados
    df = carregar_dados()
    
    if df is None:
        print("Não foi possível carregar os dados. Encerrando.")
        return
    
    # Analisar os dados
    df = analisar_dados(df)
    
    # Identificar outliers
    df = identificar_outliers(df)
    
    # Criar features temporais
    df = criar_features_temporais(df)
    
    # Visualizar os dados
    visualizar_dados(df)
    
    # Salvar os dados processados
    salvar_dados_limpos(df)
    
    print("\nProcesso de análise e limpeza de dados concluído com sucesso!")

if __name__ == "__main__":
    main()
