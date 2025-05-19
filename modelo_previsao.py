import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def carregar_dados():
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

def preparar_dados_para_modelo(df, janela=30, previsao_dias=30, split_ratio=0.8):
    """
    Prepara os dados para o modelo de previsão.
    
    Args:
        df: DataFrame com os dados do petróleo Brent.
        janela: Número de dias anteriores a serem usados para previsão.
        previsao_dias: Número de dias a serem previstos.
        split_ratio: Proporção dos dados para treinamento.
        
    Returns:
        Dados preparados para o modelo.
    """
    print("=== Preparando Dados para o Modelo ===")
    
    # Ordenar os dados por data
    df = df.sort_values('Data')
    
    # Selecionar as features para o modelo
    features = ['Preco', 'Variacao', 'MM7', 'MM30']
    
    # Adicionar features de dia da semana, mês e trimestre
    df['DiaSemana_sin'] = np.sin(2 * np.pi * df['DiaSemana'] / 7)
    df['DiaSemana_cos'] = np.cos(2 * np.pi * df['DiaSemana'] / 7)
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
    df['Trimestre_sin'] = np.sin(2 * np.pi * df['Trimestre'] / 4)
    df['Trimestre_cos'] = np.cos(2 * np.pi * df['Trimestre'] / 4)
    
    # Adicionar features de tendência
    df['Tendencia'] = np.arange(len(df))
    df['Tendencia'] = df['Tendencia'] / df['Tendencia'].max()
    
    # Selecionar as features finais
    features_finais = ['Preco', 'Variacao', 'MM7', 'MM30', 
                      'DiaSemana_sin', 'DiaSemana_cos', 
                      'Mes_sin', 'Mes_cos', 
                      'Trimestre_sin', 'Trimestre_cos',
                      'Tendencia']
    
    # Selecionar apenas as colunas necessárias
    df_modelo = df[['Data'] + features_finais].copy()
    
    # Remover linhas com valores ausentes
    df_modelo = df_modelo.dropna()
    
    # Normalizar os dados
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Ajustar o scaler para os dados de entrada
    dados_normalizados = scaler_X.fit_transform(df_modelo[features_finais])
    
    # Ajustar o scaler apenas para o preço (target)
    scaler_y.fit(df_modelo[['Preco']])
    
    # Criar sequências para o modelo LSTM
    X, y = [], []
    
    for i in range(len(dados_normalizados) - janela - previsao_dias + 1):
        X.append(dados_normalizados[i:i+janela])
        # Para previsão de múltiplos dias, usamos os próximos 'previsao_dias' valores
        y.append(dados_normalizados[i+janela:i+janela+previsao_dias, 0])  # Índice 0 corresponde ao preço
    
    X = np.array(X)
    y = np.array(y)
    
    # Dividir os dados em conjuntos de treinamento e teste
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Datas correspondentes para o conjunto de teste
    datas_teste = df_modelo['Data'].iloc[split+janela:split+janela+len(X_test)]
    
    print(f"Tamanho do conjunto de treinamento: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")
    print(f"Formato dos dados de entrada: {X_train.shape}")
    print(f"Formato dos dados de saída: {y_train.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'datas_teste': datas_teste,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': features_finais,
        'janela': janela,
        'previsao_dias': previsao_dias,
        'df_modelo': df_modelo
    }

def criar_modelo_lstm(input_shape, previsao_dias):
    """
    Cria um modelo LSTM para previsão de séries temporais.
    
    Args:
        input_shape: Formato dos dados de entrada.
        previsao_dias: Número de dias a serem previstos.
        
    Returns:
        Modelo LSTM compilado.
    """
    print("=== Criando Modelo LSTM ===")
    
    modelo = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(previsao_dias)
    ])
    
    modelo.compile(optimizer='adam', loss='mse')
    
    print(modelo.summary())
    
    return modelo

def treinar_modelo(modelo, dados_preparados, epochs=100, batch_size=32, patience=20):
    """
    Treina o modelo LSTM.
    
    Args:
        modelo: Modelo LSTM compilado.
        dados_preparados: Dados preparados para o modelo.
        epochs: Número máximo de épocas de treinamento.
        batch_size: Tamanho do lote para treinamento.
        patience: Número de épocas sem melhoria para parar o treinamento.
        
    Returns:
        Histórico de treinamento.
    """
    print("=== Treinando o Modelo ===")
    
    # Configurar callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Criar diretório para salvar o modelo
    modelo_dir = os.path.join(os.getcwd(), 'modelo')
    os.makedirs(modelo_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        os.path.join(modelo_dir, 'melhor_modelo.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Treinar o modelo
    historico = modelo.fit(
        dados_preparados['X_train'],
        dados_preparados['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return historico

def avaliar_modelo(modelo, dados_preparados):
    """
    Avalia o desempenho do modelo.
    
    Args:
        modelo: Modelo treinado.
        dados_preparados: Dados preparados para o modelo.
        
    Returns:
        Métricas de avaliação.
    """
    print("=== Avaliando o Modelo ===")
    
    # Fazer previsões no conjunto de teste
    previsoes_norm = modelo.predict(dados_preparados['X_test'])
    
    # Preparar para desnormalização
    previsoes_desnorm = np.zeros((previsoes_norm.shape[0], previsoes_norm.shape[1], 1))
    y_test_desnorm = np.zeros((dados_preparados['y_test'].shape[0], dados_preparados['y_test'].shape[1], 1))
    
    # Desnormalizar as previsões e os valores reais
    for i in range(previsoes_norm.shape[0]):
        for j in range(previsoes_norm.shape[1]):
            previsoes_desnorm[i, j, 0] = previsoes_norm[i, j]
            y_test_desnorm[i, j, 0] = dados_preparados['y_test'][i, j]
    
    # Reshape para desnormalização
    previsoes_desnorm = previsoes_desnorm.reshape(-1, 1)
    y_test_desnorm = y_test_desnorm.reshape(-1, 1)
    
    # Desnormalizar
    previsoes = dados_preparados['scaler_y'].inverse_transform(previsoes_desnorm)
    y_real = dados_preparados['scaler_y'].inverse_transform(y_test_desnorm)
    
    # Reshape de volta para o formato original
    previsoes = previsoes.reshape(dados_preparados['y_test'].shape)
    y_real = y_real.reshape(dados_preparados['y_test'].shape)
    
    # Calcular métricas para cada horizonte de previsão
    metricas = []
    
    for i in range(dados_preparados['previsao_dias']):
        rmse = np.sqrt(mean_squared_error(y_real[:, i], previsoes[:, i]))
        mae = mean_absolute_error(y_real[:, i], previsoes[:, i])
        mape = np.mean(np.abs((y_real[:, i] - previsoes[:, i]) / y_real[:, i])) * 100
        r2 = r2_score(y_real[:, i], previsoes[:, i])
        
        metricas.append({
            'Horizonte': i+1,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })
    
    # Criar DataFrame com as métricas
    df_metricas = pd.DataFrame(metricas)
    
    print("\nMétricas de avaliação por horizonte de previsão:")
    print(df_metricas)
    
    # Calcular métricas gerais
    rmse_geral = np.sqrt(mean_squared_error(y_real.flatten(), previsoes.flatten()))
    mae_geral = mean_absolute_error(y_real.flatten(), previsoes.flatten())
    mape_geral = np.mean(np.abs((y_real.flatten() - previsoes.flatten()) / y_real.flatten())) * 100
    r2_geral = r2_score(y_real.flatten(), previsoes.flatten())
    
    print(f"\nMétricas gerais:")
    print(f"RMSE: ${rmse_geral:.2f}")
    print(f"MAE: ${mae_geral:.2f}")
    print(f"MAPE: {mape_geral:.2f}%")
    print(f"R²: {r2_geral:.4f}")
    
    # Criar diretório para salvar as visualizações
    vis_dir = os.path.join(os.getcwd(), 'visualizacoes')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualizar as previsões vs. valores reais para diferentes horizontes
    horizontes = [0, 6, 13, 29]  # 1 dia, 1 semana, 2 semanas, 1 mês
    
    plt.figure(figsize=(14, 10))
    
    for i, h in enumerate(horizontes):
        plt.subplot(2, 2, i+1)
        plt.plot(y_real[:, h], label='Real')
        plt.plot(previsoes[:, h], label='Previsto')
        plt.title(f'Horizonte: {h+1} dias')
        plt.xlabel('Índice do Teste')
        plt.ylabel('Preço (USD)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, '13_previsoes_vs_real_horizontes.png'))
    plt.close()
    
    # Visualizar as previsões vs. valores reais para um exemplo específico
    exemplo_idx = len(previsoes) // 2  # Meio do conjunto de teste
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, dados_preparados['previsao_dias']+1), y_real[exemplo_idx], 'b-', label='Real')
    plt.plot(range(1, dados_preparados['previsao_dias']+1), previsoes[exemplo_idx], 'r--', label='Previsto')
    plt.title(f'Previsão vs. Real para {dados_preparados["previsao_dias"]} dias')
    plt.xlabel('Horizonte de Previsão (dias)')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, '14_previsao_exemplo.png'))
    plt.close()
    
    # Visualizar o erro de previsão por horizonte
    plt.figure(figsize=(12, 6))
    plt.plot(df_metricas['Horizonte'], df_metricas['RMSE'], 'o-')
    plt.title('Erro de Previsão (RMSE) por Horizonte')
    plt.xlabel('Horizonte de Previsão (dias)')
    plt.ylabel('RMSE (USD)')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, '15_erro_por_horizonte.png'))
    plt.close()
    
    # Visualizar a distribuição dos erros
    erros = y_real.flatten() - previsoes.flatten()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(erros, kde=True)
    plt.title('Distribuição dos Erros de Previsão')
    plt.xlabel('Erro (USD)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, '16_distribuicao_erros.png'))
    plt.close()
    
    return {
        'previsoes': previsoes,
        'y_real': y_real,
        'metricas': df_metricas,
        'rmse_geral': rmse_geral,
        'mae_geral': mae_geral,
        'mape_geral': mape_geral,
        'r2_geral': r2_geral
    }

def fazer_previsao_futura(modelo, dados_preparados, dias_futuros=30):
    """
    Faz previsões para os próximos dias.
    
    Args:
        modelo: Modelo treinado.
        dados_preparados: Dados preparados para o modelo.
        dias_futuros: Número de dias para prever no futuro.
        
    Returns:
        DataFrame com as previsões futuras.
    """
    print(f"=== Fazendo Previsões para os Próximos {dias_futuros} Dias ===")
    
    # Obter os dados mais recentes para a janela de entrada
    df_modelo = dados_preparados['df_modelo']
    janela = dados_preparados['janela']
    previsao_dias = dados_preparados['previsao_dias']
    
    # Obter os últimos 'janela' dias de dados
    ultimos_dados = df_modelo.iloc[-janela:][dados_preparados['features']].values
    
    # Normalizar os dados
    ultimos_dados_norm = dados_preparados['scaler_X'].transform(ultimos_dados)
    
    # Reshape para o formato esperado pelo modelo
    X_futuro = np.array([ultimos_dados_norm])
    
    # Fazer a previsão
    previsao_norm = modelo.predict(X_futuro)
    
    # Preparar para desnormalização
    previsao_desnorm = np.zeros((1, previsao_dias, 1))
    
    # Preparar para desnormalização
    for j in range(previsao_dias):
        previsao_desnorm[0, j, 0] = previsao_norm[0, j]
    
    # Reshape para desnormalização
    previsao_desnorm = previsao_desnorm.reshape(-1, 1)
    
    # Desnormalizar
    previsao = dados_preparados['scaler_y'].inverse_transform(previsao_desnorm)
    
    # Reshape de volta para o formato original
    previsao = previsao.reshape(1, previsao_dias)[0]
    
    # Criar datas futuras
    ultima_data = df_modelo['Data'].iloc[-1]
    datas_futuras = [ultima_data + timedelta(days=i+1) for i in range(previsao_dias)]
    
    # Criar DataFrame com as previsões
    df_previsao = pd.DataFrame({
        'Data': datas_futuras,
        'Preco_Previsto': previsao
    })
    
    print("\nPrevisões para os próximos dias:")
    print(df_previsao)
    
    # Criar diretório para salvar as visualizações
    vis_dir = os.path.join(os.getcwd(), 'visualizacoes')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualizar as previsões futuras
    plt.figure(figsize=(12, 6))
    
    # Plotar os últimos 60 dias de dados históricos
    plt.plot(df_modelo['Data'].iloc[-60:], df_modelo['Preco'].iloc[-60:], 'b-', label='Histórico')
    
    # Plotar as previsões futuras
    plt.plot(df_previsao['Data'], df_previsao['Preco_Previsto'], 'r--', label='Previsão')
    
    # Adicionar uma linha vertical para separar histórico e previsão
    plt.axvline(x=ultima_data, color='gray', linestyle='--')
    
    plt.title(f'Previsão do Preço do Petróleo Brent para os Próximos {dias_futuros} Dias')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, '17_previsao_futura.png'))
    plt.close()
    
    # Salvar as previsões em CSV
    csv_path = os.path.join('dados_processados', 'previsao_futura.csv')
    df_previsao.to_csv(csv_path, index=False)
    
    print(f"\nPrevisões futuras salvas em: {csv_path}")
    
    return df_previsao

def salvar_modelo_e_artefatos(modelo, dados_preparados, resultados_avaliacao):
    """
    Salva o modelo treinado e artefatos relacionados.
    
    Args:
        modelo: Modelo treinado.
        dados_preparados: Dados preparados para o modelo.
        resultados_avaliacao: Resultados da avaliação do modelo.
    """
    print("=== Salvando Modelo e Artefatos ===")
    
    # Criar diretório para salvar o modelo
    modelo_dir = os.path.join(os.getcwd(), 'modelo')
    os.makedirs(modelo_dir, exist_ok=True)
    
    # Salvar o modelo
    modelo.save(os.path.join(modelo_dir, 'modelo_lstm.h5'))
    
    # Salvar os scalers
    joblib.dump(dados_preparados['scaler_X'], os.path.join(modelo_dir, 'scaler_X.pkl'))
    joblib.dump(dados_preparados['scaler_y'], os.path.join(modelo_dir, 'scaler_y.pkl'))
    
    # Salvar os parâmetros do modelo
    parametros = {
        'janela': dados_preparados['janela'],
        'previsao_dias': dados_preparados['previsao_dias'],
        'features': dados_preparados['features'],
        'metricas': {
            'rmse_geral': resultados_avaliacao['rmse_geral'],
            'mae_geral': resultados_avaliacao['mae_geral'],
            'mape_geral': resultados_avaliacao['mape_geral'],
            'r2_geral': resultados_avaliacao['r2_geral']
        }
    }
    
    joblib.dump(parametros, os.path.join(modelo_dir, 'parametros.pkl'))
    
    # Salvar as métricas em CSV
    resultados_avaliacao['metricas'].to_csv(os.path.join(modelo_dir, 'metricas_por_horizonte.csv'), index=False)
    
    print(f"Modelo e artefatos salvos no diretório: {modelo_dir}")

def criar_documentacao_modelo(dados_preparados, resultados_avaliacao):
    """
    Cria documentação sobre o modelo e seus resultados.
    
    Args:
        dados_preparados: Dados preparados para o modelo.
        resultados_avaliacao: Resultados da avaliação do modelo.
    """
    print("=== Criando Documentação do Modelo ===")
    
    # Criar diretório para a documentação
    docs_dir = os.path.join(os.getcwd(), 'documentacao')
    os.makedirs(docs_dir, exist_ok=True)
    
    # Criar arquivo de documentação
    doc_path = os.path.join(docs_dir, 'documentacao_modelo.md')
    
    with open(doc_path, 'w') as f:
        f.write("# Documentação do Modelo de Previsão do Preço do Petróleo Brent\n\n")
        
        f.write("## Visão Geral\n\n")
        f.write("Este documento descreve o modelo de Machine Learning desenvolvido para prever o preço diário do petróleo Brent.\n\n")
        
        f.write("## Arquitetura do Modelo\n\n")
        f.write("O modelo utiliza uma arquitetura de Rede Neural Recorrente (RNN) do tipo LSTM (Long Short-Term Memory), ")
        f.write("que é especialmente adequada para previsão de séries temporais devido à sua capacidade de capturar dependências de longo prazo nos dados.\n\n")
        
        f.write("### Estrutura da Rede\n\n")
        f.write("- Camada LSTM 1: 100 unidades, com retorno de sequências\n")
        f.write("- Camada Dropout 1: 20% para evitar overfitting\n")
        f.write("- Camada LSTM 2: 50 unidades\n")
        f.write("- Camada Dropout 2: 20% para evitar overfitting\n")
        f.write(f"- Camada Dense (saída): {dados_preparados['previsao_dias']} unidades (uma para cada dia de previsão)\n\n")
        
        f.write("### Parâmetros do Modelo\n\n")
        f.write(f"- Janela de entrada: {dados_preparados['janela']} dias\n")
        f.write(f"- Horizonte de previsão: {dados_preparados['previsao_dias']} dias\n")
        f.write(f"- Features utilizadas: {', '.join(dados_preparados['features'])}\n")
        f.write("- Função de perda: Erro Quadrático Médio (MSE)\n")
        f.write("- Otimizador: Adam\n\n")
        
        f.write("## Preparação dos Dados\n\n")
        f.write("### Pré-processamento\n\n")
        f.write("- Normalização das features usando MinMaxScaler\n")
        f.write("- Criação de features cíclicas para dia da semana, mês e trimestre\n")
        f.write("- Adição de feature de tendência\n")
        f.write("- Criação de sequências de entrada com janela deslizante\n\n")
        
        f.write("### Divisão dos Dados\n\n")
        f.write(f"- Conjunto de treinamento: {len(dados_preparados['X_train'])} amostras\n")
        f.write(f"- Conjunto de teste: {len(dados_preparados['X_test'])} amostras\n\n")
        
        f.write("## Desempenho do Modelo\n\n")
        f.write("### Métricas Gerais\n\n")
        f.write(f"- RMSE (Erro Quadrático Médio): ${resultados_avaliacao['rmse_geral']:.2f}\n")
        f.write(f"- MAE (Erro Absoluto Médio): ${resultados_avaliacao['mae_geral']:.2f}\n")
        f.write(f"- MAPE (Erro Percentual Absoluto Médio): {resultados_avaliacao['mape_geral']:.2f}%\n")
        f.write(f"- R² (Coeficiente de Determinação): {resultados_avaliacao['r2_geral']:.4f}\n\n")
        
        f.write("### Métricas por Horizonte de Previsão\n\n")
        f.write("O modelo apresenta desempenho variado dependendo do horizonte de previsão. ")
        f.write("Como esperado, a precisão diminui à medida que o horizonte de previsão aumenta.\n\n")
        
        f.write("#### Horizontes Selecionados:\n\n")
        
        # Selecionar alguns horizontes para destacar
        horizontes = [0, 6, 13, 29]  # 1 dia, 1 semana, 2 semanas, 1 mês
        
        for h in horizontes:
            metrica = resultados_avaliacao['metricas'].iloc[h]
            f.write(f"**Horizonte {h+1} dias:**\n")
            f.write(f"- RMSE: ${metrica['RMSE']:.2f}\n")
            f.write(f"- MAE: ${metrica['MAE']:.2f}\n")
            f.write(f"- MAPE: {metrica['MAPE']:.2f}%\n")
            f.write(f"- R²: {metrica['R2']:.4f}\n\n")
        
        f.write("## Limitações e Considerações\n\n")
        f.write("### Limitações do Modelo\n\n")
        f.write("- O modelo não incorpora diretamente eventos geopolíticos ou econômicos futuros\n")
        f.write("- A precisão diminui significativamente para horizontes de previsão mais longos\n")
        f.write("- O modelo pode não capturar adequadamente choques de mercado extremos ou eventos inesperados\n\n")
        
        f.write("### Possíveis Melhorias\n\n")
        f.write("- Incorporar variáveis exógenas como preços de outras commodities, taxas de câmbio, etc.\n")
        f.write("- Experimentar arquiteturas mais complexas como modelos híbridos LSTM-CNN\n")
        f.write("- Implementar técnicas de ensemble combinando múltiplos modelos\n")
        f.write("- Adicionar mecanismos de atenção para melhorar a captura de dependências de longo prazo\n\n")
        
        f.write("## Uso do Modelo\n\n")
        f.write("### Integração com o Dashboard\n\n")
        f.write("O modelo está integrado ao dashboard interativo desenvolvido em Streamlit, permitindo:\n\n")
        f.write("- Visualização das previsões para os próximos dias\n")
        f.write("- Comparação com valores históricos\n")
        f.write("- Análise de cenários\n\n")
        
        f.write("### Atualização do Modelo\n\n")
        f.write("Recomenda-se atualizar o modelo periodicamente (mensalmente) para incorporar novos dados e manter a precisão das previsões.\n\n")
        
        f.write("## Conclusão\n\n")
        f.write("O modelo LSTM desenvolvido oferece uma ferramenta valiosa para prever os preços do petróleo Brent no curto prazo. ")
        f.write("Embora nenhum modelo possa prever com precisão perfeita os movimentos futuros dos preços, ")
        f.write("especialmente em um mercado tão volátil e influenciado por fatores geopolíticos como o do petróleo, ")
        f.write("as previsões geradas podem servir como um guia útil para tomada de decisões estratégicas e planejamento.\n\n")
        
        f.write("A combinação deste modelo com a análise dos insights históricos identificados no dashboard ")
        f.write("proporciona uma visão mais completa e fundamentada do mercado de petróleo Brent.")
    
    print(f"Documentação do modelo criada em: {doc_path}")

def main():
    # Carregar os dados
    df = carregar_dados()
    
    if df is None:
        print("Não foi possível carregar os dados. Encerrando.")
        return
    
    # Definir parâmetros do modelo
    janela = 30  # Usar 30 dias anteriores para previsão
    previsao_dias = 30  # Prever 30 dias à frente
    
    # Preparar os dados para o modelo
    dados_preparados = preparar_dados_para_modelo(df, janela, previsao_dias)
    
    # Criar o modelo
    modelo = criar_modelo_lstm(
        input_shape=(dados_preparados['X_train'].shape[1], dados_preparados['X_train'].shape[2]),
        previsao_dias=previsao_dias
    )
    
    # Treinar o modelo
    historico = treinar_modelo(modelo, dados_preparados)
    
    # Avaliar o modelo
    resultados_avaliacao = avaliar_modelo(modelo, dados_preparados)
    
    # Fazer previsões futuras
    previsoes_futuras = fazer_previsao_futura(modelo, dados_preparados)
    
    # Salvar o modelo e artefatos
    salvar_modelo_e_artefatos(modelo, dados_preparados, resultados_avaliacao)
    
    # Criar documentação do modelo
    criar_documentacao_modelo(dados_preparados, resultados_avaliacao)
    
    print("\nProcesso de criação e avaliação do modelo concluído com sucesso!")

if __name__ == "__main__":
    main()
