# Documentação do Modelo de Previsão do Preço do Petróleo Brent

## Visão Geral

Este documento descreve o modelo de Machine Learning desenvolvido para prever o preço diário do petróleo Brent.

## Arquitetura do Modelo

O modelo utiliza uma arquitetura de Rede Neural Recorrente (RNN) do tipo LSTM (Long Short-Term Memory), que é especialmente adequada para previsão de séries temporais devido à sua capacidade de capturar dependências de longo prazo nos dados.

### Estrutura da Rede

- Camada LSTM 1: 100 unidades, com retorno de sequências
- Camada Dropout 1: 20% para evitar overfitting
- Camada LSTM 2: 50 unidades
- Camada Dropout 2: 20% para evitar overfitting
- Camada Dense (saída): 30 unidades (uma para cada dia de previsão)

### Parâmetros do Modelo

- Janela de entrada: 30 dias
- Horizonte de previsão: 30 dias
- Features utilizadas: Preco, Variacao, MM7, MM30, DiaSemana_sin, DiaSemana_cos, Mes_sin, Mes_cos, Trimestre_sin, Trimestre_cos, Tendencia
- Função de perda: Erro Quadrático Médio (MSE)
- Otimizador: Adam

## Preparação dos Dados

### Pré-processamento

- Normalização das features usando MinMaxScaler
- Criação de features cíclicas para dia da semana, mês e trimestre
- Adição de feature de tendência
- Criação de sequências de entrada com janela deslizante

### Divisão dos Dados

- Conjunto de treinamento: 9065 amostras
- Conjunto de teste: 2267 amostras

## Desempenho do Modelo

### Métricas Gerais

- RMSE (Erro Quadrático Médio): $6.83
- MAE (Erro Absoluto Médio): $4.79
- MAPE (Erro Percentual Absoluto Médio): 7.83%
- R² (Coeficiente de Determinação): 0.8610

### Métricas por Horizonte de Previsão

O modelo apresenta desempenho variado dependendo do horizonte de previsão. Como esperado, a precisão diminui à medida que o horizonte de previsão aumenta.

#### Horizontes Selecionados:

**Horizonte 1 dias:**
- RMSE: $4.23
- MAE: $3.10
- MAPE: 4.43%
- R²: 0.9469

**Horizonte 7 dias:**
- RMSE: $5.19
- MAE: $3.79
- MAPE: 5.81%
- R²: 0.9199

**Horizonte 14 dias:**
- RMSE: $6.36
- MAE: $4.57
- MAPE: 7.43%
- R²: 0.8797

**Horizonte 30 dias:**
- RMSE: $9.21
- MAE: $6.54
- MAPE: 11.40%
- R²: 0.7465

## Uso do Modelo

### Integração com o Dashboard

O modelo está integrado ao dashboard interativo desenvolvido em Streamlit, permitindo:

- Visualização das previsões para os próximos dias
- Comparação com valores históricos
- Análise de cenários

### Atualização do Modelo

Recomenda-se atualizar o modelo periodicamente (mensalmente) para incorporar novos dados e manter a precisão das previsões.

## Conclusão

O modelo LSTM desenvolvido oferece uma ferramenta valiosa para prever os preços do petróleo Brent no curto prazo. Embora nenhum modelo possa prever com precisão perfeita os movimentos futuros dos preços, especialmente em um mercado tão volátil e influenciado por fatores geopolíticos como o do petróleo, as previsões geradas podem servir como um guia útil para tomada de decisões estratégicas e planejamento.

A combinação deste modelo com a análise dos insights históricos identificados no dashboard proporciona uma visão mais completa e fundamentada do mercado de petróleo Brent.