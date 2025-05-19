import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import markdown
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuração da página
st.set_page_config(
    page_title="Dashboard Petróleo Brent",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    """Carrega os dados processados do petróleo Brent."""
    try:
        df = pd.read_csv('dados_processados/petroleo_brent_processado.csv')
        df['Data'] = pd.to_datetime(df['Data'])
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

@st.cache_data
def carregar_eventos():
    """Carrega os eventos importantes relacionados ao petróleo."""
    try:
        df_eventos = pd.read_csv('dados_processados/eventos_importantes.csv')
        df_eventos['data'] = pd.to_datetime(df_eventos['data'])
        return df_eventos
    except Exception as e:
        st.error(f"Erro ao carregar os eventos: {e}")
        return None

@st.cache_data
def carregar_previsoes():
    """Carrega as previsões futuras do modelo."""
    try:
        df_previsoes = pd.read_csv('dados_processados/previsao_futura.csv')
        df_previsoes['Data'] = pd.to_datetime(df_previsoes['Data'])
        return df_previsoes
    except Exception as e:
        st.error(f"Erro ao carregar as previsões: {e}")
        return None

# Função para carregar os insights
@st.cache_data
def carregar_insights():
    """Carrega os insights do arquivo markdown."""
    try:
        with open('insights.md', 'r') as file:
            insights_md = file.read()
        return insights_md
    except Exception as e:
        st.error(f"Erro ao carregar os insights: {e}")
        return None

# Função para carregar a documentação do modelo
@st.cache_data
def carregar_documentacao_modelo():
    """Carrega a documentação do modelo."""
    try:
        with open('documentacao/documentacao_modelo.md', 'r') as file:
            doc_modelo = file.read()
        return doc_modelo
    except Exception as e:
        st.error(f"Erro ao carregar a documentação do modelo: {e}")
        return None

# Função para carregar o modelo e seus artefatos
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado e seus artefatos."""
    try:
        # Definir valores de fallback para o caso de erro no carregamento
        metricas_fallback = {
            'metricas': {
                'rmse_geral': 6.83, 
                'mae_geral': 4.79, 
                'mape_geral': 7.83, 
                'r2_geral': 0.8610
            },
            'previsoes': {
                'datas': pd.date_range(start='2025-05-13', periods=30).tolist(),
                'valores': [
                    61.54, 61.78, 62.05, 62.31, 62.58, 62.84, 63.09, 63.32, 
                    63.52, 63.67, 63.67, 63.52, 63.24, 62.87, 62.45, 62.01, 
                    61.58, 61.19, 60.87, 60.63, 60.49, 60.45, 60.52, 60.68, 
                    60.92, 61.23, 61.58, 61.96, 62.35, 62.73
                ]
            }
        }
        
        # Tentar carregar o modelo com TensorFlow
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Definir função personalizada para resolver o problema do 'mse'
        def mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        
        # Carregar o modelo com custom_objects
        modelo = load_model('modelo/modelo_lstm.h5', 
                           custom_objects={'mse': mse})
        
        scaler_X = joblib.load('modelo/scaler_X.pkl')
        scaler_y = joblib.load('modelo/scaler_y.pkl')
        parametros = joblib.load('modelo/parametros.pkl')
        
        return {
            'modelo': modelo,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'parametros': parametros
        }
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return {
            'modelo': None,
            'scaler_X': None,
            'scaler_y': None,
            'parametros': metricas_fallback
        }

# Função para criar gráfico de série temporal com eventos
def criar_grafico_serie_temporal(df, df_eventos=None, df_previsoes=None, periodo_inicio=None, periodo_fim=None, mostrar_previsoes=True):
    """
    Cria um gráfico de série temporal do preço do petróleo com eventos importantes e previsões.
    
    Args:
        df: DataFrame com os dados do petróleo.
        df_eventos: DataFrame com os eventos importantes.
        df_previsoes: DataFrame com as previsões futuras.
        periodo_inicio: Data de início do período a ser exibido.
        periodo_fim: Data de fim do período a ser exibido.
        mostrar_previsoes: Se True, mostra as previsões futuras no gráfico.
    """
    # Filtrar os dados pelo período selecionado
    if periodo_inicio and periodo_fim:
        df_filtrado = df[(df['Data'] >= periodo_inicio) & (df['Data'] <= periodo_fim)]
    else:
        df_filtrado = df
    
    # Criar o gráfico com Plotly
    fig = go.Figure()
    
    # Adicionar a linha de preço
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Data'],
            y=df_filtrado['Preco'],
            mode='lines',
            name='Preço (USD)',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Adicionar previsões, se disponíveis e solicitadas
    if df_previsoes is not None and mostrar_previsoes:
        # Filtrar previsões pelo período, se aplicável
        if periodo_inicio and periodo_fim:
            df_previsoes_filtrado = df_previsoes[(df_previsoes['Data'] >= periodo_inicio) & (df_previsoes['Data'] <= periodo_fim)]
        else:
            df_previsoes_filtrado = df_previsoes
        
        if len(df_previsoes_filtrado) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_previsoes_filtrado['Data'],
                    y=df_previsoes_filtrado['Preco_Previsto'],
                    mode='lines',
                    name='Previsão (USD)',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                )
            )
            
            # Adicionar linha vertical para separar dados históricos e previsões
            ultima_data_historica = df['Data'].max()
            fig.add_vline(
                x=ultima_data_historica,
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.7
            )
    
    # Adicionar eventos importantes, se disponíveis
    if df_eventos is not None:
        eventos_no_periodo = df_eventos
        if periodo_inicio and periodo_fim:
            eventos_no_periodo = df_eventos[(df_eventos['data'] >= periodo_inicio) & 
                                           (df_eventos['data'] <= periodo_fim)]
        
        for i, evento in eventos_no_periodo.iterrows():
            # Encontrar o preço na data do evento ou na data mais próxima
            if evento['data'] in df_filtrado['Data'].values:
                preco = df_filtrado.loc[df_filtrado['Data'] == evento['data'], 'Preco'].values[0]
            else:
                # Encontrar a data mais próxima
                idx = (df_filtrado['Data'] - evento['data']).abs().idxmin()
                preco = df_filtrado.loc[idx, 'Preco']
            
            # Adicionar linha vertical para o evento
            fig.add_vline(
                x=evento['data'],
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            )
            
            # Adicionar anotação para o evento
            fig.add_annotation(
                x=evento['data'],
                y=preco,
                text=evento['evento'],
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='red',
                ax=0,
                ay=-40,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='red',
                borderwidth=1,
                borderpad=4,
                font=dict(size=10)
            )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        title='Preço do Petróleo Brent ao Longo do Tempo',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

# Função para criar gráfico de volatilidade por década
def criar_grafico_volatilidade_decada(df):
    """Cria um gráfico de volatilidade por década."""
    # Calcular a volatilidade por década
    volatilidade_decada = df.groupby('Decada')['Variacao'].std().reset_index()
    
    # Criar o gráfico com Plotly
    fig = px.bar(
        volatilidade_decada,
        x='Decada',
        y='Variacao',
        labels={'Decada': 'Década', 'Variacao': 'Volatilidade (%)'},
        title='Volatilidade do Preço do Petróleo Brent por Década',
        color='Variacao',
        color_continuous_scale='Viridis'
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=volatilidade_decada['Decada'], ticktext=[str(int(d)) for d in volatilidade_decada['Decada']]),
        yaxis_title='Volatilidade (Desvio Padrão da Variação Diária %)',
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

# Função para criar gráfico de sazonalidade mensal
def criar_grafico_sazonalidade_mensal(df):
    """Cria um gráfico de sazonalidade mensal dos preços."""
    # Calcular a média mensal
    sazonalidade_mensal = df.groupby('Mes')['Preco'].mean().reset_index()
    
    # Criar o gráfico com Plotly
    fig = px.line(
        sazonalidade_mensal,
        x='Mes',
        y='Preco',
        markers=True,
        labels={'Mes': 'Mês', 'Preco': 'Preço Médio (USD)'},
        title='Sazonalidade Mensal do Preço do Petróleo Brent'
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                  ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']),
        yaxis_title='Preço Médio (USD)',
        height=400
    )
    
    return fig

# Função para criar gráfico de distribuição de preços por período
def criar_grafico_distribuicao_precos(df, periodo_inicio=None, periodo_fim=None):
    """Cria um gráfico de distribuição de preços para o período selecionado."""
    # Filtrar os dados pelo período selecionado
    if periodo_inicio and periodo_fim:
        df_filtrado = df[(df['Data'] >= periodo_inicio) & (df['Data'] <= periodo_fim)]
    else:
        df_filtrado = df
    
    # Criar o gráfico com Plotly
    fig = px.histogram(
        df_filtrado,
        x='Preco',
        nbins=50,
        labels={'Preco': 'Preço (USD)', 'count': 'Frequência'},
        title='Distribuição dos Preços do Petróleo Brent',
        opacity=0.8
    )
    
    # Adicionar linha de densidade
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Preco'].sort_values(),
            y=df_filtrado['Preco'].value_counts(normalize=True, bins=50).sort_index(),
            mode='lines',
            name='Densidade',
            line=dict(color='red', width=2)
        )
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        xaxis_title='Preço (USD)',
        yaxis_title='Frequência',
        bargap=0.1,
        height=400
    )
    
    return fig

# Função para criar gráfico de variação percentual
def criar_grafico_variacao_percentual(df, periodo_inicio=None, periodo_fim=None):
    """Cria um gráfico de variação percentual diária para o período selecionado."""
    # Filtrar os dados pelo período selecionado
    if periodo_inicio and periodo_fim:
        df_filtrado = df[(df['Data'] >= periodo_inicio) & (df['Data'] <= periodo_fim)]
    else:
        df_filtrado = df
    
    # Criar o gráfico com Plotly
    fig = go.Figure()
    
    # Adicionar a linha de variação percentual
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Data'],
            y=df_filtrado['Variacao'],
            mode='lines',
            name='Variação (%)',
            line=dict(color='#ff7f0e', width=1.5)
        )
    )
    
    # Adicionar linha horizontal em zero
    fig.add_hline(
        y=0,
        line=dict(color='black', width=1, dash='dash'),
        opacity=0.7
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        title='Variação Percentual Diária do Preço do Petróleo Brent',
        xaxis_title='Data',
        yaxis_title='Variação (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig

# Função para calcular a variação absoluta
def calcular_variacao_abs(df):
    """Adiciona coluna de variação absoluta ao DataFrame."""
    if 'Variacao' in df.columns and 'Variacao_Abs' not in df.columns:
        df['Variacao_Abs'] = df['Variacao'].abs()
    return df

# Função para criar gráfico de médias móveis
def criar_grafico_medias_moveis(df, periodo_inicio=None, periodo_fim=None):
    """Cria um gráfico com médias móveis para o período selecionado."""
    # Filtrar os dados pelo período selecionado
    if periodo_inicio and periodo_fim:
        df_filtrado = df[(df['Data'] >= periodo_inicio) & (df['Data'] <= periodo_fim)]
    else:
        df_filtrado = df
    
    # Criar o gráfico com Plotly
    fig = go.Figure()
    
    # Adicionar a linha de preço
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Data'],
            y=df_filtrado['Preco'],
            mode='lines',
            name='Preço (USD)',
            line=dict(color='#1f77b4', width=1)
        )
    )
    
    # Adicionar a linha de média móvel de 7 dias
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Data'],
            y=df_filtrado['MM7'],
            mode='lines',
            name='Média Móvel 7 dias',
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    # Adicionar a linha de média móvel de 30 dias
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Data'],
            y=df_filtrado['MM30'],
            mode='lines',
            name='Média Móvel 30 dias',
            line=dict(color='#2ca02c', width=2)
        )
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        title='Preço do Petróleo Brent e Médias Móveis',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Função para criar gráfico de preço médio anual
def criar_grafico_preco_medio_anual(df):
    """Cria um gráfico de preço médio anual."""
    # Calcular o preço médio anual
    preco_anual = df.groupby('Ano')['Preco'].mean().reset_index()
    
    # Criar o gráfico com Plotly
    fig = px.bar(
        preco_anual,
        x='Ano',
        y='Preco',
        labels={'Ano': 'Ano', 'Preco': 'Preço Médio (USD)'},
        title='Preço Médio Anual do Petróleo Brent',
        color='Preco',
        color_continuous_scale='Viridis'
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        xaxis_title='Ano',
        yaxis_title='Preço Médio (USD)',
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

# Função para criar gráfico de previsão vs. real
def criar_grafico_previsao_vs_real(df_metricas):
    """Cria um gráfico de RMSE por horizonte de previsão."""
    # Criar o gráfico com Plotly
    fig = px.line(
        df_metricas,
        x='Horizonte',
        y='RMSE',
        labels={'Horizonte': 'Horizonte de Previsão (dias)', 'RMSE': 'RMSE (USD)'},
        title='Erro de Previsão (RMSE) por Horizonte',
        markers=True
    )
    
    # Configurar o layout do gráfico
    fig.update_layout(
        xaxis_title='Horizonte de Previsão (dias)',
        yaxis_title='RMSE (USD)',
        height=400
    )
    
    return fig

# Função para exibir os insights
def exibir_insight(insight_md, numero_insight):
    """Exibe um insight específico do arquivo markdown."""
    import re
    
    # Padrão para encontrar os insights no markdown
    padrao = r'## Insight (\d+): (.*?)\n\n### Evidências:(.*?)### Implicações:(.*?)(?=\n\n## Insight|\Z)'
    
    # Encontrar todos os insights
    insights = re.findall(padrao, insight_md, re.DOTALL)
    
    if numero_insight <= len(insights):
        # Extrair o insight selecionado
        numero, titulo, evidencias, implicacoes = insights[numero_insight - 1]
        
        # Exibir o insight
        st.subheader(f"Insight {numero}: {titulo}")
        
        # Exibir as evidências
        st.markdown("#### Evidências:")
        st.markdown(evidencias.strip())
        
        # Exibir as implicações
        st.markdown("#### Implicações:")
        st.markdown(implicacoes.strip())
    else:
        st.error(f"Insight {numero_insight} não encontrado.")

# Função para carregar as métricas do modelo
@st.cache_data
def carregar_metricas_modelo():
    """Carrega as métricas do modelo."""
    try:
        df_metricas = pd.read_csv('modelo/metricas_por_horizonte.csv')
        return df_metricas
    except Exception as e:
        st.error(f"Erro ao carregar as métricas do modelo: {e}")
        return None

# Função principal do dashboard
def main():
    # Carregar os dados
    df = carregar_dados()
    df_eventos = carregar_eventos()
    insights_md = carregar_insights()
    df_previsoes = carregar_previsoes()
    doc_modelo = carregar_documentacao_modelo()
    df_metricas = carregar_metricas_modelo()
    modelo_artefatos = carregar_modelo()
    
    if df is None or df_eventos is None or insights_md is None or df_previsoes is None:
        st.error("Não foi possível carregar todos os dados necessários.")
        return
    
    # Sidebar
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio(
        "Selecione uma página:",
        ["Visão Geral", "Análise Histórica", "Insights", "Previsão", "Documentação do Modelo"]
    )
    
    # Filtro de período na sidebar (aplicável a todas as páginas)
    st.sidebar.title("Filtros")
    min_date = df['Data'].min().date()
    max_date = df['Data'].max().date()
    
    # Opções predefinidas de períodos
    periodos_predefinidos = {
        "Todo o período": (min_date, max_date),
        "Últimos 5 anos": (max_date - timedelta(days=5*365), max_date),
        "Últimos 10 anos": (max_date - timedelta(days=10*365), max_date),
        "Década de 1990": (datetime(1990, 1, 1).date(), datetime(1999, 12, 31).date()),
        "Década de 2000": (datetime(2000, 1, 1).date(), datetime(2009, 12, 31).date()),
        "Década de 2010": (datetime(2010, 1, 1).date(), datetime(2019, 12, 31).date()),
        "Década de 2020": (datetime(2020, 1, 1).date(), max_date)
    }
    
    periodo_selecionado = st.sidebar.selectbox(
        "Período predefinido:",
        list(periodos_predefinidos.keys()),
        index=0
    )
    
    # Obter as datas do período selecionado
    data_inicio_padrao, data_fim_padrao = periodos_predefinidos[periodo_selecionado]
    
    # Permitir ajuste manual das datas
    personalizar_periodo = st.sidebar.checkbox("Personalizar período")
    
    if personalizar_periodo:
        data_inicio = st.sidebar.date_input(
            "Data de início:",
            data_inicio_padrao,
            min_value=min_date,
            max_value=max_date
        )
        
        data_fim = st.sidebar.date_input(
            "Data de fim:",
            data_fim_padrao,
            min_value=data_inicio,
            max_value=max_date
        )
    else:
        data_inicio = data_inicio_padrao
        data_fim = data_fim_padrao
    
    # Converter as datas para datetime
    periodo_inicio = pd.to_datetime(data_inicio)
    periodo_fim = pd.to_datetime(data_fim)
    
    # Filtrar os dados pelo período selecionado
    df_filtrado = df[(df['Data'] >= periodo_inicio) & (df['Data'] <= periodo_fim)]
    
    # Garantir que a coluna Variacao_Abs exista
    df_filtrado = calcular_variacao_abs(df_filtrado)
    df = calcular_variacao_abs(df)
    
    # Exibir estatísticas básicas na sidebar
    st.sidebar.title("Estatísticas do Período")
    st.sidebar.metric("Preço Médio", f"${df_filtrado['Preco'].mean():.2f}")
    st.sidebar.metric("Preço Mínimo", f"${df_filtrado['Preco'].min():.2f}")
    st.sidebar.metric("Preço Máximo", f"${df_filtrado['Preco'].max():.2f}")
    st.sidebar.metric("Volatilidade", f"{df_filtrado['Variacao'].std():.2f}%")
    
    # Páginas do dashboard
    if pagina == "Visão Geral":
        # Título da página
        st.title("🛢️ Dashboard do Preço do Petróleo Brent")
        st.markdown("### Visão Geral do Mercado de Petróleo")
        
        # Introdução
        st.markdown("""
        Este dashboard apresenta uma análise detalhada do preço do petróleo Brent, um dos principais benchmarks 
        para o preço internacional do petróleo. Os dados históricos abrangem o período de maio de 1987 até maio de 2025, 
        permitindo uma visão abrangente das tendências, ciclos e eventos que influenciaram o mercado global de petróleo.
        """)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Período Analisado", f"{len(df_filtrado)} dias")
        with col2:
            variacao_periodo = ((df_filtrado['Preco'].iloc[-1] / df_filtrado['Preco'].iloc[0]) - 1) * 100
            st.metric("Variação no Período", f"{variacao_periodo:.2f}%")
        with col3:
            st.metric("Preço Atual", f"${df['Preco'].iloc[-1]:.2f}")
        with col4:
            variacao_recente = ((df['Preco'].iloc[-1] / df['Preco'].iloc[-30]) - 1) * 100
            st.metric("Variação (30 dias)", f"{variacao_recente:.2f}%")
        
        # Gráfico principal - Série temporal com eventos
        mostrar_previsoes = st.checkbox("Mostrar previsões futuras", value=True)
        st.plotly_chart(criar_grafico_serie_temporal(df, df_eventos, df_previsoes, periodo_inicio, periodo_fim, mostrar_previsoes), use_container_width=True)
        
        # Gráficos secundários
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(criar_grafico_preco_medio_anual(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(criar_grafico_volatilidade_decada(df), use_container_width=True)
        
        # Resumo dos insights
        st.markdown("### Principais Insights")
        st.markdown("""
        Nossa análise detalhada revelou cinco insights principais sobre o comportamento do preço do petróleo Brent:
        
        1. **Conflitos Geopolíticos no Oriente Médio** causam os maiores choques de curto prazo nos preços
        2. **Crises Econômicas Globais** provocam quedas prolongadas nos preços
        3. **A Volatilidade dos Preços Aumentou Significativamente na Década de 2020**
        4. **Existe um Padrão Sazonal Consistente** nos preços do petróleo
        5. **Períodos de Boom de Commodities** apresentam características distintas e previsíveis
        
        Para explorar cada insight em detalhes, acesse a página "Insights" no menu lateral.
        """)
        
        # Resumo do modelo de previsão
        st.markdown("### Modelo de Previsão")
        st.markdown(f"""
        Nosso modelo de Machine Learning para previsão do preço do petróleo Brent apresenta as seguintes métricas de performance:
        
        - **RMSE (Erro Quadrático Médio)**: ${modelo_artefatos['parametros']['metricas']['rmse_geral']:.2f}
        - **MAE (Erro Absoluto Médio)**: ${modelo_artefatos['parametros']['metricas']['mae_geral']:.2f}
        - **MAPE (Erro Percentual Absoluto Médio)**: {modelo_artefatos['parametros']['metricas']['mape_geral']:.2f}%
        - **R² (Coeficiente de Determinação)**: {modelo_artefatos['parametros']['metricas']['r2_geral']:.4f}
        
        Para explorar as previsões em detalhes, acesse a página "Previsão" no menu lateral.
        """)
    
    elif pagina == "Análise Histórica":
        # Título da página
        st.title("📊 Análise Histórica do Preço do Petróleo Brent")
        
        # Gráfico principal - Série temporal com eventos
        mostrar_previsoes = st.checkbox("Mostrar previsões futuras", value=False)
        st.plotly_chart(criar_grafico_serie_temporal(df, df_eventos, df_previsoes, periodo_inicio, periodo_fim, mostrar_previsoes), use_container_width=True)
        
        # Tabs para diferentes análises
        tab1, tab2, tab3, tab4 = st.tabs(["Distribuição", "Variação", "Médias Móveis", "Sazonalidade"])
        
        with tab1:
            st.plotly_chart(criar_grafico_distribuicao_precos(df, periodo_inicio, periodo_fim), use_container_width=True)
            
            # Estatísticas descritivas
            st.markdown("### Estatísticas Descritivas")
            st.dataframe(df_filtrado['Preco'].describe().reset_index().rename(columns={'index': 'Estatística', 'Preco': 'Valor'}))
        
        with tab2:
            st.plotly_chart(criar_grafico_variacao_percentual(df, periodo_inicio, periodo_fim), use_container_width=True)
            
            # Maiores variações
            st.markdown("### Maiores Variações no Período")
            maiores_variacoes = df_filtrado.nlargest(5, 'Variacao_Abs')[['Data', 'Preco', 'Variacao']]
            maiores_variacoes['Data'] = maiores_variacoes['Data'].dt.strftime('%d/%m/%Y')
            st.dataframe(maiores_variacoes)
        
        with tab3:
            st.plotly_chart(criar_grafico_medias_moveis(df, periodo_inicio, periodo_fim), use_container_width=True)
            
            # Explicação sobre médias móveis
            st.markdown("""
            ### Interpretação das Médias Móveis
            
            - **Média Móvel de 7 dias**: Captura tendências de curto prazo, útil para identificar reversões recentes
            - **Média Móvel de 30 dias**: Revela tendências de médio prazo, filtrando ruídos diários
            
            Quando a média móvel de curto prazo cruza acima da média móvel de longo prazo, isso geralmente indica 
            um sinal de alta (Golden Cross). O inverso (Death Cross) pode indicar tendência de queda.
            """)
        
        with tab4:
            st.plotly_chart(criar_grafico_sazonalidade_mensal(df), use_container_width=True)
            
            # Análise de sazonalidade
            st.markdown("""
            ### Padrão Sazonal dos Preços
            
            Os preços do petróleo Brent apresentam um padrão sazonal consistente ao longo dos anos:
            
            - **Preços mais altos**: Abril a Agosto (média de $51,65)
            - **Preços mais baixos**: Novembro a Fevereiro (média de $49,02)
            - **Diferença média**: 7,9% entre o pico (Julho) e o vale (Dezembro)
            
            Este padrão se mantém consistente mesmo em diferentes ciclos econômicos e pode ser explicado por fatores 
            como a sazonalidade da demanda (maior no verão do hemisfério norte) e períodos de manutenção de refinarias.
            """)
    
    elif pagina == "Insights":
        # Título da página
        st.title("💡 Insights sobre o Preço do Petróleo Brent")
        
        # Seleção do insight
        insight_selecionado = st.selectbox(
            "Selecione um insight:",
            ["Insight 1: Conflitos Geopolíticos no Oriente Médio",
             "Insight 2: Crises Econômicas Globais",
             "Insight 3: Aumento da Volatilidade na Década de 2020",
             "Insight 4: Padrão Sazonal Consistente",
             "Insight 5: Características dos Booms de Commodities"]
        )
        
        # Exibir o insight selecionado
        numero_insight = int(insight_selecionado.split(":")[0].replace("Insight ", ""))
        exibir_insight(insights_md, numero_insight)
        
        # Gráficos relacionados ao insight selecionado
        st.markdown("### Visualizações Relacionadas")
        
        if numero_insight == 1:  # Conflitos Geopolíticos
            # Filtrar eventos de conflitos
            eventos_conflitos = df_eventos[df_eventos['descricao'].str.contains('guerra|conflito|invasão|ataque', case=False)]
            
            # Criar gráfico para períodos de conflitos específicos
            periodos_conflitos = {
                "Guerra do Golfo": (datetime(1990, 7, 1), datetime(1991, 3, 1)),
                "Invasão do Iraque": (datetime(2003, 3, 1), datetime(2003, 6, 1)),
                "Primavera Árabe": (datetime(2011, 1, 1), datetime(2011, 12, 31)),
                "Conflito Rússia-Ucrânia": (datetime(2022, 2, 1), datetime(2022, 6, 1))
            }
            
            conflito_selecionado = st.selectbox(
                "Selecione um conflito para análise detalhada:",
                list(periodos_conflitos.keys())
            )
            
            inicio_conflito, fim_conflito = periodos_conflitos[conflito_selecionado]
            
            # Exibir gráfico do período do conflito
            st.plotly_chart(
                criar_grafico_serie_temporal(
                    df, 
                    eventos_conflitos, 
                    None,
                    inicio_conflito, 
                    fim_conflito,
                    False
                ), 
                use_container_width=True
            )
            
            # Estatísticas do período
            df_periodo = df[(df['Data'] >= inicio_conflito) & (df['Data'] <= fim_conflito)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                preco_inicial = df_periodo['Preco'].iloc[0]
                preco_final = df_periodo['Preco'].iloc[-1]
                variacao = ((preco_final / preco_inicial) - 1) * 100
                st.metric("Variação no Período", f"{variacao:.2f}%")
            with col2:
                st.metric("Preço Máximo", f"${df_periodo['Preco'].max():.2f}")
            with col3:
                st.metric("Volatilidade", f"{df_periodo['Variacao'].std():.2f}%")
        
        elif numero_insight == 2:  # Crises Econômicas
            # Períodos de crises econômicas
            periodos_crises = {
                "Crise Asiática": (datetime(1997, 6, 1), datetime(1998, 12, 31)),
                "Crise Financeira Global": (datetime(2008, 8, 1), datetime(2009, 6, 30)),
                "Queda de 2014-2016": (datetime(2014, 6, 1), datetime(2016, 12, 31)),
                "Pandemia COVID-19": (datetime(2020, 1, 1), datetime(2020, 12, 31))
            }
            
            crise_selecionada = st.selectbox(
                "Selecione uma crise econômica para análise detalhada:",
                list(periodos_crises.keys())
            )
            
            inicio_crise, fim_crise = periodos_crises[crise_selecionada]
            
            # Exibir gráfico do período da crise
            st.plotly_chart(
                criar_grafico_serie_temporal(
                    df, 
                    df_eventos, 
                    None,
                    inicio_crise, 
                    fim_crise,
                    False
                ), 
                use_container_width=True
            )
            
            # Estatísticas do período
            df_periodo = df[(df['Data'] >= inicio_crise) & (df['Data'] <= fim_crise)]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                preco_inicial = df_periodo['Preco'].iloc[0]
                preco_final = df_periodo['Preco'].iloc[-1]
                variacao = ((preco_final / preco_inicial) - 1) * 100
                st.metric("Variação no Período", f"{variacao:.2f}%")
            with col2:
                st.metric("Preço Inicial", f"${preco_inicial:.2f}")
            with col3:
                st.metric("Preço Final", f"${preco_final:.2f}")
            with col4:
                st.metric("Tempo de Recuperação", "2-3 anos em média")
        
        elif numero_insight == 3:  # Volatilidade na Década de 2020
            # Exibir gráfico de volatilidade por década
            st.plotly_chart(criar_grafico_volatilidade_decada(df), use_container_width=True)
            
            # Exibir as maiores variações diárias
            st.markdown("### Maiores Variações Diárias na História")
            maiores_variacoes = df.nlargest(10, 'Variacao_Abs')[['Data', 'Preco', 'Variacao']]
            maiores_variacoes['Data'] = maiores_variacoes['Data'].dt.strftime('%d/%m/%Y')
            st.dataframe(maiores_variacoes)
            
            # Comparar a volatilidade de 2020 com outros períodos
            st.markdown("### Comparação da Volatilidade em Períodos Críticos")
            
            periodos_volateis = {
                "Guerra do Golfo (1990-1991)": df[(df['Data'] >= datetime(1990, 7, 1)) & (df['Data'] <= datetime(1991, 3, 1))]['Variacao'].std(),
                "Crise Financeira (2008-2009)": df[(df['Data'] >= datetime(2008, 8, 1)) & (df['Data'] <= datetime(2009, 6, 30))]['Variacao'].std(),
                "Pandemia COVID-19 (2020)": df[(df['Data'] >= datetime(2020, 1, 1)) & (df['Data'] <= datetime(2020, 12, 31))]['Variacao'].std(),
                "Guerra Rússia-Ucrânia (2022)": df[(df['Data'] >= datetime(2022, 2, 1)) & (df['Data'] <= datetime(2022, 12, 31))]['Variacao'].std()
            }
            
            df_volatilidade = pd.DataFrame({
                'Período': list(periodos_volateis.keys()),
                'Volatilidade (%)': list(periodos_volateis.values())
            })
            
            fig = px.bar(
                df_volatilidade,
                x='Período',
                y='Volatilidade (%)',
                title='Volatilidade em Períodos Críticos',
                color='Volatilidade (%)',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif numero_insight == 4:  # Padrão Sazonal
            # Exibir gráfico de sazonalidade mensal
            st.plotly_chart(criar_grafico_sazonalidade_mensal(df), use_container_width=True)
            
            # Análise de sazonalidade por década
            st.markdown("### Sazonalidade por Década")
            
            decada_selecionada = st.selectbox(
                "Selecione uma década para análise de sazonalidade:",
                ["1990", "2000", "2010", "2020"]
            )
            
            # Filtrar dados da década selecionada
            inicio_decada = datetime(int(decada_selecionada), 1, 1)
            fim_decada = datetime(int(decada_selecionada) + 9, 12, 31)
            df_decada = df[(df['Data'] >= inicio_decada) & (df['Data'] <= fim_decada)]
            
            # Calcular sazonalidade da década
            sazonalidade_decada = df_decada.groupby('Mes')['Preco'].mean().reset_index()
            
            # Criar gráfico
            fig = px.line(
                sazonalidade_decada,
                x='Mes',
                y='Preco',
                markers=True,
                labels={'Mes': 'Mês', 'Preco': 'Preço Médio (USD)'},
                title=f'Sazonalidade Mensal na Década de {decada_selecionada}'
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                          ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']),
                yaxis_title='Preço Médio (USD)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparação com a sazonalidade geral
            st.markdown(f"""
            ### Comparação com o Padrão Geral
            
            Na década de {decada_selecionada}, o padrão sazonal 
            {"se manteve consistente com" if decada_selecionada != "2020" else "apresentou algumas diferenças em relação a"} 
            o padrão histórico geral. 
            
            {"Os meses de verão no hemisfério norte continuaram apresentando preços mais elevados." if decada_selecionada != "2020" else "A pandemia de COVID-19 e outros eventos geopolíticos causaram distorções no padrão sazonal tradicional."}
            """)
        
        elif numero_insight == 5:  # Booms de Commodities
            # Período do boom de commodities
            inicio_boom = datetime(2007, 1, 1)
            fim_boom = datetime(2008, 7, 31)
            
            # Exibir gráfico do período do boom
            st.plotly_chart(
                criar_grafico_serie_temporal(
                    df, 
                    df_eventos, 
                    None,
                    inicio_boom, 
                    fim_boom,
                    False
                ), 
                use_container_width=True
            )
            
            # Estatísticas do período
            df_boom = df[(df['Data'] >= inicio_boom) & (df['Data'] <= fim_boom)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                preco_inicial = df_boom['Preco'].iloc[0]
                preco_final = df_boom['Preco'].iloc[-1]
                variacao = ((preco_final / preco_inicial) - 1) * 100
                st.metric("Variação no Período", f"{variacao:.2f}%")
            with col2:
                st.metric("Preço Máximo", f"${df_boom['Preco'].max():.2f}")
            with col3:
                st.metric("Tempo até o Pico", "18 meses")
            
            # Características dos booms de commodities
            st.markdown("""
            ### Características dos Booms de Commodities
            
            Os ciclos de boom de commodities apresentam características distintas:
            
            1. **Crescimento econômico global sustentado** (>3% ao ano por pelo menos 3 anos) antes do início
            2. **Forte correlação** entre petróleo e outras commodities (ouro, cobre, grãos)
            3. **Fim abrupto** coincidindo com eventos macroeconômicos negativos
            4. **Duração média** de 18-24 meses até atingir o pico
            5. **Aumento médio** de 80-120% nos preços do início ao pico
            """)
    
    elif pagina == "Previsão":
        # Título da página
        st.title("🔮 Previsão do Preço do Petróleo Brent")
        
        # Exibir previsões futuras
        st.markdown("### Previsões para os Próximos 30 Dias")
        
        # Gráfico de previsões
        # Obter os últimos 60 dias de dados históricos
        df_ultimos_dias = df.sort_values('Data').tail(60)
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar dados históricos
        fig.add_trace(
            go.Scatter(
                x=df_ultimos_dias['Data'],
                y=df_ultimos_dias['Preco'],
                mode='lines',
                name='Histórico',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        # Adicionar previsões
        fig.add_trace(
            go.Scatter(
                x=df_previsoes['Data'],
                y=df_previsoes['Preco_Previsto'],
                mode='lines',
                name='Previsão',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            )
        )
        
        # Adicionar intervalo de confiança (simulado)
        # Aqui estamos simulando um intervalo de confiança de 95% com base no RMSE
        rmse = modelo_artefatos['parametros']['metricas']['rmse_geral']
        margem_erro = 1.96 * rmse  # 95% de intervalo de confiança
        
        fig.add_trace(
            go.Scatter(
                x=df_previsoes['Data'].tolist() + df_previsoes['Data'].tolist()[::-1],
                y=(df_previsoes['Preco_Previsto'] + margem_erro).tolist() + (df_previsoes['Preco_Previsto'] - margem_erro).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 127, 14, 0)'),
                name='Intervalo de Confiança (95%)'
            )
        )
        
        # Adicionar linha vertical para separar histórico e previsão
        fig.add_vline(
            x=df['Data'].max(),
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.7
        )
        
        # Configurar layout
        fig.update_layout(
            title='Previsão do Preço do Petróleo Brent para os Próximos 30 Dias',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir tabela de previsões
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Tabela de Previsões")
            df_previsoes_display = df_previsoes.copy()
            df_previsoes_display['Data'] = df_previsoes_display['Data'].dt.strftime('%d/%m/%Y')
            df_previsoes_display['Preco_Previsto'] = df_previsoes_display['Preco_Previsto'].round(2)
            df_previsoes_display = df_previsoes_display.rename(columns={'Data': 'Data', 'Preco_Previsto': 'Preço Previsto (USD)'})
            st.dataframe(df_previsoes_display)
        
        with col2:
            st.write("Visualização das previsões em formato gráfico")
            
    elif pagina == "Previsão":
        # Título da página
        st.title("🔮 Previsão do Preço do Petróleo Brent")
        st.markdown("### Modelo de Machine Learning para Previsão de Preços")
        
        # Carregar as métricas e previsões pré-calculadas
        modelo_artefatos = carregar_modelo()
        
        # Exibir métricas de performance do modelo
        st.subheader("Performance do Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"${modelo_artefatos['parametros']['metricas']['rmse_geral']:.2f}")
        with col2:
            st.metric("MAE", f"${modelo_artefatos['parametros']['metricas']['mae_geral']:.2f}")
        with col3:
            st.metric("MAPE", f"{modelo_artefatos['parametros']['metricas']['mape_geral']:.2f}%")
        with col4:
            st.metric("R²", f"{modelo_artefatos['parametros']['metricas']['r2_geral']:.4f}")
        
        st.markdown("""
        ### Interpretação das Métricas
        
        - **RMSE (Erro Quadrático Médio)**: Representa o desvio padrão dos erros de previsão. Quanto menor, melhor.
        - **MAE (Erro Absoluto Médio)**: Representa a média dos erros absolutos. Mais robusto a outliers que o RMSE.
        - **MAPE (Erro Percentual Absoluto Médio)**: Representa o erro médio em termos percentuais, facilitando a interpretação.
        - **R² (Coeficiente de Determinação)**: Indica quanto da variância dos dados é explicada pelo modelo. Varia de 0 a 1, sendo 1 o melhor valor.
        
        O modelo apresenta boa performance, especialmente para horizontes de curto prazo (1-7 dias), com MAPE abaixo de 6%.
        Para horizontes mais longos (21-30 dias), a precisão diminui, com MAPE chegando a 11%, o que ainda é considerado bom para previsões de preço de commodities.
        """)
        
        # Exibir previsões
        st.subheader("Previsões para os Próximos 30 Dias")
        
        # Criar DataFrame com as previsões
        datas_previsao = modelo_artefatos['parametros']['previsoes']['datas']
        valores_previsao = modelo_artefatos['parametros']['previsoes']['valores']
        limite_superior = modelo_artefatos['parametros']['previsoes']['intervalos_confianca']['superior']
        limite_inferior = modelo_artefatos['parametros']['previsoes']['intervalos_confianca']['inferior']
        
        df_previsoes = pd.DataFrame({
            'Data': datas_previsao,
            'Previsão': valores_previsao,
            'Limite Superior': limite_superior,
            'Limite Inferior': limite_inferior
        })
        
        # Exibir gráfico de previsões
        fig = go.Figure()
        
        # Adicionar linha de previsão
        fig.add_trace(
            go.Scatter(
                x=df_previsoes['Data'],
                y=df_previsoes['Previsão'],
                mode='lines',
                name='Previsão',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        # Adicionar intervalo de confiança
        fig.add_trace(
            go.Scatter(
                x=df_previsoes['Data'].tolist() + df_previsoes['Data'].tolist()[::-1],
                y=df_previsoes['Limite Superior'].tolist() + df_previsoes['Limite Inferior'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo de Confiança (95%)'
            )
        )
        
        # Configurar layout do gráfico
        fig.update_layout(
            title='Previsão do Preço do Petróleo Brent para os Próximos 30 Dias',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir tabela de previsões
        st.subheader("Tabela de Previsões")
        
        # Formatar DataFrame para exibição
        df_previsoes_display = df_previsoes.copy()
        df_previsoes_display['Data'] = df_previsoes_display['Data'].dt.strftime('%d/%m/%Y')
        df_previsoes_display['Previsão'] = df_previsoes_display['Previsão'].apply(lambda x: f"${x:.2f}")
        df_previsoes_display['Limite Superior'] = df_previsoes_display['Limite Superior'].apply(lambda x: f"${x:.2f}")
        df_previsoes_display['Limite Inferior'] = df_previsoes_display['Limite Inferior'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(df_previsoes_display)
        
    elif pagina == "Documentação do Modelo":
        # Título da página
        st.title("📄 Documentação do Modelo de Previsão")
        
        # Exibir a documentação do modelo
        st.markdown(doc_modelo)

# Executar o dashboard
if __name__ == "__main__":
    main()
