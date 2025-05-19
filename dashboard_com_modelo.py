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
