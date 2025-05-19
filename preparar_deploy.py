import os
import shutil

# Criar diretório para o deploy
deploy_dir = "deploy_petroleo"
os.makedirs(deploy_dir, exist_ok=True)

# Copiar arquivos necessários
files_to_copy = [
    "dashboard_com_modelo.py",
    "requirements.txt",
    "README.md"
]

for file in files_to_copy:
    shutil.copy(file, os.path.join(deploy_dir, file))

# Criar diretórios necessários
os.makedirs(os.path.join(deploy_dir, "dados_processados"), exist_ok=True)
os.makedirs(os.path.join(deploy_dir, "modelo"), exist_ok=True)
os.makedirs(os.path.join(deploy_dir, "visualizacoes"), exist_ok=True)
os.makedirs(os.path.join(deploy_dir, "documentacao"), exist_ok=True)

# Copiar dados processados
data_files = [
    "petroleo_brent_processado.csv",
    "eventos_importantes.csv",
    "previsao_futura.csv"
]

for file in data_files:
    src = os.path.join("dados_processados", file)
    dst = os.path.join(deploy_dir, "dados_processados", file)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Arquivo {src} não encontrado")

# Copiar arquivos do modelo
model_files = [
    "modelo_lstm.h5",
    "scaler_X.pkl",
    "scaler_y.pkl",
    "parametros.pkl",
    "metricas_por_horizonte.csv"
]

for file in model_files:
    src = os.path.join("modelo", file)
    dst = os.path.join(deploy_dir, "modelo", file)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Arquivo {src} não encontrado")

# Copiar visualizações
for file in os.listdir("visualizacoes"):
    if file.endswith(".png"):
        src = os.path.join("visualizacoes", file)
        dst = os.path.join(deploy_dir, "visualizacoes", file)
        shutil.copy(src, dst)

# Copiar documentação
doc_files = ["documentacao_modelo.md"]
for file in doc_files:
    src = os.path.join("documentacao", file)
    dst = os.path.join(deploy_dir, "documentacao", file)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Arquivo {src} não encontrado")

# Copiar insights
if os.path.exists("insights.md"):
    shutil.copy("insights.md", os.path.join(deploy_dir, "insights.md"))
else:
    print("Arquivo insights.md não encontrado")

print(f"Estrutura para deploy criada em: {deploy_dir}")
