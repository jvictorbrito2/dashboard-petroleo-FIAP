import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime

def scrape_ipea_petroleo_brent():
    """
    Função para extrair os dados históricos do preço do petróleo Brent do site do IPEA.
    
    Returns:
        DataFrame com os dados históricos do preço do petróleo Brent.
    """
    print("Iniciando a extração dos dados do preço do petróleo Brent do IPEA...")
    
    # URL da página com os dados
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    
    # Fazer a requisição HTTP
    response = requests.get(url)
    
    # Verificar se a requisição foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao acessar a página. Código de status: {response.status_code}")
        return None
    
    # Parsear o HTML da página
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontrar a tabela com os dados
    table = soup.find('table', {'class': 'dxgvTable'})
    
    if not table:
        print("Tabela não encontrada na página.")
        return None
    
    # Extrair os dados da tabela
    data = []
    rows = table.find_all('tr')
    
    # Pular a primeira linha (cabeçalho)
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) >= 2:
            date_str = cols[0].text.strip()
            price_str = cols[1].text.strip()
            
            # Converter o preço para float (substituindo vírgula por ponto, se necessário)
            price_str = price_str.replace(',', '.')
            try:
                price = float(price_str)
            except ValueError:
                print(f"Erro ao converter o preço '{price_str}' para float. Pulando esta linha.")
                continue
            
            # Converter a data para o formato datetime
            try:
                date = datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                print(f"Erro ao converter a data '{date_str}' para datetime. Pulando esta linha.")
                continue
            
            data.append({'Data': date, 'Preco': price})
    
    # Criar o DataFrame
    df = pd.DataFrame(data)
    
    # Ordenar o DataFrame por data (do mais antigo para o mais recente)
    df = df.sort_values('Data')
    
    # Resetar o índice
    df = df.reset_index(drop=True)
    
    print(f"Extração concluída. Total de {len(df)} registros obtidos.")
    
    return df

def main():
    # Criar o diretório de dados se não existir
    data_dir = os.path.join(os.getcwd(), 'dados')
    os.makedirs(data_dir, exist_ok=True)
    
    # Extrair os dados
    df = scrape_ipea_petroleo_brent()
    
    if df is not None:
        # Salvar os dados em CSV
        csv_path = os.path.join(data_dir, 'petroleo_brent.csv')
        df.to_csv(csv_path, index=False)
        print(f"Dados salvos em {csv_path}")
        
        # Exibir informações básicas sobre os dados
        print("\nInformações sobre os dados:")
        print(f"Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
        print(f"Preço mínimo: ${df['Preco'].min():.2f}")
        print(f"Preço máximo: ${df['Preco'].max():.2f}")
        print(f"Preço médio: ${df['Preco'].mean():.2f}")
    else:
        print("Não foi possível extrair os dados.")

if __name__ == "__main__":
    main()
