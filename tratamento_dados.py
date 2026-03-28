import pandas as pd
import matplotlib.pyplot as plt
import os

# resolve o caminho relativo ao diretório do script
pasta = os.path.dirname(os.path.abspath(__file__))
arquivo_entrada = os.path.join(pasta, 'tech_mental_health_burnout.csv')

# leitura do dataset
df = pd.read_csv(arquivo_entrada)

print(f"Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"Colunas: {list(df.columns)}\n")

# 1 - adicionar coluna de ID como primeira coluna
df.insert(0, 'id', range(1, len(df) + 1))

# 2 - converter burnout_level: Low -> 0, Moderate -> 1
mapa_burnout = {'Low': 0, 'Moderate': 1}
df['burnout_level'] = df['burnout_level'].map(mapa_burnout).astype('Int64')

# 3 - remover 50% das linhas onde burnout_level == 0
zeros = df[df['burnout_level'] == 0]
remover = zeros.sample(frac=0.60, random_state=42)
df = df.drop(remover.index)
print(f"Linhas com burnout_level == 0 removidas (50%): {len(remover)}")


# 4 - remover linhas onde burnout_score == 1
antes = len(df)
df = df[df['burnout_score'] != 1]
depois = len(df)
print(f"Linhas removidas (burnout_score == 1): {antes - depois}")
print(f"Dataset final: {depois} linhas, {df.shape[1]} colunas\n")


# resetar index e atualizar IDs
df = df.reset_index(drop=True)
df['id'] = range(1, len(df) + 1)

arquivo_saida = os.path.join(pasta, 'tech_mental_health_burnout_tratado.csv')
df.to_csv(arquivo_saida, index=False)
print(f"Arquivo salvo: {arquivo_saida}")
print("\nPrimeiras 5 linhas do dataset tratado:")
print(df.head().to_string())

# =============================================
# GERAÇÃO DE GRÁFICOS
# =============================================

pasta_graficos = os.path.join(pasta, 'graficos')
os.makedirs(pasta_graficos, exist_ok=True)

# colunas para ignorar no gráfico
ignorar = ['id']

# colunas categóricas/binárias (poucos valores únicos)
colunas_categoricas = ['gender', 'job_role', 'company_size', 'work_mode',
                       'has_therapy', 'burnout_level', 'seeks_professional_help']

colunas = [c for c in df.columns if c not in ignorar]

for col in colunas:
    fig, ax = plt.subplots(figsize=(10, 5))

    if col in colunas_categoricas:
        # gráfico de barras para categóricas
        contagem = df[col].value_counts().sort_index()
        contagem.plot(kind='bar', ax=ax, color='#4C72B0', edgecolor='black')
        ax.set_ylabel('Quantidade')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        # histograma para numéricas
        ax.hist(df[col].dropna(), bins=30, color='#4C72B0', edgecolor='black', alpha=0.8)
        ax.set_ylabel('Frequência')

    ax.set_title(f'Distribuição - {col}', fontsize=14, fontweight='bold')
    ax.set_xlabel(col)
    plt.tight_layout()

    caminho_grafico = os.path.join(pasta_graficos, f'{col}.png')
    fig.savefig(caminho_grafico, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo: {caminho_grafico}")

print(f"\nTodos os gráficos foram salvos na pasta: {pasta_graficos}")
