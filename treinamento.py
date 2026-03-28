import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================
# 1. CARREGAMENTO DOS DADOS
# =============================================

pasta = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta, 'tech_mental_health_burnout_tratado.csv')
df = pd.read_csv(arquivo)

print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas\n")

# =============================================
# 2. PRÉ-PROCESSAMENTO
# =============================================

# remover colunas que não devem entrar no modelo
colunas_remover = ['burnout_score']  # id não é feature, burnout_score vazaria informação
df = df.drop(columns=colunas_remover)

# remover linhas com burnout_level vazio (não servem como alvo)
df = df.dropna(subset=['burnout_level'])

# separar features (X) e target (y)
X = df.drop(columns=['burnout_level'])
y = df['burnout_level'].astype(int)

# codificar colunas categóricas (string -> número)
colunas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}

for col in colunas_categoricas:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"Codificado '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nFeatures utilizadas ({len(X.columns)}): {list(X.columns)}")
print(f"Distribuição do target: {dict(y.value_counts().sort_index())}\n")

# =============================================
# 3. DIVISÃO TREINO / TESTE (Hold-out)
# =============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras\n")

# =============================================
# 4. ESCALONAMENTO (Padronização)
# =============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # só transform no teste, sem fit

# =============================================
# 5. DEFINIÇÃO DO MODELO E HIPERPARÂMETROS
# =============================================

mlp = MLPClassifier(max_iter=1000, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

# total de combinações: 4 * 2 * 1 * 3 * 2 = 48 modelos x 5 folds = 240 ajustes
print("Grade de hiperparâmetros:")
for chave, valores in param_grid.items():
    print(f"  {chave}: {valores}")

# =============================================
# 6. GRID SEARCH COM VALIDAÇÃO CRUZADA (K=5)
# =============================================

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,               # 5-Fold Cross Validation
    scoring='accuracy',
    n_jobs=-1,           # usar todos os cores
    verbose=2
)

print("\nIniciando o Grid Search (pode levar alguns minutos)...\n")
grid_search.fit(X_train_scaled, y_train)

# =============================================
# 7. RESULTADOS DA OTIMIZAÇÃO
# =============================================

print("\n" + "=" * 55)
print("  RESULTADOS DA OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("=" * 55)

print(f"\nMelhores hiperparâmetros encontrados:")
for param, valor in grid_search.best_params_.items():
    print(f"  {param}: {valor}")

print(f"\nMelhor acurácia na validação cruzada (5-Fold): {grid_search.best_score_:.4f}")

# =============================================
# 8. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# =============================================

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\n" + "=" * 55)
print("  AVALIAÇÃO NO CONJUNTO DE TESTE (dados nunca vistos)")
print("=" * 55)

print(f"\nAcurácia no teste: {accuracy_score(y_test, y_pred):.4f}")

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':>15} Predito 0   Predito 1")
print(f"  {'Real 0':>15}   {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"  {'Real 1':>15}   {cm[1][0]:>6}      {cm[1][1]:>6}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Sem Burnout (0)', 'Com Burnout (1)']))

# =============================================
# 9. DETALHES DA ARQUITETURA FINAL
# =============================================

print("=" * 55)
print("  ARQUITETURA DA REDE NEURAL OTIMIZADA")
print("=" * 55)
print(f"  Camadas ocultas: {best_model.hidden_layer_sizes}")
print(f"  Função de ativação: {best_model.activation}")
print(f"  Otimizador: {best_model.solver}")
print(f"  Regularização (alpha): {best_model.alpha}")
print(f"  Taxa de aprendizado: {best_model.learning_rate}")
print(f"  Iterações realizadas: {best_model.n_iter_}")
print(f"  Nº de features de entrada: {best_model.n_features_in_}")
print(f"  Nº de camadas (total): {best_model.n_layers_}")