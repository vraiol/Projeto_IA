# -*- coding: utf-8 -*-
# MLP (Keras/TensorFlow) para classificação de Burnout
# Requisitos: scikit-learn==1.5.2, scikeras>=1.0.0

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from scikeras.wrappers import KerasClassifier

# ============================================================
# 1. Carregamento e pré-processamento dos dados
# ============================================================

pasta = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta, 'tech_mental_health_burnout_tratado.csv')
df = pd.read_csv(arquivo)

print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas\n")

# remover colunas que não devem entrar no modelo
df = df.drop(columns=['burnout_score'])  # vazaria informação
df = df.dropna(subset=['burnout_level'])

# codificar colunas categóricas (string -> número)
colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
encoders = {}

for col in colunas_categoricas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Codificado '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# separar features (X) e target (y)
X = df.drop(columns=['burnout_level']).values
y = df['burnout_level'].astype(int).values

nomes_classes = ['Sem Burnout (0)', 'Com Burnout (1)']

print(f"\nFeatures: {df.drop(columns=['burnout_level']).columns.tolist()}")
print(f"Distribuição do target: 0={np.sum(y==0)}, 1={np.sum(y==1)}\n")

# ============================================================
# 2. Divisão treino / teste
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {x_train.shape[0]} amostras | Teste: {x_test.shape[0]} amostras\n")

# ============================================================
# 3. Escalonamento (Padronização)
# ============================================================

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # só transform no teste

n_features = x_train.shape[1]

# ============================================================
# 4. Construção do modelo MLP
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # saída binária
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# 5. Treinamento
# ============================================================

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

historico = model.fit(x_train, y_train,
                      epochs=100,
                      batch_size=32,
                      validation_split=0.2,
                      callbacks=[early_stopping])

# ============================================================
# 6. Avaliação no conjunto de teste
# ============================================================

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

# guardar acurácia do baseline para comparação posterior
acuracia_baseline = test_acc

# ============================================================
# 7. Curvas de acurácia e perda
# ============================================================

pasta_graficos = os.path.join(pasta, 'graficos')
os.makedirs(pasta_graficos, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(historico.history['accuracy'], label='Acurácia Treino')
ax1.plot(historico.history['val_accuracy'], label='Acurácia Validação')
ax1.set_title('Evolução da Acurácia')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Acurácia')
ax1.legend()

ax2.plot(historico.history['loss'], label='Perda Treino')
ax2.plot(historico.history['val_loss'], label='Perda Validação')
ax2.set_title('Evolução da Perda/Erro')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Erro')
ax2.legend()

fig.savefig(os.path.join(pasta_graficos, 'curvas_treinamento.png'), dpi=150)

print("Gráfico salvo: curvas_treinamento.png")

# ============================================================
# 8. Matriz de Confusão
# ============================================================

y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print(f"\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=nomes_classes))

cm_abs = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_abs, display_labels=nomes_classes)
disp1.plot(ax=ax[0], cmap="plasma", values_format='d')
ax[0].set_title("Matriz de Confusão (Valores Absolutos)")

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=nomes_classes)
disp2.plot(ax=ax[1], cmap="plasma", values_format=".2f")
ax[1].set_title("Matriz de Confusão (Percentual)")

plt.tight_layout()
fig.savefig(os.path.join(pasta_graficos, 'matriz_confusao.png'), dpi=150)

print("Gráfico salvo: matriz_confusao.png")

# ============================================================
# 9. Cross-Validation
# ============================================================

def criar_rede_cross():
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

rede_cv = KerasClassifier(model=criar_rede_cross,
                          epochs=30,
                          batch_size=64,
                          verbose=0)

# usa todos os dados de treino já escalonados
resultados_cv = cross_val_score(estimator=rede_cv,
                                X=x_train,
                                y=y_train,
                                cv=5,
                                scoring='accuracy')

print(f"\nCross-Validation (5-Fold):")
print(f"  Acurácias por fold: {resultados_cv}")
print(f"  Acurácia média: {resultados_cv.mean():.4f} ± {resultados_cv.std():.4f}")

# ============================================================
# 10. Fine Tuning com GridSearchCV
# ============================================================

def criar_rede_finetuning(optimizer='adam', activation='relu', neurons=64):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(n_features,)))
    model.add(tf.keras.layers.Dense(units=neurons, activation=activation))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

classificador = KerasClassifier(model=criar_rede_finetuning, verbose=0)

parametros = {
    'batch_size': [32, 64],
    'epochs': [30],
    'model__optimizer': ['adam', 'sgd'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [64, 128, 256]
}

# total: 2 * 1 * 2 * 2 * 3 = 24 combinações x 3 folds = 72 treinamentos
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           cv=3,
                           verbose=2)

print("\nIniciando Grid Search (pode levar alguns minutos)...\n")
grid_search.fit(x_train, y_train)

melhores = grid_search.best_params_
melhor_acuracia = grid_search.best_score_

print(f"\nMelhores parâmetros encontrados: {melhores}")
print(f"Melhor acurácia: {melhor_acuracia:.4f}")

# tabela de resultados
resultados_grid = pd.DataFrame(grid_search.cv_results_)

tabela = resultados_grid[[
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'param_batch_size',
    'param_epochs',
    'param_model__optimizer',
    'param_model__activation',
    'param_model__neurons'
]].copy()

tabela.columns = [
    'ranking', 'acuracia', 'd_padrao',
    'batch', 'epochs', 'optimizer', 'activation', 'neurons'
]

tabela = tabela.sort_values(by='ranking')
print("\nResultados do Grid Search:")
print(tabela.to_string(index=False))

# ============================================================
# Gráfico comparativo: Baseline vs. Modelo Otimizado
# ============================================================

acuracia_otimizado = melhor_acuracia  # acurácia média do melhor modelo no Grid Search (CV)

rotulos = ['MLP Baseline\n(sem ajuste)', 'MLP Otimizado\n(Grid Search + CV)']
valores = [acuracia_baseline, acuracia_otimizado]
cores = ['#6c757d', '#198754']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(rotulos, valores, color=cores, width=0.4, edgecolor='white', linewidth=1.5)

# valor percentual em cima de cada barra
for bar, val in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.2%}',
            ha='center', va='bottom',
            fontsize=13, fontweight='bold')

# linha de referência na acurácia do baseline
ax.axhline(y=acuracia_baseline, color='#6c757d', linestyle='--', linewidth=1.2, alpha=0.6)

ax.set_ylim(0, 1.05)
ax.set_ylabel('Acurácia', fontsize=12)
ax.set_title('Comparação: MLP Baseline vs. MLP Otimizado', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(pasta_graficos, 'comparativo_baseline_otimizado.png'), dpi=150)
print("\nGráfico salvo: comparativo_baseline_otimizado.png")

ganho = acuracia_otimizado - acuracia_baseline
print(f"\nGanho de acurácia com o ajuste de hiperparâmetros: {ganho:+.2%}")

# ============================================================
# 11. Análise de Sensibilidade de Hiperparâmetros
# ============================================================
# Procedimento: fixar todos os hiperparâmetros na melhor configuração
# e variar apenas UM por vez, com pelo menos 5 valores diferentes.

melhor_batch = melhores['batch_size']
melhor_epochs = melhores['epochs']
melhor_optimizer = melhores['model__optimizer']
melhor_activation = melhores['model__activation']
melhor_neurons = melhores['model__neurons']

def avaliar_config(batch_size, epochs, optimizer, activation, neurons, cv=3):
    """Treina e avalia com cross-validation, retorna acurácia média e desvio."""
    def criar_modelo():
        tf.keras.backend.clear_session()
        m = tf.keras.models.Sequential()
        m.add(tf.keras.layers.InputLayer(shape=(n_features,)))
        m.add(tf.keras.layers.Dense(units=neurons, activation=activation))
        m.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        m.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        return m

    clf = KerasClassifier(model=criar_modelo,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=0)
    scores = cross_val_score(clf, x_train, y_train, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

print("\n" + "=" * 60)
print("  ANÁLISE DE SENSIBILIDADE DE HIPERPARÂMETROS")
print("=" * 60)
print(f"\nConfiguração base (melhor do GridSearch):")
print(f"  batch_size={melhor_batch}, epochs={melhor_epochs}, "
      f"optimizer={melhor_optimizer}, activation={melhor_activation}, "
      f"neurons={melhor_neurons}")

# --------------------------------------------------
# Sensibilidade 1: Número de neurônios
# --------------------------------------------------
print("\n--- Sensibilidade: Número de Neurônios ---")
valores_neurons = [16, 32, 64, 128, 256, 512]
resultados_neurons = []

for n in valores_neurons:
    media, desvio = avaliar_config(melhor_batch, melhor_epochs,
                                   melhor_optimizer, melhor_activation, n)
    resultados_neurons.append((n, media, desvio))
    print(f"  neurons={n:>4} -> acurácia: {media:.4f} ± {desvio:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ns = [r[0] for r in resultados_neurons]
ms = [r[1] for r in resultados_neurons]
ds = [r[2] for r in resultados_neurons]
ax.errorbar(ns, ms, yerr=ds, marker='o', capsize=5, linewidth=2, markersize=8)
ax.set_title('Sensibilidade: Número de Neurônios', fontsize=14, fontweight='bold')
ax.set_xlabel('Número de Neurônios')
ax.set_ylabel('Acurácia (CV)')
ax.set_xticks(ns)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(pasta_graficos, 'sensibilidade_neurons.png'), dpi=150)

print("Gráfico salvo: sensibilidade_neurons.png")

# --------------------------------------------------
# Sensibilidade 2: Tamanho do batch
# --------------------------------------------------
print("\n--- Sensibilidade: Tamanho do Batch ---")
valores_batch = [16, 32, 64, 128, 256]
resultados_batch = []

for b in valores_batch:
    media, desvio = avaliar_config(b, melhor_epochs,
                                   melhor_optimizer, melhor_activation,
                                   melhor_neurons)
    resultados_batch.append((b, media, desvio))
    print(f"  batch_size={b:>4} -> acurácia: {media:.4f} ± {desvio:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
bs = [r[0] for r in resultados_batch]
ms = [r[1] for r in resultados_batch]
ds = [r[2] for r in resultados_batch]
ax.errorbar(bs, ms, yerr=ds, marker='s', capsize=5, linewidth=2, markersize=8, color='green')
ax.set_title('Sensibilidade: Tamanho do Batch', fontsize=14, fontweight='bold')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Acurácia (CV)')
ax.set_xticks(bs)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(pasta_graficos, 'sensibilidade_batch.png'), dpi=150)

print("Gráfico salvo: sensibilidade_batch.png")

# --------------------------------------------------
# Sensibilidade 3: Número de épocas
# --------------------------------------------------
print("\n--- Sensibilidade: Número de Épocas ---")
valores_epochs = [5, 10, 30, 50, 100]
resultados_epochs = []

for e in valores_epochs:
    media, desvio = avaliar_config(melhor_batch, e,
                                   melhor_optimizer, melhor_activation,
                                   melhor_neurons)
    resultados_epochs.append((e, media, desvio))
    print(f"  epochs={e:>4} -> acurácia: {media:.4f} ± {desvio:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
es = [r[0] for r in resultados_epochs]
ms = [r[1] for r in resultados_epochs]
ds = [r[2] for r in resultados_epochs]
ax.errorbar(es, ms, yerr=ds, marker='^', capsize=5, linewidth=2, markersize=8, color='red')
ax.set_title('Sensibilidade: Número de Épocas', fontsize=14, fontweight='bold')
ax.set_xlabel('Épocas')
ax.set_ylabel('Acurácia (CV)')
ax.set_xticks(es)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(pasta_graficos, 'sensibilidade_epochs.png'), dpi=150)

print("Gráfico salvo: sensibilidade_epochs.png")

# --------------------------------------------------
# Resumo comparativo
# --------------------------------------------------
print("\n" + "=" * 60)
print("  RESUMO DA ANÁLISE DE SENSIBILIDADE")
print("=" * 60)

def melhor_de(resultados, nome):
    melhor = max(resultados, key=lambda x: x[1])
    return f"  {nome}: melhor valor = {melhor[0]} (acurácia: {melhor[1]:.4f})"

print(melhor_de(resultados_neurons, "Neurônios"))
print(melhor_de(resultados_batch, "Batch Size"))
print(melhor_de(resultados_epochs, "Épocas"))