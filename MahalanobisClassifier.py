import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def carregar_dados(caminho_arquivo):
    # Carregar dados do arquivo CSV
    df = pd.read_csv(caminho_arquivo)
    return df

def preprocessar_dados(df):
    # Extrair nomes das características numéricas
    caracteristicas_numericas = df.columns[2:-2].tolist()

    # Escalonar características com os nomes fornecidos
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df[caracteristicas_numericas]), columns=caracteristicas_numericas)

    # Calcular média e matriz de covariância inversa regularizada
    media = X_scaled.mean().values
    cov_matrix = X_scaled.cov().values
    reg_param = 1e-5  # Ajustar o parâmetro de regularização conforme necessário
    inv_cov_matrix = np.linalg.inv(cov_matrix + reg_param * np.identity(cov_matrix.shape[0]))

    # Calcular distância de Mahalanobis
    distancias = [mahalanobis(row.values, media, inv_cov_matrix) for _, row in X_scaled.iterrows()]

    # Adicionar distâncias e colunas originais ao DataFrame
    df["distancia_mahalanobis"] = distancias

    # Criar a coluna "cancer_positive" que classifica se a amostra é positiva ou negativa para câncer
    df["cancer_positive"] = df["image_class"].apply(lambda x: 0 if x == "Negative for intraepithelial lesion" else 1)

    return df, X_scaled, media, inv_cov_matrix

def treinar_e_avaliar_modelo(X_treino, y_treino, X_teste, y_teste, inv_cov_matrix):
    # Definir o grid de parâmetros
    param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "p": [1, 2]}

    # Criar a busca em grade
    grid_search = GridSearchCV(
        KNeighborsClassifier(metric="mahalanobis", metric_params={"V": inv_cov_matrix}),
        param_grid,
        cv=5,
        scoring="accuracy",
    )

    # Ajustar a busca em grade aos dados
    grid_search.fit(X_treino, y_treino)

    # Obter os melhores parâmetros
    melhores_parametros = grid_search.best_params_
    print(f"Melhores Parâmetros: {melhores_parametros}")

    # Utilizar o melhor modelo para a previsão
    melhor_modelo = grid_search.best_estimator_
    y_pred = melhor_modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    print(f'Acurácia com o Melhor Modelo: {acuracia}')

def plotar_histogramas(df, coluna, hue, titulo, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.histplot(df, x=coluna, hue=hue, bins=50, kde=True)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotar_box_scatter(df, x_coluna, y_coluna, hue, palette, titulo, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=x_coluna, y=y_coluna, data=df, hue=hue, palette=palette)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotar_heatmap(matriz, cmap, cbar_label, titulo, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz, cmap=cmap, cbar_kws={"label": cbar_label})
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    caminho_arquivo = "nuclei_data.csv"
    df, X_scaled, media, inv_cov_matrix = preprocessar_dados(carregar_dados(caminho_arquivo))
    X_treino, X_teste, y_treino, y_teste = train_test_split(X_scaled, df["image_class"], test_size=0.2, random_state=42)

    # Treinar e avaliar o modelo
    treinar_e_avaliar_modelo(X_treino, y_treino, X_teste, y_teste, inv_cov_matrix)

    # Plotar histogramas por classe
    plotar_histogramas(df, "distancia_mahalanobis", "image_class", "Distribuição das Distâncias de Mahalanobis por Classe", "Distância de Mahalanobis", "Frequência")

    # Box Plot e Scatter Plot das Distâncias de Mahalanobis por Classe
    plotar_box_scatter(df, "distancia_mahalanobis", "image_class", "image_class", "viridis", "Gráfico de Dispersão das Distâncias de Mahalanobis por Classe", "Distância de Mahalanobis", "Image Class")

    # Mapa de Calor da Distância de Mahalanobis
    matriz_mahalanobis = np.array([mahalanobis(row.values, media, inv_cov_matrix) for _, row in X_scaled.iterrows()])
    matriz_mahalanobis = matriz_mahalanobis.reshape((len(matriz_mahalanobis), 1))
    plotar_heatmap(matriz_mahalanobis, "viridis", "Distância de Mahalanobis", "Mapa de Calor da Distância de Mahalanobis", "Índice da Amostra", "Distância de Mahalanobis")

    # Plotar histogramas por cancer_positive
    plotar_histogramas(df, "distancia_mahalanobis", "cancer_positive", "Distribuição das Distâncias de Mahalanobis por Classe", "Distância de Mahalanobis", "Frequência")

    # Box Plot e Scatter Plot das Distâncias de Mahalanobis por cancer_positive
    plotar_box_scatter(df, "distancia_mahalanobis", "cancer_positive", "cancer_positive", "viridis", "Gráfico de Dispersão das Distâncias de Mahalanobis por Classe", "Distância de Mahalanobis", "Cancer Positive")

main()