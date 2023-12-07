import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import cv2
import os
from PIL import Image
import time

# Variávels globais
N = 100
treinar_modelo = False
mostrar_graficos = True
caminho_arquivo = "nuclei_data.csv"
arquivo_modelo = "mahalanobis_model.joblib"


def calculate_shape_descriptors(image):
    # Convertendo a imagem para escala de cinza e binarizando
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Encontrando contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculando descritores para o maior contorno (assumindo ser o núcleo)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = (
            4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
        )

        # Verificar se o contorno tem pelo menos 5 pontos
        if len(contour) >= 5:
            (x, y), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)
            eccentricity = (
                np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
            )
        else:
            eccentricity = (
                -1
            )  # Ou algum outro valor para indicar que não foi possível calcular

        compactness = (
            (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
        )

        area_cm = round(area / 100, 4)
        perimeter_cm = round(perimeter / 100, 4)
        circularity = round(circularity, 4)
        eccentricity = round(eccentricity, 4)
        compactness = round(compactness, 4)

        return area_cm, perimeter_cm, circularity, eccentricity, compactness
    else:
        return None, None, None, None, None


def carregar_dados(caminho_arquivo):
    # Carregar dados do arquivo CSV
    df = pd.read_csv(caminho_arquivo)
    return df


def preprocessar_dados(df):
    # Select relevant columns for training
    caracteristicas_numericas = df[
        ["area", "perimeter", "circularity", "eccentricity", "compactness"]
    ]

    # Standardize the selected features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(caracteristicas_numericas),
        columns=caracteristicas_numericas.columns,
    )

    # Encode the "image_class" column
    label_encoder = LabelEncoder()
    df["image_class_encoded"] = label_encoder.fit_transform(df["image_class"])

    # Calculate mean and regularized inverse covariance matrix
    media = X_scaled.mean().values
    cov_matrix = X_scaled.cov().values
    reg_param = 1e-5
    inv_cov_matrix = np.linalg.inv(
        cov_matrix + reg_param * np.identity(cov_matrix.shape[0])
    )

    # Calculate Mahalanobis distances
    distancias = [
        mahalanobis(row.values, media, inv_cov_matrix) for _, row in X_scaled.iterrows()
    ]

    # Add distances and original columns to the DataFrame
    df["distancia_mahalanobis"] = distancias

    # Create the "cancer_positive" column
    df["cancer_positive"] = df["image_class_encoded"].apply(
        lambda x: 0
        if x == label_encoder.transform(["Negative for intraepithelial lesion"])[0]
        else 1
    )

    return df, X_scaled, media, inv_cov_matrix, label_encoder


def treinar_e_avaliar_modelo(X_treino, y_treino, X_teste, y_teste, inv_cov_matrix):
    # Define the parameter grid
    param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }

    # Create the grid search
    grid_search = GridSearchCV(
        KNeighborsClassifier(metric="mahalanobis", metric_params={"V": inv_cov_matrix}),
        param_grid,
        cv=5,
        scoring="accuracy",
    )

    # Fit the grid search to the data
    grid_search.fit(X_treino, y_treino)

    # Get the best parameters
    melhores_parametros = grid_search.best_params_
    print(f"Melhores Parâmetros: {melhores_parametros}")

    # Use the best model for prediction
    melhor_modelo = grid_search.best_estimator_
    y_pred = melhor_modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    print(f"Acurácia com o Melhor Modelo: {acuracia}")

    # Save the best model to a file
    joblib.dump(melhor_modelo, arquivo_modelo)
    print("Melhor modelo salvo como 'mahalanobis_model.joblib'.")
    return melhor_modelo


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


def prever(img, modelo, label_encoder, printar_output):
    df = obter_atributos_por_imagem(img)
    df = pd.DataFrame(df)

    # Carregar o modelo salvo do arquivo
    modelo = joblib.load(arquivo_modelo)

    # Prever a classificação para cada linha no DataFrame
    predictions = modelo.predict(
        df.drop(["image_id", "nucleus_id", "image_class"], axis=1)
    )

    # Adicionar as predições ao DataFrame
    df["predicao"] = predictions

    # Convert the "predicao" column to numerical values
    df["predicao"] = df["predicao"].apply(
        lambda x: 0
        if x == label_encoder.transform(["Negative for intraepithelial lesion"])[0]
        else 1
    )

    # Calcular a média das predições para obter a classificação final
    media_predicao = df["predicao"].mean()

    # Classificar como cancer_positive 0 ou 1 (usando uma regra, por exemplo, média > 0.5)
    cancer_positive = 1 if media_predicao > 0.5 else 0

    if printar_output:
        print("Imagem processada:", img)
        print(f"Predição média de cancer_positive: {media_predicao:.4f}")
        print(f"Classificação final de cancer_positive: {cancer_positive}")

    return media_predicao, cancer_positive


def obter_atributos_por_imagem(img_path):
    global img_class

    # Lista para armazenar os dados dos núcleos
    nuclei_data = []

    if img_path:
        df = pd.read_csv("classifications.csv")

    nuclei_size = N
    half_size = nuclei_size // 2

    for index, row in df.iterrows():
        if row["image_filename"] == os.path.basename(img_path):
            nucleus_id = row["cell_id"]
            img_class = row["bethesda_system"]
            nucleus_x = row["nucleus_x"]
            nucleus_y = row["nucleus_y"]
            image_id = row["image_id"]

            with Image.open(img_path) as img:
                # Definindo a área de corte
                left = max(nucleus_x - half_size, 0)
                top = max(nucleus_y - half_size, 0)
                right = min(nucleus_x + half_size, img.width)
                bottom = min(nucleus_y + half_size, img.height)

                # Recortando a imagem
                cropped_img = img.crop((left, top, right, bottom))
                cropped_img_np = np.array(cropped_img)

                gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                thresholded = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2,
                )

                contours, _ = cv2.findContours(
                    thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # Calcular descritores de forma
                    (
                        area,
                        perimeter,
                        circularity,
                        eccentricity,
                        compactness,
                    ) = calculate_shape_descriptors(cropped_img)

                    if area is not None and perimeter is not None:
                        nuclei_data.append(
                            {
                                "image_id": image_id,
                                "nucleus_id": nucleus_id,
                                "area": area,
                                "perimeter": perimeter,
                                "circularity": circularity,
                                "eccentricity": eccentricity,
                                "compactness": compactness,
                                "image_class": img_class,
                            }
                        )

    return nuclei_data


def main():
    df, X_scaled, media, inv_cov_matrix, label_encoder = preprocessar_dados(
        carregar_dados(caminho_arquivo)
    )
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_scaled, df["image_class"], test_size=0.2, random_state=42
    )

    # Treinar o modelo ou carregar o modelo salvo
    if treinar_modelo:
        # Treinar e avaliar o modelo
        melhor_modelo = treinar_e_avaliar_modelo(
            X_treino, y_treino, X_teste, y_teste, inv_cov_matrix
        )
    else:
        melhor_modelo = joblib.load(arquivo_modelo)

    # Prever alguma imagem com o modelo
    # img_arr = ["dataset/011fda505d7e4af4b8cc57545343624d.png"]
    img_arr = [
        ("dataset/" + f)
        for f in os.listdir("dataset")
        if os.path.isfile(os.path.join("dataset", f))
    ]

    media_medias = 0
    media_preds = 0

    tamanho_do_lote = 50
    total_de_imagens = len(img_arr)

    tempo_inicial = time.time()

    for i in range(0, total_de_imagens, tamanho_do_lote):
        imagens_do_lote = img_arr[i : i + tamanho_do_lote]

        tempo_inicial_do_lote = time.time()

        for img in imagens_do_lote:
            printar_output = True
            media, pred = prever(img, melhor_modelo, label_encoder, printar_output)
            media_medias += media / total_de_imagens
            media_preds += pred / total_de_imagens

        tempo_final_do_lote = time.time()
        tempo_decorrido_do_lote = tempo_final_do_lote - tempo_inicial_do_lote

        imagens_processadas = i + len(imagens_do_lote)
        imagens_restantes = total_de_imagens - imagens_processadas

        print(
            f"Lote de {tamanho_do_lote} imagens processado em {tempo_decorrido_do_lote:.2f} segundos. "
            f"{imagens_processadas} imagens processadas. {imagens_restantes} imagens restantes."
        )

    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial

    print(f"Processamento completo em {tempo_decorrido} segundos.")
    print(f"Média de predições: {media_preds:.4f}")
    print(f"Média de cancer_positive: {media_medias:.4f}")

    ######################### GRÁFICOS #########################
    if mostrar_graficos:
        # Plotar histogramas por classe
        plotar_histogramas(
            df,
            "distancia_mahalanobis",
            "image_class",
            "Distribuição das Distâncias de Mahalanobis por Classe",
            "Distância de Mahalanobis",
            "Frequência",
        )

        # Box Plot e Scatter Plot das Distâncias de Mahalanobis por Classe
        plotar_box_scatter(
            df,
            "distancia_mahalanobis",
            "image_class",
            "image_class",
            "viridis",
            "Gráfico de Dispersão das Distâncias de Mahalanobis por Classe",
            "Distância de Mahalanobis",
            "Image Class",
        )

        # Mapa de Calor da Distância de Mahalanobis
        matriz_mahalanobis = np.array(
            [
                mahalanobis(row.values, media, inv_cov_matrix)
                for _, row in X_scaled.iterrows()
            ]
        )
        matriz_mahalanobis = matriz_mahalanobis.reshape((len(matriz_mahalanobis), 1))
        plotar_heatmap(
            matriz_mahalanobis,
            "viridis",
            "Distância de Mahalanobis",
            "Mapa de Calor da Distância de Mahalanobis",
            "Índice da Amostra",
            "Distância de Mahalanobis",
        )

        # Plotar histogramas por cancer_positive
        plotar_histogramas(
            df,
            "distancia_mahalanobis",
            "cancer_positive",
            "Distribuição das Distâncias de Mahalanobis por Classe",
            "Distância de Mahalanobis",
            "Frequência",
        )

        # Box Plot e Scatter Plot das Distâncias de Mahalanobis por cancer_positive
        plotar_box_scatter(
            df,
            "distancia_mahalanobis",
            "cancer_positive",
            "cancer_positive",
            "viridis",
            "Gráfico de Dispersão das Distâncias de Mahalanobis por Classe",
            "Distância de Mahalanobis",
            "Cancer Positive",
        )


main()
