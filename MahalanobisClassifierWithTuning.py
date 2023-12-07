# Classifica as amostras de células de acordo com a presença ou não de câncer
# Tuning de hiperparâmetros com busca em grade

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar dados do arquivo CSV
file_path = "nuclei_data.csv"
df = pd.read_csv(file_path)

# Extrair nomes das características numéricas
numerical_features = df.columns[
    2:-2
].tolist()  # Excluir 'mahalanobis_distance' e 'cancer_positive'

# Escalonamento das características com os nomes fornecidos
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df[numerical_features]), columns=numerical_features
)

# Calcular média e matriz de covariância inversa regularizada
mean = X_scaled.mean().values
cov_matrix = X_scaled.cov().values
reg_param = 1e-5  # Ajustar o parâmetro de regularização conforme necessário
inv_cov_matrix = np.linalg.inv(
    cov_matrix + reg_param * np.identity(cov_matrix.shape[0])
)

# Calcular distância de Mahalanobis
distances = []
for index, row in X_scaled.iterrows():
    sample = row.values
    distance = mahalanobis(sample, mean, inv_cov_matrix)
    distances.append(distance)

# Adicionar distâncias e colunas originais ao DataFrame
df["mahalanobis_distance"] = distances

# Criando a coluna "cancer_positive" que classifica se a amostra é positiva ou negativa para câncer
df["cancer_positive"] = df["image_class"].apply(
    lambda x: 0 if x == "Negative for intraepithelial lesion" else 1
)
# df['cancer_positive'] = 1 - df['cancer_positive']  # Trocar 0 por 1 e vice-versa

# Divisão em treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["image_class"], test_size=0.2, random_state=42
)

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
grid_search.fit(X_train, y_train)

# Obter os melhores parâmetros
best_params = grid_search.best_params_
print(f"Melhores Parâmetros: {best_params}")

# Utilizar o melhor modelo para a previsão
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia com o Melhor Modelo: {accuracy}")
