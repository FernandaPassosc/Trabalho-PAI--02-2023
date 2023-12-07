## COM DISTÂNCIA E JANELINHA E N
from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis globais
img_class = None
N = 100  # Valor padrão de N

def calculate_shape_descriptors(contours):    
    # Calculando descritores para o maior contorno (assumindo ser o núcleo)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0

        # Verificar se o contorno tem pelo menos 5 pontos
        if len(contour) >= 5:
            (x, y), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
        else:
            eccentricity = -1  # Ou algum outro valor para indicar que não foi possível calcular

        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0

        area_cm = round(area / 100, 4)
        perimeter_cm = round(perimeter / 100, 4)
        circularity = round(circularity, 4)
        eccentricity = round(eccentricity, 4)
        compactness = round(compactness, 4)

        return area_cm, perimeter_cm, circularity, eccentricity, compactness
    else:
        return None, None, None, None, None

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def process_image(img_path):
    global img_class

    # Lista para armazenar os dados dos núcleos
    nuclei_data = []

    if img_path:
        df = pd.read_csv('classifications.csv')

    nuclei_size = N
    half_size = nuclei_size // 2

    for index, row in df.iterrows():
        if row['image_filename'] == os.path.basename(img_path):
            nucleus_id = row['cell_id']
            img_class = row['bethesda_system']
            nucleus_x = row['nucleus_x']
            nucleus_y = row['nucleus_y']
            image_id = row['image_id']

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
                thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

                    # Aplicando a máscara na imagem original recortada
                    segmented_image = np.zeros_like(cropped_img_np)
                    
                    for i in range(3):  # Aplicar a máscara em cada canal de cor
                        segmented_image[:,:,i] = cropped_img_np[:,:,i] & mask

                    # Salvar a imagem segmentada
                    segmented_path = os.path.join(img_class, f"{nucleus_id}_segmented.png")

                    cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
                    
                    # Calcular descritores de forma
                    area, perimeter, circularity, eccentricity, compactness = calculate_shape_descriptors(contours)
                
                    if area is not None and perimeter is not None:
                        print(f"Área: {area} milímetros, Perímetro: {perimeter} milímetros, Circularidade: {circularity}, Excentricidade: {eccentricity}, Compacidade: {compactness}")
                        print(f"{nucleus_id}/12.229.511")

                        nuclei_data.append({
                            "image_id": image_id,
                            "nucleus_id": nucleus_id,
                            "area": area,
                            "perimeter": perimeter,
                            "circularity": circularity,
                            "eccentricity": eccentricity,
                            "compactness": compactness,
                            "image_class": img_class
                        })

    return nuclei_data 

# Criando subdiretórios para cada classe
classes = ['ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC', 'Negative for intraepithelial lesion']

def process_all_images():
    df = pd.read_csv('classifications.csv')
    unique_images = df['image_filename'].unique()

    all_nuclei_data = []

    for cls in classes:
        os.makedirs(cls, exist_ok=True)

    for img_name in unique_images:
        img_path = 'dataset/' + img_name  # Substitua com o caminho correto
        nuclei_data = process_image(img_path)
        all_nuclei_data.extend(nuclei_data)

    # Convertendo a lista de dicionários em um DataFrame
    all_nuclei_data_df = pd.DataFrame(all_nuclei_data)

    # Salvando o DataFrame em um arquivo CSV
    all_nuclei_data_df.to_csv('nuclei_data.csv', index=False)
    print("Dados salvos em nuclei_data.csv")

    return all_nuclei_data_df

# Chamada da função para processar todas as imagens
all_nuclei_data = process_all_images()