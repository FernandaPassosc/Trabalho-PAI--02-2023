## COM DISTÂNCIA E JANELINHA E N

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import os
import cv2
import numpy as np

# Variáveis globais
img_path = None
img_class = None
N = 100  # Valor padrão de N
root = tk.Tk()
root.title("Visualizador e Analisador de Imagens")

bethesda_var = tk.StringVar()
bethesda_var.set("Bethesda: ")

# Adiciona uma variável de controle para o widget de entrada
n_var = tk.StringVar()
n_var.set(str(N))  # Inicializa o widget de entrada com o valor padrão



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

        # Calculando circularidade
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0

        # Calculando excentricidade (usando a relação entre os eixos da elipse ajustada)
        (x, y), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0

        return area, perimeter, circularity, eccentricity
    else:
        return None, None, None, None


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def open_image():
    global img_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png;.jpg")])
    if file_path:
        img_path = file_path
        img = Image.open(img_path)

        if img.width > 500 or img.height > 500:
            img.thumbnail((500, 500))

        display_image(img)

def display_image(image, zoom=1):
    # Aplicar o zoom na imagem
    width, height = image.size
    # Verifica a versão da Pillow e escolhe o método de redimensionamento apropriado
    zoomed_image = image.resize((int(width * zoom), int(height * zoom)), Image.Resampling.LANCZOS)

    tk_image = ImageTk.PhotoImage(zoomed_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image
    image_label.pack(expand=True)

def update_zoom(event):
    global zoom_level, img_path
    zoom_level = zoom_slider.get()
    if img_path:
        img = Image.open(img_path)
        display_image(img, zoom=zoom_level)

def process_image(show_class=False):
    global img_path, img_class

    if img_path:
        # Lendo o arquivo CSV
        df = pd.read_csv('classifications.csv')

        # Criando subdiretórios para cada classe
        classes = ['ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC', 'Negative for intraepithelial lesion']

        for cls in classes:
            os.makedirs(cls, exist_ok=True)

    # Metade do tamanho da área de corte
    nuclei_size = N
    half_size = nuclei_size // 2

  # Processando as imagens
    for index, row in df.iterrows():
            if row['image_filename'] == os.path.basename(img_path):
                cell_id = row['cell_id']
                img_class = row['bethesda_system']
                nucleus_x = row['nucleus_x']
                nucleus_y = row['nucleus_y']

                with Image.open(img_path) as img:
                    # Definindo a área de corte
                    left = max(nucleus_x - half_size, 0)
                    top = max(nucleus_y - half_size, 0)
                    right = min(nucleus_x + half_size, img.width)
                    bottom = min(nucleus_y + half_size, img.height)

                    print(f"Núcleo: ({nucleus_x}, {nucleus_y})")
                    print(f"Centro da imagem recortada: ({(left + right) / 2}, {(top + bottom) / 2})")

                    # Recortando a imagem
                    cropped_img = img.crop((left, top, right, bottom))



                    # Calcular descritores de forma
                    area, perimeter, circularity, eccentricity = calculate_shape_descriptors(cropped_img)
                    if area is not None and perimeter is not None:
                        print(f"Área: {area} milímetros, Perímetro: {perimeter} milímetros, Circularidade: {circularity}, Excentricidade: {eccentricity}")

                    # Calculando os novos valores de x e y para fazer a distância
                    new_x = cropped_img.width // 2
                    new_y = cropped_img.height // 2

                    distance_to_center = calculate_distance(
                        nucleus_x, nucleus_y,
                        new_x, new_y
                    )

                    # Atualiza a variável de controle da Bethesda
                    bethesda_var.set(f"Bethesda: {img_class}")

                    # Salvando a sub-imagem no subdiretório correspondente
                    cropped_img.save(f"{img_class}/{cell_id}.png")

                    # Atualiza a imagem na interface
                    display_image(cropped_img)

                    # Atualiza o rótulo de distância
                    distance_label.config(text=f"Distância para o centro: {distance_to_center}")

                    if show_class:
                        show_class_label.config(text=f"Bethesda: {img_class}")
                        show_class_label.pack()

                    # Mostra a janela com os resultados
                    show_results_window(area, perimeter, circularity, eccentricity, distance_to_center)

                    break


def show_class():
    process_image(show_class=True)

def show_results_window(area, perimeter, circularity, eccentricity, distance_to_center):
    results_window = tk.Toplevel(root)
    results_window.title("Resultados")
    # Rótulos para os resultados
    area_label = tk.Label(results_window, text=f"Área: {area} milímetros")
    area_label.pack()

    perimeter_label = tk.Label(results_window, text=f"Perímetro: {perimeter} milímetros")
    perimeter_label.pack()

    circularity_label = tk.Label(results_window, text=f"Circularidade: {circularity}")
    circularity_label.pack()

    eccentricity_label = tk.Label(results_window, text=f"Excentricidade: {eccentricity}")
    eccentricity_label.pack()

    distance_label = tk.Label(results_window, text=f"Distância para o centro: {distance_to_center}")
    distance_label.pack()

# Label para exibir a imagem
image_label = tk.Label(root)

# Adicionando botões
button_frame = tk.Frame(root)
button_frame.pack()

open_button = tk.Button(button_frame, text="Abrir Imagem", command=open_image)
open_button.pack(side=tk.LEFT)

process_button = tk.Button(button_frame, text="Processar Imagem", command=lambda: process_image(show_class=False))
process_button.pack(side=tk.LEFT)

# Adiciona um rótulo e uma entrada para permitir que o usuário defina N
n_label = tk.Label(button_frame, text="Valor de N:")
n_label.pack(side=tk.LEFT)

n_entry = tk.Entry(button_frame, textvariable=n_var)
n_entry.pack(side=tk.LEFT)

# Adicionar um controle deslizante para o zoom
zoom_slider = tk.Scale(root, from_=1, to=5, orient=tk.HORIZONTAL, resolution=0.1, command=update_zoom)
zoom_slider.pack()

# Botão para mostrar a classe
show_class_button = tk.Button(button_frame, text="Bethesda", command=show_class)
show_class_button.pack(side=tk.LEFT)
show_class_label = tk.Label(root, textvariable=bethesda_var)

# Label para mostrar a distância
distance_label = tk.Label(root)

root.mainloop()

