from PIL import Image, ImageTk
import pandas as pd
import numpy as np 
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

## PROCESSAMENTO E ANÁLISE DE IMAGENS - 02/2023
## Camila Lacerda Grandini - 725193 
## Fernanda Ribeiro Passos Cirino - 680624
## Luiz Fernando Oliveira Maciel - 727245
## Milena Soares Barreira - 727541


# Global variables
root = tk.Tk()
root.title ("Trabalho de Processamento e Análise de Imagens")
N = 100
original_image = None
processed_image = None
segmented_image = None
image_path = None
nuclei = []





# def set_original_image(org_img):
#     orginal_img = org_img
#     return original_image 

# # This function returns the image path
# def get_original_image():
#     original_image = set_original_image()
#     return original_image
    

def detect_and_draw_rectangles(image_path, size=N):
    global nuclei
    # Leitura do CSV
    df = pd.read_csv('classifications.csv')

    # Leitura da imagem
    img = Image.open(image_path)

    # Processando os núcleos
    nuclei = []
    for index, row in df.iterrows():
        if row['image_filename'] == os.path.basename(image_path):
            nucleus_x = row['nucleus_x']
            nucleus_y = row['nucleus_y']
            nucleus_id = row['cell_id']  # Adicionando a coluna 'cell_id' como ID do núcleo
            nuclei.append({'nucleus_id': nucleus_id, 'nucleus_x': nucleus_x, 'nucleus_y': nucleus_y})


    # Desenhando retângulos na imagem original
    draw = ImageDraw.Draw(img)
    drawn_rectangles = []

    for nucleus in nuclei:
        nucleus_x = nucleus['nucleus_x']
        nucleus_y = nucleus['nucleus_y']

        # Definindo a área de corte
        left = max(nucleus_x - size // 2, 0)
        top = max(nucleus_y - size // 2, 0)
        right = min(nucleus_x + size // 2, img.width)
        bottom = min(nucleus_y + size // 2, img.height)

        # Verificando se há sobreposição com retângulos já desenhados
        overlap = any(
            [((left < rect[2] and right > rect[0]) and (top < rect[3] and bottom > rect[1]))
             for rect in drawn_rectangles]
        )

        # Se não houver sobreposição, desenha o retângulo e adiciona à lista
        if not overlap:
            draw.rectangle([left, top, right, bottom], outline="red")
            drawn_rectangles.append((left, top, right, bottom))

    print("draw image teste: ", nuclei)
    return img, nuclei
    
def open_image():   
    global original_image, image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png;.jpg")])
    if file_path:
        image_path = file_path
        img = Image.open(image_path)

        if img.width > 500 or img.height > 500:
            img.thumbnail((500, 500))

        original_image = img
        display_image(img)

# This function will dispplay the image in the interface and apply the zoom
def display_image(image, zoom=1):
    # Aplicar o zoom na imagem
    width, height = image.size
    # Verifica a versão da Pillow e escolhe o método de redimensionamento apropriado
    zoomed_image = image.resize((int(width * zoom), int(height * zoom)), Image.Resampling.LANCZOS)

    tk_image = ImageTk.PhotoImage(zoomed_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image
    image_label.pack(expand=True)

# This function updates and returns the zoom 
def update_zoom(event):
    global image_path, zoom_level
    zoom_level = zoom_slider.get()

    if image_path:
        image = Image.open(image_path)
        display_image(image, zoom=zoom_level)
    

# ----------------------- PART ONE -------------------------

# This function will read the csv and return the df 
def get_csv():

    df = pd.read_csv('classifications.csv')

    return df


# This function will create the subdirectorys of the classes contained in the df and return them
def get_classes():
    df = get_csv()
    classes = ['ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC', 'Negative for intraepithelial lesion']

    for _class in classes:
        os.makedirs(_class, exist_ok=True)
    
    return classes 


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

        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0

        area_cm = round(area / 100, 4)
        perimeter_cm = round(perimeter / 100, 4)
        circularity = round(circularity, 4)
        eccentricity = round(eccentricity, 4)
        compactness = round(compactness, 4)

        return area_cm, perimeter_cm, circularity, eccentricity, compactness
    else:
        return None, None, None, None, None
    
# This function calculates the distance between the nucleus center found and the values in the csv
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)



from tkinter import Toplevel

# ... (código existente)



def show_segmentation_results(segmented_images):
    results_window = Toplevel(root)
    results_window.title("Resultados de Segmentação")

    for nucleus_id, segmented_img in segmented_images:
        # Exibir cada imagem segmentada com o ID do núcleo
        tk_image = ImageTk.PhotoImage(segmented_img)
        img_label = tk.Label(results_window, image=tk_image)
        img_label.image = tk_image
        img_label.pack()

        # Adicionar rótulo com o ID do núcleo
        id_label = tk.Label(results_window, text=f"Núcleo ID: {nucleus_id}")
        id_label.pack()

def segment_nuclei():
    global original_image, nuclei
    print("a")
    print(nuclei)
    print('b')
    if original_image is not None and nuclei:
        
        # Segmentar cada núcleo
        segmented_images = []
        for nucleus in nuclei:
            nucleus_id = nucleus.get('nucleus_id', '')  # Obtém o ID do núcleo
            segmented_img = perform_segmentation(original_image, nucleus)
            segmented_images.append((nucleus_id, segmented_img))

        # Exibir resultados em uma nova janela
        show_segmentation_results(segmented_images)


def perform_segmentation(original_img, nucleus):
    nucleus_x, nucleus_y = nucleus['nucleus_x'], nucleus['nucleus_y']

    # Implemente sua lógica de segmentação aqui, por exemplo, usando limiarização
    # Você pode usar bibliotecas como OpenCV para realizar a segmentação

    # Exemplo: Limiarização
    img_array = np.array(original_img)
    segmented_img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)[1]

    segmented_img = Image.fromarray(segmented_img_array)
    return segmented_img



def plot_scatter_graph():
    # Lendo os dados da tabela
    area_values = []
    eccentricity_values = []
    class_colors = {
        'ASC-US': 'blue',
        'ASC-H': 'green',
        'LSIL': 'yellow',
        'HSIL': 'orange',
        'SCC': 'red',
        'Negative for intraepithelial lesion': 'black'
    }
    colors = []

    for child in results_table.get_children():
        data = results_table.item(child)['values']
        area_values.append(data[1])
        eccentricity_values.append(data[4])
        nucleus_class = data[6]  # Classe do núcleo armazenada na tabela
        colors.append(class_colors.get(nucleus_class))

    # Criando a figura do Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(area_values, eccentricity_values, c=colors)

    ax.set_xlabel('Área (cm²)')
    ax.set_ylabel('Excentricidade')
    ax.set_title('Gráfico de Dispersão das Características dos Núcleos')

    # Integrando o gráfico com Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)  # A 'root' é a janela principal do Tkinter
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

def create_results_table(root):
    columns = ('image_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'compactness')
    results_table = ttk.Treeview(root, columns=columns, show='headings')
    results_table.heading('image_id', text='ID da Imagem')
    results_table.heading('area', text='Área')
    results_table.heading('perimeter', text='Perímetro')
    results_table.heading('circularity', text='Circularidade')
    results_table.heading('eccentricity', text='Excentricidade')
    results_table.heading('compactness', text='Compacidade')
    return results_table

def update_results_table(table, image_id, area, perimeter, circularity, eccentricity, compactness, nucleus_class):
    # Adiciona a classe do núcleo como um item oculto no final
    table.insert('', tk.END, values=(
        image_id, 
        f"{area:.4f} cm²", 
        f"{perimeter:.4f} cm", 
        f"{circularity:.4f}", 
        f"{eccentricity:.4f}", 
        f"{compactness:.4f}",
        nucleus_class  # Classe do núcleo
    ))


from PIL import ImageDraw

from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import pandas as pd

def detect_and_draw_rectangles(image_path):
    global nuclei
    # Leitura do CSV
    df = pd.read_csv('classifications.csv')

    # Leitura da imagem
    img = Image.open(image_path)

    # Processando os núcleos
    
    for index, row in df.iterrows():
        if row['image_filename'] == os.path.basename(image_path):
            nucleus_x = row['nucleus_x']
            nucleus_y = row['nucleus_y']
            nuclei.append({'nucleus_x': nucleus_x, 'nucleus_y': nucleus_y})

            

    # Desenhando retângulos na imagem original
    draw = ImageDraw.Draw(img)
    drawn_rectangles = []

    for nucleus in nuclei:
        nucleus_x = nucleus['nucleus_x']
        nucleus_y = nucleus['nucleus_y']

        # Definindo a área de corte
        left = max(nucleus_x - N // 2, 0)
        top = max(nucleus_y - N // 2, 0)
        right = min(nucleus_x + N // 2, img.width)
        bottom = min(nucleus_y + N // 2, img.height)

        # Verificando se há sobreposição com retângulos já desenhados
        overlap = any(
            [((left < rect[2] and right > rect[0]) and (top < rect[3] and bottom > rect[1]))
             for rect in drawn_rectangles]
        )

        # Se não houver sobreposição, desenha o retângulo e adiciona à lista
        if not overlap:
            draw.rectangle([left, top, right, bottom], outline="red")
            drawn_rectangles.append((left, top, right, bottom))

    return img, nuclei






def update_rect_size():
    global image_path, N
    
    # Obtém o valor da entrada
    new_N = int(rect_size_entry.get())

    if image_path and new_N > 0:
        N = new_N
        image = Image.open(image_path)
        process_image()

# Modifica a função process_image para chamar a nova função
def process_image():
    global image_path

    if image_path:
        img, nuclei = detect_and_draw_rectangles(image_path)
        # Exibe a imagem na interface
        display_image(img)





# --------------------------- GRAFFIC INTERFACE ----------------------------------



def show_results_window(area, perimeter, circularity, eccentricity, compactness, distance_to_center):
    global N

    results_window = tk.Toplevel(root)
    results_window.title("Resultados")
    # Rótulos para os resultados
    area_label = tk.Label(results_window, text=f"Área: {area} milímetros")
    area_label.pack()

    perimeter_label = tk.Label(results_window, text=f"Perímetro: {perimeter} milímetros")
    perimeter_label.pack()

    circularity_label = tk.Label(results_window, text=f"Circularidade: {circularity}")
    circularity_label.pack()

    compactness_label = tk.Label(results_window, text=f"Compacidade: {compactness}")
    compactness_label.pack()

    eccentricity_label = tk.Label(results_window, text=f"Excentricidade: {eccentricity}")
    eccentricity_label.pack()

    distance_label = tk.Label(results_window, text=f"Distância para o centro: {distance_to_center}")
    distance_label.pack()

    
# def show_segmentation_results(segmented_images):
#     results_window = tk.Toplevel(root)
#     results_window.title("Resultados de Segmentação")

#     for nucleus_id, segmented_img in segmented_images:
#         # Exibir cada imagem segmentada com o ID do núcleo
#         tk_image = ImageTk.PhotoImage(segmented_img)
#         img_label = tk.Label(results_window, image=tk_image)
#         img_label.image = tk_image
#         img_label.pack()

#         # Adicionar rótulo com o ID do núcleo
#         id_label = tk.Label(results_window, text=f"Núcleo ID: {nucleus_id}")
#         id_label.pack()


# Label para exibir a imagem
image_label = tk.Label(root)
#image_label.pack(expand=True)

# Adicionando botões
button_frame = tk.Frame(root)
button_frame.pack()

open_button = tk.Button(button_frame, text="Abrir Imagem", command=open_image)
open_button.pack(side=tk.LEFT)

process_button = tk.Button(button_frame, text="Processar Imagem", command=lambda: process_image())
process_button.pack(side=tk.LEFT)

# Adiciona um rótulo e uma entrada para permitir que o usuário defina N
n_label = tk.Label(button_frame, text="Valor de N:")
n_label.pack(side=tk.LEFT)

# n_entry = tk.Entry(button_frame, textvariable=n_var)
# n_entry.pack(side=tk.LEFT)

# Adicionar um controle deslizante para o zoom
zoom_slider = tk.Scale(root, from_=1, to=5, orient=tk.HORIZONTAL, resolution=0.1, command=update_zoom)
zoom_slider.pack()

# Label para mostrar a distância
distance_label = tk.Label(root)

# results_table = create_results_table(root)
# results_table.pack(fill=tk.BOTH, expand=True)

# Label para mostrar a distância
distance_label = tk.Label(root)

results_table = create_results_table(root)
results_table.pack(fill=tk.BOTH, expand=True)


rect_size_label = tk.Label(button_frame, text="Tamanho dos Retângulos:")
rect_size_label.pack(side=tk.LEFT)

rect_size_entry = tk.Entry(button_frame, textvariable = N)
rect_size_entry.pack(side=tk.LEFT)

rect_size_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1, variable=N, command=update_rect_size)
rect_size_slider.pack()

n_label = tk.Label(button_frame, text="Valor de N:")
n_label.pack(side=tk.LEFT)

rect_size_entry = tk.Entry(button_frame)
rect_size_entry.pack(side=tk.LEFT)


segment_button = tk.Button(button_frame, text="Segmentar", command=segment_nuclei)
segment_button.pack(side=tk.LEFT)


# Adiciona um botão para atualizar o tamanho do retângulo
update_button = tk.Button(button_frame, text="Atualizar", command=update_rect_size)
update_button.pack(side=tk.LEFT)
#----------------------- GRAFFIC INTERFACE ------------------------


root.mainloop()