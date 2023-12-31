import tkinter as tk
from ResultsDisplayer import ResultsDisplayer
import cv2
import numpy as np
from PIL import Image, ImageTk

class Characterizer:
    def __init__(self, root, results_displayer, segmentor):
        self.root = root
        self.results_displayer = results_displayer
        self.segmentor = segmentor
        self.bethesda_var = tk.StringVar()
        self.bethesda_var.set("Bethesda: ")
        self.nuclei = None
        self.table_img=None

        # Adiciona uma variável de controle para o widget de entrada
        n_var = tk.StringVar()
        n_var.set(str(100))  # Inicializa o widget de entrada com o valor padrão

    def create_buttons(self, menu_frame, root):
        self.table_img = Image.open("table.jpeg")
        self.table_img = self.table_img.resize((20, 20))
        self.table_img = ImageTk.PhotoImage(self.table_img)

        characterize_button = tk.Button(
            menu_frame,
            text="Caracterizar Núcleos",
            command=self.characterize,
            background="#FEDBDC",
            activebackground="#FADCD2",
            image=self.table_img,
            compound=tk.LEFT  # Place the image to the left of the text
        )
        characterize_button.grid(row=7, column=0, pady=5, sticky='n', columnspan=2)

    def characterize(self):
        for one_nuclei in self.nuclei:
            # Carrega a imagem do caminho especificado
            with Image.open(one_nuclei['nucleus_path']) as img:
                contours = one_nuclei['nucleus_contour']
                # Calcular descritores de forma para cada imagem
                area, perimeter, circularity, eccentricity, compactness = self.calculate_shape_descriptors(contours)

                if area is not None and perimeter is not None:
                    #print(f"Imagem: {one_nuclei['nucleus_path']}")
                    #print(f"Área: {area} milímetros, Perímetro: {perimeter} milímetros, Circularidade: {circularity}, Excentricidade: {eccentricity}, Compacidade: {compactness}")
                    
                    # Aqui, assumindo que você quer calcular a distância para o centro da imagem
                    new_x = img.width // 2
                    new_y = img.height // 2

                    distance_to_center = self.calculate_distance(
                        one_nuclei['nucleus_x'], one_nuclei['nucleus_y'],
                        new_x, new_y
                    )

                    distance_to_center = round(distance_to_center / 100, 4)

                    # Atualize a tabela de resultados com as informações de cada imagem
                    self.results_displayer.update_results_table(one_nuclei['nucleus_id'], area, perimeter, circularity, eccentricity, compactness, distance_to_center, one_nuclei['nucleus_class'])                                

    def calculate_shape_descriptors(self, contours):
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

    def calculate_distance(self, x1, y1, x2, y2):
        return round(np.sqrt((x1 - x2)**2 + (y1 - y2)**2), 4)    
    
    def setup_results_displayer(self):
        # Configure o callback de exibição no ResultsDisplayer
        self.results_displayer.set_display_callback(self.display_results)
    
    def display_results(self, area, perimeter, circularity, eccentricity, compactness, distance_to_center, nucleus_class):
        self.results_displayer.show_results_window(area, perimeter, circularity, eccentricity, compactness, distance_to_center, nucleus_class)