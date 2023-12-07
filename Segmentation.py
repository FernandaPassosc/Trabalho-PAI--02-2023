import pandas as pd
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import numpy as np

N = 100  # Valor padrão de N
class Segmentor:
    def __init__(self, root, image_display, results_display, img_path, upload_callback=None):
        self.root = root
        self.image_display = image_display
        self.results_display = results_display
        self.img_path = img_path
        self.upload_callback = upload_callback
        self.cropped_img = None
        self.img_class = None
        self.image_id = None
        self.nucleus_id = None
        self.nucleus_x = None
        self.nucleus_y = None
        self.N = N
        self.nuclei = []
        self.segment_img=None

    def create_buttons(self, menu_frame, root):
        self.segment_img = Image.open("segment.jpeg")
        self.segment_img = self.segment_img.resize((20, 20))
        self.segment_img = ImageTk.PhotoImage(self.segment_img)

        process_button = tk.Button(
            menu_frame,
            text="Segmentar Núcleos",
            command=self.detect_and_draw_rectangles,
            background="#FEDBDC",
            activebackground="#FADCD2",
            image=self.segment_img,
            compound=tk.LEFT  # Place the image to the left of the text
        )
        process_button.grid(row=1, column=0, pady=5, sticky='n', columnspan=2)

        # Passando o parâmetro root ao criar a tabela de resultados
        results_table = self.results_display.create_results_table(root)

        results_table.pack(fill=tk.BOTH, expand=True)

    def detect_and_draw_rectangles(self):
        # Leitura do CSV
        df = pd.read_csv('classifications.csv')

        # Leitura da imagem
        img = Image.open(self.img_path)

        image_to_draw = img.copy()

        # Desenhando retângulos na imagem original
        draw = ImageDraw.Draw(image_to_draw)
        drawn_rectangles = []

        font = font = ImageFont.truetype("arial.ttf", 20)

        for index, row in df.iterrows():
            if row['image_filename'] == os.path.basename(self.img_path):
                self.img_class = row['bethesda_system']
                self.nucleus_id = row['cell_id']
                self.image_id = row['image_id']
                self.nucleus_x = row['nucleus_x']
                self.nucleus_y = row['nucleus_y']

                # Definindo a área de corte
                left = max(self.nucleus_x - self.N // 2, 0)
                top = max(self.nucleus_y - self.N // 2, 0)
                right = min(self.nucleus_x + self.N // 2, img.width)
                bottom = min(self.nucleus_y + self.N // 2, img.height)

                # Adicionando a coluna 'cell_id' como ID do núcleo
                self.nuclei.append({
                    'nucleus_id': self.nucleus_id,
                    'nucleus_x': self.nucleus_x,
                    'nucleus_y': self.nucleus_y,
                    'nucleus_path': f"{self.img_class}/{self.nucleus_id}-segmented.png",
                    'nucleus_class': self.img_class,
                    'image_id': self.image_id,
                    'nucleus_contour': None
                })

                self.cropped_img = img.crop((left, top, right, bottom))
                self.cropped_img.save(f"{self.img_class}/{self.nucleus_id}.png")

                # Carregar a imagem
                image = cv2.imread(f"{self.img_class}/{self.nucleus_id}.png")

                image = np.array(image)
                
                # Converter para escala de cinza
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

                    # Aplicando a máscara na imagem original recortada
                    segmented_image = np.zeros_like(image)
                    for i in range(3):  # Aplicar a máscara em cada canal de cor
                        segmented_image[:,:,i] = image[:,:,i] & mask

                    # Salvar a imagem segmentada
                    segmented_path = os.path.join(self.img_class, f"{self.nucleus_id}-segmented.png")

                    cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))            

                self.nuclei[-1]['nucleus_contour'] = contours                

                # Verificando se há sobreposição com retângulos já desenhados
                overlap = any(
                    [((left < rect[2] and right > rect[0]) and (top < rect[3] and bottom > rect[1]))
                     for rect in drawn_rectangles]
                )

                # Se não houver sobreposição, desenha o retângulo e adiciona à lista
                if not overlap:
                    draw.rectangle([left, top, right, bottom], outline="red")
                    drawn_rectangles.append((left, top, right, bottom))
                    text_position = (right + 5, top)  # Ajuste a posição conforme necessário
                    draw.text(text_position, str(self.nucleus_id), fill="red", font=font)
        
        image_to_draw.save(f"./image-{self.image_id}-rects.png")

        self.img_path = f"image-{self.image_id}-rects.png"

        self.image_display.display_image_with_rects(image_to_draw)
        
        self.on_segmentation_complete()

        return image_to_draw, self.nuclei

    def on_segmentation_complete(self):
        # Chama o callback com a informação necessária
        if self.upload_callback:
            self.upload_callback()