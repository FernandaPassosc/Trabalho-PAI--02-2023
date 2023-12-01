import pandas as pd
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont

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
        self.N = None
        self.nuclei = []

    def create_buttons(self, button_frame, root):
        process_button = tk.Button(button_frame, text="Segmentar Núcleos", command=self.detect_and_draw_rectangles)
        process_button.pack(side=tk.LEFT)        

        # Passando o parâmetro root ao criar a tabela de resultados
        results_table = self.results_display.create_results_table(root)

        results_table.pack(fill=tk.BOTH, expand=True)

    def segment_nuclei(self, show_class=False):
        if self.img_path:
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
                if row['image_filename'] == os.path.basename(self.img_path):
                    self.nucleus_id = row['cell_id']
                    self.img_class = row['bethesda_system']
                    self.image_id = row['image_id']
                    self.nucleus_x = row['nucleus_x']
                    self.nucleus_y = row['nucleus_y']

                    with Image.open(self.img_path) as img:
                        # Definindo a área de corte
                        left = max(self.nucleus_x - half_size, 0)
                        top = max(self.nucleus_y - half_size, 0)
                        right = min(self.nucleus_x + half_size, img.width)
                        bottom = min(self.nucleus_y + half_size, img.height)

                        # print(f"Núcleo: ({nucleus_x}, {nucleus_y})")
                        # print(f"Centro da imagem recortada: ({(left + right) / 2}, {(top + bottom) / 2})")

                        # Recortando a imagem
                        self.cropped_img = img.crop((left, top, right, bottom))

                        self.cropped_img.save(f"{self.img_class}/{self.cell_id}.png")

                        # Atualiza a imagem na interface
                        self.image_display.display_image(self.cropped_img)

                        self.on_segmentation_complete()

    def detect_and_draw_rectangles(self):
        # Leitura do CSV
        df = pd.read_csv('classifications.csv')

        # Leitura da imagem
        img = Image.open(self.img_path)

        # Processando os núcleos
        self.nuclei = []

        for index, row in df.iterrows():
            if row['image_filename'] == os.path.basename(self.img_path):
                self.img_class = row['bethesda_system']
                self.nucleus_id = row['cell_id']
                self.image_id = row['image_id']
                self.nucleus_x = row['nucleus_x']
                self.nucleus_y = row['nucleus_y']

                self.nuclei.append({'nucleus_id': self.nucleus_id, 
                               'nucleus_x': self.nucleus_x, 
                               'nucleus_y': self.nucleus_y, 
                               'nucleus_path': '', 
                               'nucleus_class': self.img_class, 
                               'image_id': self.image_id})

        # Desenhando retângulos na imagem original
        draw = ImageDraw.Draw(img)
        drawn_rectangles = []

        font = ImageFont.load_default()
        
        for nucleus in self.nuclei:
            self.nucleus_id = nucleus['nucleus_id']  # Adicionando a coluna 'cell_id' como ID do núcleo
            self.nucleus_x = nucleus['nucleus_x']
            self.nucleus_y = nucleus['nucleus_y']

            # Definindo a área de corte
            left = max(self.nucleus_x - self.N // 2, 0)
            top = max(self.nucleus_y - self.N // 2, 0)
            right = min(self.nucleus_x + self.N // 2, img.width)
            bottom = min(self.nucleus_y + self.N // 2, img.height)

            self.cropped_img = img.crop((left, top, right, bottom))

            self.cropped_img.save(f"{self.img_class}/{self.nucleus_id}.png")
            
            nucleus['nucleus_path'] = f"{self.img_class}/{self.nucleus_id}.png"

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
        
        self.image_display.display_image(img)

        self.on_segmentation_complete()
        
        return img, self.nuclei

    def on_segmentation_complete(self):
        # Chama o callback com a informação necessária
        if self.upload_callback:
            self.upload_callback()               