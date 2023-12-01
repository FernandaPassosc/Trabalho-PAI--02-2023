from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk

class ImageDisplay:
    def __init__(self, root, upload_callback=None):
        self.root = root
        self.image_path = None
        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.zoom_slider = None
        self.rect_size_entry = None
        self.upload_callback = upload_callback
        self.N = 100

    def create_buttons(self, button_frame, root):
        open_button = tk.Button(button_frame, text="Abrir Imagem", command=self.open_image)
        open_button.pack(side=tk.LEFT)

        # Configuração do slider de zoom
        self.zoom_slider = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, command=self.update_zoom)
        self.zoom_slider.set(1.0)  # Define o valor inicial do zoom
        self.zoom_slider.pack()

        update_button = tk.Button(button_frame, text="Atualizar", command=self.update_rect_size)
        update_button.pack(side=tk.RIGHT)

        self.rect_size_entry = tk.Entry(button_frame, textvariable = self.N)
        self.rect_size_entry.pack(side=tk.RIGHT)

        rect_size_label = tk.Label(button_frame, text="Tamanho dos Retângulos:")
        rect_size_label.pack(side=tk.RIGHT)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png;.jpg")])

        if file_path:
            self.image_path = file_path
            img = Image.open(self.image_path)

            if img.width > 500 or img.height > 500:
                img.thumbnail((500, 500))

            self.original_image = img
            self.display_image(img)
            self.on_upload_complete()

    def display_image(self, image, zoom=1):
        max_size = 700  # Tamanho máximo para largura e altura
        width, height = image.size

        # Redimensionar a imagem para manter a proporção e limitar ao tamanho máximo
        ratio = min(max_size/width, max_size/height)
        new_width, new_height = int(width * ratio), int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Aplicar zoom à imagem redimensionada
        zoomed_width, zoomed_height = int(new_width * zoom), int(new_height * zoom)
        zoomed_image = resized_image.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(zoomed_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image
        self.image_label.pack(expand=True)

    def update_zoom(self, zoom):
        zoom_level = self.zoom_slider.get()
        if self.image_path:
            img = Image.open(self.image_path)
            self.display_image(img, zoom=zoom_level)
    
    def update_rect_size(self):        
        # Obtém o valor da entrada
        new_N = int(self.rect_size_entry.get())

        if self.image_path and new_N > 0:
            N = new_N
            image = Image.open(self.image_path)
            self.process_image()

    def on_upload_complete(self):
        # Chama o callback com a informação necessária
        if self.upload_callback:
            self.upload_callback()

    def get_image_path(self):
        return self.image_path
    
    def get_zoom_slider(self):
        return self.zoom_slider