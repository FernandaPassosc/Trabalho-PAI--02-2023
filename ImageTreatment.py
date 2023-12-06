from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk

class ImageDisplay:
    def __init__(self, root, upload_callback=None, rect_size_callback=None):
        self.root = root
        self.image_path = None
        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.zoom_slider = None
        self.rect_size_entry = None
        self.upload_callback = upload_callback
        self.rect_size_callback = rect_size_callback  # Novo callback para o tamanho do retângulo
        self.N = 100

    def create_buttons(self, button_frame, root):
        open_button = tk.Button(button_frame, text="Abrir Imagem", command=self.open_image)
        open_button.pack(side=tk.LEFT)

        style = ttk.Style(root)
        style.configure("Zoom.TScale", sliderlength=20, troughcolor="#f0f0f0", background="#0078D7")

        # Criar o slider de zoom com estilo personalizado
        self.zoom_slider = tk.Scale(root, from_=0.1, to=2.0, orient="horizontal", resolution=0.1, command=self.update_zoom)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack()

        # Adicionar rótulo para mostrar o valor atual do zoom
        self.zoom_label = tk.Label(root, text=f"Zoom: {self.zoom_slider.get()}x")
        self.zoom_label.pack()

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

    def update_zoom(self, zoom_value):
        zoom_level = float(zoom_value)  # Convert the zoom_value to a float
        if self.image_path:
            img = Image.open(self.image_path)
            self.display_image(img, zoom=zoom_level)
    
    def update_rect_size(self):
        # Obtém o valor da entrada
        new_N = int(self.rect_size_entry.get())

        if self.image_path and new_N > 0:
            self.N = new_N  # Atualiza o valor de N
            self.on_rect_size_change()  # Chama o novo callback

    def on_rect_size_change(self):
        # Chama o novo callback com a informação necessária
        if self.rect_size_callback:
            self.rect_size_callback(self.N)

    def on_upload_complete(self):
        # Chama o callback com a informação necessária
        if self.upload_callback:
            self.upload_callback()

    def get_image_path(self):
        return self.image_path
    
    def get_zoom_slider(self):
        return self.zoom_slider