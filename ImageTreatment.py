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
        self.rect_size_entry = None
        self.upload_callback = upload_callback
        self.rect_size_callback = rect_size_callback  # Novo callback para o tamanho do retângulo
        self.N = 100
        self.zoom_factor = 1.0 

    def create_buttons(self, menu_frame, root):
        open_img = Image.open("picture.jpeg")
        open_img = open_img.resize((20, 20))  # Ajuste o tamanho conforme necessário
        open_img = ImageTk.PhotoImage(open_img)

        update_img = Image.open("image-update.jpeg")
        update_img = update_img.resize((20, 20))
        update_img = ImageTk.PhotoImage(update_img)

        size_img = Image.open("size.jpeg")
        size_img = size_img.resize((20, 20))
        size_img = ImageTk.PhotoImage(size_img)
        
        open_button = tk.Button(menu_frame, text="Abrir Imagem", command=self.open_image, background="#FEDBDC", foreground="black", activebackground="#FADCD2", image=open_img)
        open_button.grid(row=0, column=0, pady=5, sticky='n', columnspan=2)

        update_button = tk.Button(menu_frame, text="Atualizar", command=self.update_rect_size,  background="#FEDBDC",  foreground="black", activebackground="#FADCD2", image=update_img)
        update_button.grid(row=6, column=0, pady=5, sticky='n', columnspan=2)

        rect_size_entry = tk.Entry(menu_frame, textvariable=self.N, width=5, background="#FEDBDC")
        rect_size_entry.grid(row=2, column=1, pady=5, sticky='e')

        rect_size_label = tk.Label(menu_frame, text="Tamanho dos Retângulos:", width=30, background="#FEDBDC", foreground="black", image=size_img)
        rect_size_label.grid(row=2, column=0, pady=5, sticky='w')


        self.root.bind("<MouseWheel>", self.zoomer)  # Vincula o evento do mouse para simular o zoom

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

    def display_image(self, image):
        # Redimensiona a imagem somente quando ela é carregada pela primeira vez
        if not hasattr(self, 'original_image'):
            width, height = image.size
            if width > 500 or height > 500:
                image.thumbnail((500, 500))

            self.original_image = image

        # Redimensionar a imagem de acordo com o fator de zoom
        width, height = self.original_image.size
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        resized_image = self.original_image.resize((new_width, new_height))

        tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def zoomer(self, event):
        # A função que simula o zoom do touchpad
        if event.delta > 0:
            self.zoom_in()
        elif event.delta < 0:
            self.zoom_out()

    def zoom_in(self):
        self.zoom_factor *= 1.1  # Aumentar o fator de zoom
        if self.image_path:
            img = Image.open(self.image_path)
            self.display_image(img)

    def zoom_out(self):
        self.zoom_factor *= 0.9  # Reduzir o fator de zoom
        if self.image_path:
            img = Image.open(self.image_path)
            self.display_image(img)

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
