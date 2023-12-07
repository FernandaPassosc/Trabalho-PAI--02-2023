import tkinter as tk
from ImageTreatment import ImageDisplay
from Characterizer import Characterizer
from Segmentation import Segmentor
from ResultsDisplayer import ResultsDisplayer
import os

class MainApp:
    def __init__(self, root):
        self.root = root
        self.image_display = ImageDisplay(root, upload_callback=self.on_image_uploaded, rect_size_callback=self.handle_rect_size_change)
        self.results_display = ResultsDisplayer(root)
        self.segmentor = Segmentor(root, self.image_display, self.results_display, self.image_display.image_path, upload_callback=self.on_image_segmented)
        self.characterizer = Characterizer(root, self.results_display, self.segmentor)
        self.characterizer.setup_results_displayer()

        menu_frame = tk.LabelFrame(root, text='Menu de Opções', labelanchor='n', padx=10, background="pink")
        menu_frame.pack(pady=20)

        # Agora, cada classe é responsável por criar seus próprios botões
        self.image_display.create_buttons(menu_frame, root)
        self.segmentor.create_buttons(menu_frame, root)
        self.characterizer.create_buttons(menu_frame, root)
        self.results_display.create_buttons(menu_frame, root)
    
    def on_image_uploaded(self):
        # Método chamado após o upload da imagem
        image_path = self.image_display.get_image_path() 

        self.segmentor.img_path = image_path
    
    def handle_rect_size_change(self, new_size):
        self.segmentor.N = new_size
    
    def on_image_segmented(self):
        self.characterizer.nuclei = self.segmentor.nuclei
        self.image_display.image_path = self.segmentor.img_path
    
    def creat_classes(self):
        classes = ['ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC', 'Negative for intraepithelial lesion']
        for cls in classes:
            try:
                os.makedirs(cls, exist_ok=True)
            except Exception as e:
                print(f"Erro ao criar o diretório {cls}: {e}")
                
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    app.creat_classes()
    root.mainloop()
