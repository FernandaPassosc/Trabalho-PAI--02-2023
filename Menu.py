import tkinter as tk
from ImageTreatment import ImageDisplay
from Characterizer import Characterizer
from Segmentation import Segmentor
from ResultsDisplayer import ResultsDisplayer

class MainApp:
    def __init__(self, root):
        self.root = root
        self.image_display = ImageDisplay(root, upload_callback=self.on_image_uploaded)
        self.results_display = ResultsDisplayer(root)
        self.segmentor = Segmentor(root, self.image_display, self.results_display, self.image_display.image_path, upload_callback=self.on_image_segmented)
        self.characterizer = Characterizer(root, self.results_display, self.segmentor)

        button_frame = tk.Frame(root)
        button_frame.pack()

        # Agora, cada classe é responsável por criar seus próprios botões
        self.image_display.create_buttons(button_frame, root)
        self.segmentor.create_buttons(button_frame, root)
        self.characterizer.create_buttons(button_frame, root)
    
    def on_image_uploaded(self):
        # Método chamado após o upload da imagem
        image_path = self.image_display.get_image_path() 

        self.segmentor.img_path = image_path
        self.segmentor.N = self.image_display.N
    
    def on_image_segmented(self):
        cropped_img = self.segmentor.cropped_img
        self.characterizer.nuclei = self.segmentor.nuclei
                
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
