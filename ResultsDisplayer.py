from tkinter import ttk
import tkinter as tk

class ResultsDisplayer:
    def __init__(self, root):
        self.root = root         
        self.results_table = None          

    def create_results_table(self, root):
        columns = ('image_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'compactness')

        self.results_table = ttk.Treeview(root, columns=columns, show='headings')

        self.results_table.heading('image_id', text='ID do Núcleo')
        self.results_table.heading('area', text='Área')
        self.results_table.heading('perimeter', text='Perímetro')
        self.results_table.heading('circularity', text='Circularidade')
        self.results_table.heading('eccentricity', text='Excentricidade')
        self.results_table.heading('compactness', text='Compacidade')

        return self.results_table
    
    def update_results_table(self, table, image_id, area, perimeter, circularity, eccentricity, compactness, nucleus_class):
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

    def show_results_window(self, area, perimeter, circularity, eccentricity, compactness, distance_to_center):
        results_window = tk.Toplevel(self.root)
        
        results_window.title("Resultados")
        # Rótulos para os resultados
        area_label = tk.Label(results_window, text=f"Área: {area:.4f} cm²")
        area_label.pack()

        perimeter_label = tk.Label(results_window, text=f"Perímetro: {perimeter} cm")
        perimeter_label.pack()

        circularity_label = tk.Label(results_window, text=f"Circularidade: {circularity} cm")
        circularity_label.pack()

        compactness_label = tk.Label(results_window, text=f"Compacidade: {compactness} cm")
        compactness_label.pack()

        eccentricity_label = tk.Label(results_window, text=f"Excentricidade: {eccentricity} cm")
        eccentricity_label.pack()

        distance_label = tk.Label(results_window, text=f"Distância para o centro: {distance_to_center} cm")
        distance_label.pack()