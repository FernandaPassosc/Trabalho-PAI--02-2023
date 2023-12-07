from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
class ResultsDisplayer:
    def __init__(self, root):
        self.root = root         
        self.results_table = None

    def create_buttons(self, menu_frame, root):
        plot_img = Image.open("plot-graph.jpeg")
        plot_img = plot_img.resize((20, 20))
        plot_img = ImageTk.PhotoImage(plot_img)
        scatter_graph = tk.Button(menu_frame, text="Plotar Gráfico de Dispersão", command=self.plot_scatter_graph, background="#FEDBDC", image=plot_img)
        scatter_graph.grid(row=8, column=0, pady=5, sticky='n')              

    def create_results_table(self, root):
        columns = ('image_id', 'area', 'perimeter', 'circularity', 'eccentricity', 'compactness', 'distance_center')

        self.results_table = ttk.Treeview(root, columns=columns, show='headings')
        self.results_table.bind("<ButtonRelease-1>", self.on_table_click)

        self.results_table.heading('image_id', text='ID do Núcleo')
        self.results_table.heading('area', text='Área')
        self.results_table.heading('perimeter', text='Perímetro')
        self.results_table.heading('circularity', text='Circularidade')
        self.results_table.heading('eccentricity', text='Excentricidade')
        self.results_table.heading('compactness', text='Compacidade')
        self.results_table.heading('distance_center', text='Distance Until Center')

        return self.results_table
    
    def on_table_click(self, event):
        # Obter o item selecionado
        selected_item = self.results_table.focus()
        item_values = self.results_table.item(selected_item, "values")

        self.display_callback(item_values[1], item_values[2], item_values[3], item_values[4], item_values[5], item_values[6])
    
    def set_display_callback(self, callback):
        self.display_callback = callback

    def update_results_table(self, image_id, area, perimeter, circularity, eccentricity, compactness, distance_to_center, nucleus_class):
        # Adiciona a classe do núcleo como um item oculto no final
        self.results_table.insert('', tk.END, values=(
            image_id, 
            f"{area:.4f}", 
            f"{perimeter:.4f}", 
            f"{circularity:.4f}", 
            f"{eccentricity:.4f}", 
            f"{compactness:.4f}",
            f"{distance_to_center:.4f}",
            nucleus_class  # Classe do núcleo
        ))

    def show_results_window(self, area, perimeter, circularity, eccentricity, compactness, distance_to_center):
        results_window = tk.Toplevel(self.root)
        
        results_window.title("Resultados")
        # Rótulos para os resultados
        area_label = tk.Label(results_window, text=f"Área: {area} cm²")
        area_label.pack()

        perimeter_label = tk.Label(results_window, text=f"Perímetro: {perimeter} cm")
        perimeter_label.pack()

        circularity_label = tk.Label(results_window, text=f"Circularidade: {circularity} cm")
        circularity_label.pack()

        eccentricity_label = tk.Label(results_window, text=f"Excentricidade: {eccentricity} cm")
        eccentricity_label.pack()

        compactness_label = tk.Label(results_window, text=f"Compacidade: {compactness} cm")
        compactness_label.pack()

        distance_label = tk.Label(results_window, text=f"Distância para o centro: {distance_to_center} cm")
        distance_label.pack()
    
    def plot_scatter_graph(self):
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

        # Certifique-se de que há dados para plotar
        if not self.results_table.get_children():
            tk.messagebox.showinfo("Informação", "Não há dados para plotar.")
            return

        for child in self.results_table.get_children():
            data = self.results_table.item(child)['values']
            area_values.append(float(data[1]))
            eccentricity_values.append(float(data[4]))

            # A classe do núcleo é o oitavo item; os índices começam em 0
            nucleus_class = data[7]
            colors.append(class_colors.get(nucleus_class))  # Cor padrão

        fig, ax = plt.subplots(figsize=(8, 6)) 

        # Criando o gráfico de dispersão
        ax.scatter(area_values, eccentricity_values, c=colors)
        ax.set_xlabel('Área')
        ax.set_ylabel('Excentricidade')
        ax.set_title('Gráfico de Dispersão das Características dos Núcleos')

        # Criando uma legenda para as cores
        handles = [plt.Line2D([0], [0], marker='o', color=color, label=class_name, linestyle="None") 
                   for class_name, color in class_colors.items()]
        # Ajustar a posição da legenda para fora do gráfico
        ax.legend(handles=handles, title="Classes de Núcleos", loc='upper left', bbox_to_anchor=(1, 1), fontsize = 'small')
        
        # Ajustar os limites do eixo e a posição da legenda
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Criando uma nova janela para exibir o gráfico
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Gráfico de Dispersão")

        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()
