import os
import sys
import string
from tkinter import filedialog
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")

# acesso à pasta src
sys.path.append(os.path.abspath("src"))

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from treino import treinar_sistema_adaline
from processamento import preprocessar_imagem

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AppAdalineModerno(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Painel de Controle e Análise - Adaline OCR")
        self.geometry("900x550")
        self.resizable(False, False)
        
        self.redes = None
        self.caminho_imagem_carregada = None
        
        self.configurar_layout()
        
    def configurar_layout(self):
        # PAINEL ESQUERDO: CONTROLES E SELEÇÃO
        self.frame_esquerdo = ctk.CTkFrame(self, width=280)
        self.frame_esquerdo.pack(side="left", fill="both", padx=15, pady=15)
        self.frame_esquerdo.pack_propagate(False)
        
        lbl_cfg = ctk.CTkLabel(self.frame_esquerdo, text="Hiperparâmetros", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_cfg.pack(pady=10)
        
        # Épocas
        lbl_epocas = ctk.CTkLabel(self.frame_esquerdo, text="Número de Épocas:")
        lbl_epocas.pack(anchor="w", padx=20)
        self.txt_epocas = ctk.CTkEntry(self.frame_esquerdo)
        self.txt_epocas.insert(0, "40")
        self.txt_epocas.pack(fill="x", padx=20, pady=5)
        
        #Taxa de Aprendizado
        lbl_alpha = ctk.CTkLabel(self.frame_esquerdo, text="Taxa de Aprendizado (α):")
        lbl_alpha.pack(anchor="w", padx=20)
        self.txt_alpha = ctk.CTkEntry(self.frame_esquerdo)
        self.txt_alpha.insert(0, "0.005")
        self.txt_alpha.pack(fill="x", padx=20, pady=5)
        
        # Botão Treinar
        self.btn_treinar = ctk.CTkButton(self.frame_esquerdo, text="Treinar Redes", fg_color="#2c82c9", font=ctk.CTkFont(weight="bold"), command=self.executar_treinamento)
        self.btn_treinar.pack(fill="x", padx=20, pady=15)
        
        self.lbl_div = ctk.CTkLabel(self.frame_esquerdo, text="----------------------------------------", text_color="gray")
        self.lbl_div.pack()
        
        self.btn_upload = ctk.CTkButton(self.frame_esquerdo, text="Subir Foto da Letra", fg_color="gray", state="disabled", command=self.carregar_foto)
        self.btn_upload.pack(fill="x", padx=20, pady=10)
        
        self.lbl_preview_img = ctk.CTkLabel(self.frame_esquerdo, text="[ Sem Imagem ]", width=120, height=120, fg_color="#222222", corner_radius=8)
        self.lbl_preview_img.pack(pady=10)
        
        # PAINEL DIREITO: MONITOR GRÁFICO MSE E RESULTADO
        self.frame_direito = ctk.CTkFrame(self)
        self.frame_direito.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        
        self.lbl_grafico_title = ctk.CTkLabel(self.frame_direito, text="Convergência da Rede (Erro Quadrático Médio por Época)", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_grafico_title.pack(pady=10)
        
        self.frame_plot = ctk.CTkFrame(self.frame_direito, fg_color="#1e1e1e")
        self.frame_plot.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.frame_resultado = ctk.CTkFrame(self.frame_direito, height=100, fg_color="#1a1a1a")
        self.frame_resultado.pack(fill="x", padx=15, pady=15)
        
        self.lbl_res_label = ctk.CTkLabel(self.frame_resultado, text="Letra Identificada pelo Adaline:", font=ctk.CTkFont(size=14))
        self.lbl_res_label.pack(side="left", padx=30, pady=20)
        
        self.lbl_letra_final = ctk.CTkLabel(self.frame_resultado, text="-", font=ctk.CTkFont(size=54, weight="bold"), text_color="#2c82c9")
        self.lbl_letra_final.pack(side="right", padx=50, pady=5)

    def executar_treinamento(self):
        try:
            epocas = int(self.txt_epocas.get())
            alpha = float(self.txt_alpha.get())
        except ValueError:
            self.lbl_grafico_title.configure(text="Erro: Insira valores válidos!", text_color="red")
            return
            
        self.btn_treinar.configure(text="Treinando...", state="disabled")
        self.update()
        
        self.redes, historico_mse = treinar_sistema_adaline(epocas=epocas, taxa_aprendizado=alpha)
        
        for child in self.frame_plot.winfo_children():
            child.destroy()
            
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#1e1e1e")
        ax.plot(range(1, epocas + 1), historico_mse, color="#2c82c9", linewidth=2, marker='o', markersize=4)
        ax.set_facecolor("#1e1e1e")
        ax.set_xlabel("Época", color="white", fontsize=9)
        ax.set_ylabel("MSE", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#333333", linestyle="--")
        
        for spine in ax.spines.values():
            spine.set_color("#444444")
            
        fig.tight_layout()
        
        canvas_plot = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        
        self.btn_upload.configure(state="normal", fg_color="#2cc7c9")
        self.btn_treinar.configure(text="Treinar Redes", state="normal")
        self.lbl_grafico_title.configure(text="Convergência da Rede (Erro Quadrático Médio por Época)", text_color="white")

    def carregar_foto(self):
        caminho = filedialog.askopenfilename(
            initialdir="dados/teste",  # Abre direto nas fontes desconhecidas
            title="Selecione uma Letra com Fonte Desconhecida",
            filetypes=(("Imagens PNG", "*.png"), ("Todos os arquivos", "*.*"))
        )
        
        if caminho:
            self.caminho_imagem_carregada = caminho
            img = Image.open(caminho).resize((110, 110))
            img_tk = ImageTk.PhotoImage(img)
            self.lbl_preview_img.configure(image=img_tk, text="")
            self.lbl_preview_img.image = img_tk
            self.classificar_imagem()

    def classificar_imagem(self):
        if not self.redes or not self.caminho_imagem_carregada:
            return
        vetor_entrada = preprocessar_imagem(self.caminho_imagem_carregada, tamanho_padrao=(20, 20))
        letra_vencedora = None
        maior_ativacao = -float('inf')
        
        for letra, neuronio in self.redes.items():
            saida_linear = neuronio.calcular_saida_linear(vetor_entrada)
            if saida_linear > maior_ativacao:
                maior_ativacao = saida_linear
                letra_vencedora = letra
                
        self.lbl_letra_final.configure(text=letra_vencedora)

if __name__ == "__main__":
    app = AppAdalineModerno()
    app.mainloop()