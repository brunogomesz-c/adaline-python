import os
import string
from PIL import Image, ImageDraw, ImageFont

def gerar_dataset_letras():
    # Caminhos das pastas de destino (Agora usando treino e teste separadamente)
    pasta_treino = os.path.join("dados", "treino")
    pasta_teste = os.path.join("dados", "teste")
    
    os.makedirs(pasta_treino, exist_ok=True)
    os.makedirs(pasta_teste, exist_ok=True)
    
    letras = string.ascii_uppercase
    
    # Lista de fontes sugeridas pelo roteiro
    fontes_nomes = ["arial.ttf", "times.ttf", "calibri.ttf", "verdana.ttf", "cour.ttf"]
    tamanho_img = (64, 64)
    
    print("Iniciando a geração separada de imagens (Fontes conhecidas vs Desconhecidas)...")
    
    for letra in letras:
        for i, nome_fonte in enumerate(fontes_nomes):
            try:
                fonte = ImageFont.truetype(nome_fonte, 40)
            except IOError:
                fonte = ImageFont.load_default()
                
            # Criar imagem branca e desenhar letra em preto
            imagem = Image.new("L", tamanho_img, color=255)
            draw = ImageDraw.Draw(imagem)
            
            caixa_texto = draw.textbbox((0, 0), letra, font=fonte)
            largura_texto = caixa_texto[2] - caixa_texto[0]
            altura_texto = caixa_texto[3] - caixa_texto[1]
            x = (tamanho_img[0] - largura_texto) // 2
            y = (tamanho_img[1] - altura_texto) // 2
            draw.text((x, y), letra, fill=0, font=fonte)
            
            # --- ESTRATÉGIA DE SEPARAÇÃO ---
            # Fontes index 0 e 1 (Arial e Times New Roman) vão para o TREINO
            if i < 2:
                destino = pasta_treino
                sufixo = f"fonte_{i}_CONHECIDA"
            # Fontes index 2, 3 e 4 (Calibri, Verdana e Courier) vão para o TESTE
            else:
                destino = pasta_teste
                sufixo = f"fonte_{i}_DESCONHECIDA"
            
            # Salva a imagem base
            imagem.save(os.path.join(destino, f"{letra}_{sufixo}_normal.png"))
            
            # Se for para o treino, salvamos também as rotações para ajudar a rede
            if i < 2:
                img_rot1 = imagem.rotate(10, fillcolor=255)
                img_rot1.save(os.path.join(destino, f"{letra}_{sufixo}_rot10.png"))
                img_rot2 = imagem.rotate(-10, fillcolor=255)
                img_rot2.save(os.path.join(destino, f"{letra}_{sufixo}_rotN10.png"))
                
    print("Sucesso! Dataset dividido perfeitamente.")
    print(f"-> Pasta de Treino ({pasta_treino}): Arial e Times New Roman")
    print(f"-> Pasta de Teste ({pasta_teste}): Calibri, Verdana e Courier")

if __name__ == "__main__":
    gerar_dataset_letras()