import cv2
import numpy as np

def preprocessar_imagem(caminho_imagem, tamanho_padrao=(20, 20)):
    """
    Executa o pipeline de processamento de imagem exigido pelo artigo:
    1. Escala de cinza
    2. Binarização (Thresholding)
    3. Inversão de cores (Fundo 0, Letra 1)
    4. Redimensionamento
    5. Vetorização (.flatten())
    """
    # Leitura da imagem em escala de cinza diretamente
    img_gray = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")
        
    # Transforma pixels
    # THRESH_BINARY_INV para reconhecer o fundo branco e a letra branca
    # transforma o fundo em 0 e a letra em 255
    _, img_binaria = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # redimensionamento 20x20
    img_redimensionada = cv2.resize(img_binaria, tamanho_padrao, interpolation=cv2.INTER_AREA)
    
    # normalização para valores 0 e 1
    # Convertendo para float e dividindo por 255, onde tem caractere vira 1.0 e fundo vira 0.0
    img_normalizada = (img_redimensionada / 255.0).astype(np.float32)
    
    # Vetorização: Transforma a matriz bidimensional em um vetor unidimensional
    vetor_caracteristicas = img_normalizada.flatten()
    
    return vetor_caracteristicas

if __name__ == "__main__":
    
    import os
    
    pasta_treino = os.path.join("dados", "treino")
    if os.path.exists(pasta_treino):
        # Filtra apenas os arquivos que são imagens PNG reais
        arquivos_img = [f for f in os.listdir(pasta_treino) if f.lower().endswith('.png')]
        
        if arquivos_img:
            primeira_img = os.path.join(pasta_treino, arquivos_img[0])
            print(f"Testando processamento na imagem real: {primeira_img}")
            
            vetor = preprocessar_imagem(primeira_img, tamanho_padrao=(20, 20))
            print(f"Tamanho do vetor gerado: {vetor.shape}") # Deve mostrar (400,)
            print("Amostra do vetor (primeiros 20 elementos):")
            print(vetor[:20])
        else:
            print("Nenhuma imagem .png foi encontrada dentro de dados/treino/. Certifique-se de rodar: python src/extracao.py")
    else:
        print("A pasta dados/treino/ não existe.")