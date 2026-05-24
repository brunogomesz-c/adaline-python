import os
import string
import numpy as np
from processamento import preprocessar_imagem
from adaline import Adaline

def treinar_sistema_adaline(epocas=40, taxa_aprendizado=0.005):
    pasta_treino = os.path.join("dados", "treino")
    letras = string.ascii_uppercase
    
    arquivos_img = [f for f in os.listdir(pasta_treino) if f.lower().endswith('.png')]
    
    X_dados, Y_letras_reais = [], []
    for arquivo in arquivos_img:
        caminho = os.path.join(pasta_treino, arquivo)
        X_dados.append(preprocessar_imagem(caminho, tamanho_padrao=(20, 20)))
        Y_letras_reais.append(arquivo.split('_')[0])
        
    X_dados = np.array(X_dados)
    tamanho_vetor = 400
    redes_adaline = {letra: Adaline(tamanho_entrada=tamanho_vetor, taxa_aprendizado=taxa_aprendizado) for letra in letras}
    
    historico_mse = []
    
    for epoca in range(epocas):
        erro_acumulado = 0.0
        for x_amostra, letra_real in zip(X_dados, Y_letras_reais):
            for letra_neuronio, neuronio in redes_adaline.items():
                # Força a taxa de aprendizado dinamicamente atualizada pela interface
                neuronio.alpha = taxa_aprendizado
                alvo = 1.0 if letra_neuronio == letra_real else -1.0
                erro_acumulado += neuronio.treinar_amostra(x_amostra, alvo)
                
        mse = erro_acumulado / (len(X_dados) * len(letras))
        historico_mse.append(mse)
        
    return redes_adaline, historico_mse