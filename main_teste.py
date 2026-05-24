import os
import sys

sys.path.append(os.path.abspath("src"))

from extracao import gerar_dataset_letras
from treino import treinar_sistema_adaline

# 1. Garante que as imagens existem
pasta_treino = os.path.join("dados", "treino")
if not os.path.exists(pasta_treino) or len(os.listdir(pasta_treino)) == 0:
    print("Gerando dataset de imagens...")
    gerar_dataset_letras()

print("\n INICIANDO ROTINA DE TREINAMENTO DA REDE ADALINE ")
# Executa o treino por 30 épocas
redes_treinadas, historico_erros = treinar_sistema_adaline(epocas=30, taxa_aprendizado=0.005)

print("\nPrimeiro erro (Época 1):", historico_erros[0])
print("Último erro (Época 30):", historico_erros[-1])

if historico_erros[-1] < historico_erros[0]:
    print("\n SUCESSO COMPLETO! O erro diminuiu. Os neurônios Adaline estão aprendendo com a Regra Delta!")
else:
    print("\n O erro não diminuiu. Verifique os parâmetros.")