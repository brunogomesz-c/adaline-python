# OCR com Rede Neural Adaline

Este projeto implementa um sistema clássico de Reconhecimento Óptico de Caracteres (OCR) para letras maiúsculas (A-Z) utilizando a arquitetura **Adaline (Adaptive Linear Neuron)**. O pipeline de Processamento Digital de Imagens (PDI) usa OpenCV, e a interface gráfica foi feita com CustomTkinter e Matplotlib.

O projeto foi baseado na metodologia do trabalho de **Oliveira e Braga (2012)**, estendendo o experimento para testar a capacidade de generalização da rede contra fontes tipográficas que não foram apresentadas durante o treinamento.

---

## Tecnologias Utilizadas

* **Python 3.10+**
* **OpenCV (`cv2`):** Escala de cinza, binarização inversa e redimensionamento.
* **CustomTkinter:** Interface gráfica em modo escuro.
* **Matplotlib:** Gráfico em tempo real do Erro Quadrático Médio (MSE).
* **Pillow (`PIL`):** Manipulação e leitura das imagens do dataset.

---

## Estrutura de Pastas

```text
adaline_caracteres/
│
├── dados/
│   ├── treino/           # Fontes conhecidas (Arial e Times New Roman)
│   └── teste/            # Fontes desconhecidas (Calibri, Verdana e Courier)
│
├── src/
│   ├── adaline.py        # Algoritmo do neurônio Adaline e Regra Delta
│   ├── treino.py         # Treinamento multi-classe (One-vs-All)
│   └── processamento.py  # Pipeline de PDI (Binarização e Vetorização)
│
├── app.py                # Interface gráfica e execução principal
└── README.md             # Documentação do projeto