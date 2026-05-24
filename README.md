# OCR com Rede Neural Adaline

Este projeto implementa um sistema clássico de Reconhecimento Óptico de Caracteres (OCR) para letras maiúsculas (A-Z) utilizando a arquitetura **Adaline (Adaptive Linear Neuron)**. O pipeline de Processamento Digital de Imagens (PDI) usa OpenCV, e a interface gráfica foi desenvolvida com CustomTkinter e Matplotlib.

O projeto foi baseado na metodologia do trabalho de **Oliveira e Braga (2012)**, estendendo o experimento para testar a capacidade de generalização da rede contra fontes tipográficas que não foram apresentadas durante o treinamento.

---

## Tecnologias Utilizadas

* **Python 3.10+**
* **OpenCV (`cv2`):** Escala de cinza, binarização inversa e redimensionamento.
* **CustomTkinter:** Interface gráfica moderna em modo escuro.
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

```

---

---

## Instalar e Rodar o Projeto

### 1. Abrir o Terminal no VS Code

Com a pasta do projeto aberta, abra o terminal usando o atalho **`Ctrl + '`** (ou vá em **Terminal > New Terminal**).

### 2. Instalar as Dependências

Dê o comando abaixo no terminal e aperte **Enter**:

```bash
pip install -r requirements.txt

```

### 3. Executar o Sistema

Para abrir a interface gráfica e aperte **Enter**:

```bash
python app.py

```

## Como Testar na Interface

1. **Ajustar Hiperparâmetros:** Na barra lateral, defina o número de Épocas (padrão recomendado: `40`) e a Taxa de Aprendizado $\alpha$ (padrão recomendado: `0.005`).
2. **Treinar:** Clique no botão **"Treinar Redes"** para rodar a Regra Delta nos 26 neurônios paralelos. O gráfico de convergência do MSE será gerado instantaneamente na tela.
3. **Carregar Imagem:** Assim que o treino terminar, o botão de upload ficará verde. Clique nele e a interface abrirá direto na pasta de testes. Selecione qualquer arquivo `.png` de letra na pasta `dados/teste`.
4. **Resultado:** O painel inferior direito exibirá imediatamente o caractere identificado pela rede neural.

---
