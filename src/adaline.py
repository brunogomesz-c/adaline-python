import numpy as np

class Adaline:
    def __init__(self, tamanho_entrada, taxa_aprendizado=0.01):
        """
        Inicializa o neurônio Adaline com pesos aleatórios pequenos e bias.
        """
        self.alpha = taxa_aprendizado
        # Inicializa os pesos com valores aleatórios pequenos entre -0.1 e 0.1
        self.w = np.random.uniform(-0.1, 0.1, tamanho_entrada)
        # Inicializa o bias também de forma aleatória pequena
        self.b = np.random.uniform(-0.1, 0.1)
        
    def calcular_saida_linear(self, x):
        """
        Calcula a soma ponderada (saída linear): y =_sum(w_i * x_i) + b
        """
        return np.dot(self.w, x) + self.b

    def ativacao_bipolar(self, y_linear):
        """
        Função degrau bipolar: retorna 1 se >= 0, senão retorna -1
        """
        return 1 if y_linear >= 0 else -1

    def treinar_amostra(self, x, t):
        """
        Aplica a Regra Delta para atualizar os pesos baseado no erro da saída linear.
        Retorna o erro quadrático da amostra.
        """
        # 1. Calcula a saída linear (antes da função de ativação)
        y = self.calcular_saida_linear(x)
        
        # 2. Calcula o erro (alvo esperado - saída linear obtida)
        erro = t - y
        
        # 3. Atualização dos pesos e do bias (Regra Delta)
        # w_novo = w_atual + alpha * (t - y) * x
        self.w = self.w + self.alpha * erro * x
        self.b = self.b + self.alpha * erro
        
        # Retorna o erro quadrático desta amostra: 0.5 * (t - y)^2
        return 0.5 * (erro ** 2)