# trabalho-final-ia
4 questões que correspondem ao trabalho final da disciplina de inteligência artificial

## Questões

### Questão 1: Regressão Linear
Implemente um modelo de regressão linear utilizando um conjunto de dados sintético gerado com a equação:

$$y = 3x + 5 + \epsilon$$

Onde $x$ segue uma distribuição uniforme entre -10 e 10, e $\epsilon$ é um ruído gaussiano com média zero e desvio padrão de 2. 

**Passos a realizar:**
1. Gere um conjunto de dados com pelo menos 100 pontos.
2. Divida os dados em treino (80%) e teste (20%).
3. Implemente modelos de regressão linear empregando:
   * A solução de mínimos quadrados (pseudo-inversa);
   * Uma rede neural com uma camada treinada via gradiente descendente utilizando *MSE-Loss* (Erro Quadrático Médio) e otimizador SGD.
4. Apresente as soluções para cada um dos métodos acima.
5. Avalie o desempenho dos modelos e visualize os resultados.

---

### Questão 2: Regressão Logística para Classificação Binária
Implemente um modelo de regressão logística para resolver um problema de classificação binária utilizando um conjunto de dados sintético.

**Passos a realizar:**
1. Utilize a função `make_classification` da biblioteca Scikit-Learn para gerar um conjunto de dados com 500 amostras, 2 variáveis preditoras e 2 classes.
2. Divida os dados em treino (70%) e teste (30%).
3. Implemente um modelo de regressão logística (i.e., rede neural com uma única camada de saída e ativação sigmoid).
4. Treine o modelo utilizando gradiente descendente (versão não-estocástica), conforme visto em sala.
5. Avalie a acurácia no conjunto de teste e visualize a fronteira de decisão do classificador.

---

### Questão 3: Classificação Binária com uma MLP e Seleção de Neurônios
Implemente uma rede neural do tipo MLP (Multi-Layer Perceptron) para a tarefa de classificação binária, utilizando um conjunto de validação para selecionar o número adequado de neurônios na camada oculta.

**Passos a realizar:**
1. Utilize a função `make_moons` da biblioteca Scikit-Learn para gerar um conjunto de dados com 500 amostras.
2. Divida os dados em treino (70%), validação (15%) e teste (15%).
3. Implemente uma MLP com:
   * Uma camada oculta com $n$ neurônios e ativação ReLU;
   * Uma camada de saída com 1 neurônio e ativação sigmoid;
   * Função de perda `BCELoss` e o otimizador Adam ou SGD.
4. Treine modelos com diferentes números de neurônios na camada oculta (exemplo: 5, 10, 20, 50).
5. Plote a evolução da função custo (*loss*) ao longo do treinamento (épocas).
6. Escolha o melhor número de neurônios com base na menor perda no conjunto de validação.
7. Avalie o modelo escolhido no conjunto de teste e visualize a fronteira de decisão.

---

### Questão 4: Classificação Multiclasse com MLP
Implemente uma rede neural para a classificação de imagens do conjunto MNIST (dígitos escritos à mão).

**Passos a realizar:**
1. Carregue o conjunto de dados MNIST utilizando Torchvision ou Keras.
2. Implemente uma rede neural MLP com:
   * Uma camada oculta de 64 neurônios e ativação ReLU;
   * Uma camada de saída com 10 neurônios e ativação softmax.
3. Utilize a função de perda *Cross Entropy Loss* e o otimizador Adam ou SGD.
4. Treine a rede por 10 épocas e avalie a acurácia no conjunto de teste.
5. Exiba algumas previsões feitas pelo modelo, mostrando imagens e suas respectivas classes previstas.
