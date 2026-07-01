import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# np.random.seed(42)


# 1. Gerando os dados

n = 100

x = np.random.uniform(
    low=-10,
    high=10,
    size=(n, 1)
)

ruido = np.random.normal(
    loc=0,
    scale=2,
    size=(n, 1)
)

y = 3 * x + 5 + ruido


# 2. Separando treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Regressão linear por pseudo-inversa

quantidade_treino = x_treino.shape[0]

coluna_de_uns_treino = np.ones((quantidade_treino, 1))

X_treino = np.hstack((coluna_de_uns_treino, x_treino))

theta = np.linalg.pinv(X_treino) @ y_treino

b_pinv = theta[0][0]
w_pinv = theta[1][0]

print("Pseudo-inversa")
print("b =", b_pinv)
print("w =", w_pinv)


# 4. Avaliação da pseudo-inversa

quantidade_teste = x_teste.shape[0]

coluna_de_uns_teste = np.ones((quantidade_teste, 1))

X_teste = np.hstack((coluna_de_uns_teste, x_teste))

y_pred_pinv = X_teste @ theta

erros_pinv = y_teste - y_pred_pinv
erros_quadrados_pinv = erros_pinv ** 2
mse_pinv = np.mean(erros_quadrados_pinv)

print()
print("MSE pseudo-inversa:", mse_pinv)


# 5. Rede neural manual com uma camada e SGD

np.random.seed(42)

w_sgd = np.random.randn()
b_sgd = np.random.randn()

taxa_aprendizado = 0.001
epocas = 300

historico_loss = []

for epoca in range(epocas):

    indices = np.arange(x_treino.shape[0])
    np.random.shuffle(indices)

    for i in indices:
        xi = x_treino[i][0]
        yi = y_treino[i][0]

        y_pred = w_sgd * xi + b_sgd

        erro = y_pred - yi

        grad_w = 2 * erro * xi
        grad_b = 2 * erro

        w_sgd = w_sgd - taxa_aprendizado * grad_w
        b_sgd = b_sgd - taxa_aprendizado * grad_b

    y_pred_treino = w_sgd * x_treino + b_sgd
    erros_treino = y_pred_treino - y_treino
    mse_epoca = np.mean(erros_treino ** 2)

    historico_loss.append(mse_epoca)

    # if epoca % 50 == 0:
    #     print("Época:", epoca, "MSE:", mse_epoca)


print()
print("Rede neural manual com SGD")
print("b =", b_sgd)
print("w =", w_sgd)


# 6. Avaliação da rede neural

y_pred_sgd = w_sgd * x_teste + b_sgd

erros_sgd = y_pred_sgd - y_teste
erros_quadrados_sgd = erros_sgd ** 2
mse_sgd = np.mean(erros_quadrados_sgd)

print()
print("MSE rede neural manual:", mse_sgd)


# 7. Gráfico dos dados

plt.scatter(x_treino, y_treino, label="Treino")
plt.scatter(x_teste, y_teste, label="Teste")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Dados sintéticos")
plt.legend()

plt.savefig("dados_sinteticos.png", dpi=300, bbox_inches="tight")
plt.close()


# 8. Gráfico comparando os modelos

x_linha = np.linspace(-10, 10, 100).reshape(-1, 1)

coluna_de_uns_linha = np.ones((x_linha.shape[0], 1))
X_linha = np.hstack((coluna_de_uns_linha, x_linha))

y_linha_pinv = X_linha @ theta
y_linha_sgd = w_sgd * x_linha + b_sgd

plt.scatter(x_teste, y_teste, label="Dados de teste")
plt.plot(x_linha, y_linha_pinv, label="Pseudo-inversa")
plt.plot(x_linha, y_linha_sgd, label="Rede neural com SGD")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparação dos modelos")
plt.legend()

plt.savefig("comparacao_modelos.png", dpi=300, bbox_inches="tight")
plt.close()


# 9. Gráfico da evolução do erro da rede neural

plt.plot(range(epocas), historico_loss)

plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Evolução da perda no treinamento")

plt.savefig("loss_sgd.png", dpi=300, bbox_inches="tight")
plt.close()