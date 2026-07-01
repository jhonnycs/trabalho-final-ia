import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(42)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Carregamento dos dados

transformacao = transforms.ToTensor()

dados_treino = torchvision.datasets.MNIST(
    root="./dados",
    train=True,
    download=True,
    transform=transformacao
)

dados_teste = torchvision.datasets.MNIST(
    root="./dados",
    train=False,
    download=True,
    transform=transformacao
)


# 2. DataLoaders

batch_size = 64

loader_treino = torch.utils.data.DataLoader(
    dados_treino,
    batch_size=batch_size,
    shuffle=True
)

loader_teste = torch.utils.data.DataLoader(
    dados_teste,
    batch_size=batch_size,
    shuffle=False
)


# 3. Modelo MLP

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.camada_oculta = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.camada_saida = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), 784)

        x = self.camada_oculta(x)
        x = self.relu(x)
        x = self.camada_saida(x)

        return x


modelo = MLP()
modelo = modelo.to(dispositivo)


# 4. Função de perda e otimizador

criterio = nn.CrossEntropyLoss()

otimizador = optim.Adam(
    modelo.parameters(),
    lr=0.001
)


# 5. Treinamento

epocas = 10

historico_loss = []

for epoca in range(epocas):

    modelo.train()

    soma_loss = 0

    for imagens, rotulos in loader_treino:

        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        saidas = modelo(imagens)

        loss = criterio(saidas, rotulos)

        otimizador.zero_grad()
        loss.backward()
        otimizador.step()

        soma_loss = soma_loss + loss.item()

    loss_media = soma_loss / len(loader_treino)
    historico_loss.append(loss_media)

    print("Época:", epoca + 1, "Loss médio:", loss_media)


# 6. Avaliação no teste

modelo.eval()

quantidade_corretas = 0
quantidade_total = 0

with torch.no_grad():

    for imagens, rotulos in loader_teste:

        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        saidas = modelo(imagens)

        previsoes = torch.argmax(saidas, dim=1)

        quantidade_corretas = quantidade_corretas + (previsoes == rotulos).sum().item()
        quantidade_total = quantidade_total + rotulos.size(0)

acuracia = quantidade_corretas / quantidade_total

print("Acurácia no teste:", acuracia)


# 7. Gráfico da loss

plt.plot(range(1, epocas + 1), historico_loss)

plt.xlabel("Época")
plt.ylabel("Loss médio")
plt.title("Evolução da função de perda")

plt.savefig("loss_mnist.png", dpi=300, bbox_inches="tight")
plt.close()


# 8. Algumas previsões

imagens, rotulos = next(iter(loader_teste))

imagens = imagens.to(dispositivo)
rotulos = rotulos.to(dispositivo)

modelo.eval()

with torch.no_grad():
    saidas = modelo(imagens)
    previsoes = torch.argmax(saidas, dim=1)

imagens = imagens.cpu()
rotulos = rotulos.cpu()
previsoes = previsoes.cpu()

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    imagem = imagens[i].squeeze()

    plt.imshow(imagem, cmap="gray")
    plt.title("Real: " + str(rotulos[i].item()) + "\nPred: " + str(previsoes[i].item()))
    plt.axis("off")

plt.tight_layout(h_pad=2.0)
plt.savefig("previsoes_mnist.png", dpi=300, bbox_inches="tight")
plt.close()
