import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from MLP import MLP
import numpy as np
import copy

n_samples = 500
SEED = 2309
NOISE = 0.3
EPOCHS = 200
ERROR = 1e-4

def main():
    img = 1
    X, y = make_moons(
        n_samples=n_samples,
        noise=NOISE,
        random_state=SEED
    )

    curr_dir = Path(__file__).parent.resolve()

    if not Path.is_dir(curr_dir/"plots/"):
        Path.mkdir(curr_dir/"plots")

    save_dir = curr_dir/"plots"

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot()
    plt.savefig(save_dir/f"0{img}-data.png")
    plt.close()
    img += 1

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    perceptrons_quantities = [5, 10, 20, 50]

    best_model = None
    best_loss = float("inf")
    best_neurons = None

    for perceptrons in perceptrons_quantities:
        model = MLP(2, perceptrons)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):

            model.train()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            model.eval()

            with torch.no_grad():
                val_outputs = model(X_val)
                vall_loss = criterion(val_outputs, y_val)

            val_losses.append(vall_loss.item())

            if epoch > 2 and abs(val_losses[-2] - val_losses[-1]) < ERROR:
                print(f"para {perceptrons} neurônios, parou na época {epoch}")
                break

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                best_model = copy.deepcopy(model)
                best_neurons = perceptrons
        
        plt.figure()

        plt.plot(train_losses, label="Treino")
        plt.plot(val_losses, label="Validação")

        plt.title(f"Loss de Treino e Validação ({perceptrons} neurônios)")
        plt.legend()
        plt.xlabel("Época")
        plt.ylabel("Loss")

        plt.savefig(save_dir/f"0{img}-loss_{perceptrons}.png")
        plt.close()
        img += 1

    best_model.eval()

    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        Z = best_model(grid)

    Z = Z.numpy().reshape(xx.shape)

    plt.figure()

    plt.contourf(xx, yy, Z, levels=20, cmap="bwr", alpha=0.4)
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr")

    plt.savefig(save_dir/f"0{img}-decision_boundary.png")
    img += 1

    with torch.no_grad():
        outputs = best_model(X_test)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean()

    print(f"Melhor número de neurônios: {best_neurons}")
    print(f"Acurácia no teste: {accuracy:.4f}")

if __name__ == "__main__":
    main()
