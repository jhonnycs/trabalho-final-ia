import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

SEED = 2309
learning_rate = 0.1
epochs = 1000
ERROR = 1e-4

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
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
    plt.savefig(save_dir/"01-data.png")
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    n_features = X_train.shape[1]
    W = np.zeros(n_features)
    b = 0

    m = X_train.shape[0]

    losses = []
    prev_loss = 0

    for epoch in range(epochs):
        z = np.dot(X_train, W) + b
        y_hat = sigmoid(z)

        loss = -(1/m)*np.sum(
            y_train*np.log(y_hat + 1e-15)
            + (1-y_train)*np.log(1-y_hat + 1e-15)
        )
        if (abs(prev_loss - loss)) < ERROR:
            break
        prev_loss = loss

        losses.append(loss)

        dz = y_hat - y_train

        dW = (1/m)*np.dot(X_train.T, dz)
        db = (1/m)*np.sum(dz)

        # Atualização
        W -= learning_rate*dW
        b -= learning_rate*db

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.savefig(save_dir/"02-loss.png")
    plt.close()

    z_test = np.dot(X_test, W) + b
    y_prob = sigmoid(z_test)

    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    print("Acurácia:", acc)


if __name__ == "__main__":
    main()