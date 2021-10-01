from itertools import product

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from redlight_approach.bayesian_update.model import NeuralNetwork, test, train


# Make dataset
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def function_thing(x, y):
    return gaussian(x, mu=y, sig=y)


if __name__ == "__main__":

    # Training params
    batch_size = 64
    epochs = 5

    x = np.arange(1, 10, 0.01)
    y = np.arange(1, 10, 0.01)

    input = []
    output = []
    for i, j in product(range(len(x)), repeat=2):
        input.append([x[i], y[j]])
        output.append([function_thing(x[i], y[j])])

    X_train, X_test, z_train, z_test = train_test_split(
        input, output, test_size=0.2, random_state=42
    )

    training_data = TensorDataset(
        torch.Tensor(np.array(X_train)), torch.Tensor(np.array(z_train))
    )
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data = TensorDataset(
        torch.Tensor(np.array(X_test)), torch.Tensor(np.array(z_test))
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print("Done making dataset")

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Define model
    model = NeuralNetwork().to(device)
    print(model)

    # Define loss fn and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Do the actual training
    for e in range(epochs):
        print(f"Epoch {e + 1} -----------------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)
    print("Done Training !!!!!!!!!!!!!!!!!!!!!!!!!")

    # Save model to file
    torch.save(model.state_dict(), "model.parameters")
