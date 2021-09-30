from itertools import product

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Make dataset
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def function_thing(x, y):
    return gaussian(x, mu=y, sig=y)


# Define training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, p) in enumerate(dataloader):
        x, p = x.to(device), p.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, p)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Define testing procedure
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, p in dataloader:
            x, p = x.to(device), p.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, p).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss = {test_loss:>8f}")


batch_size = 64

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
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Do stuff
model = NeuralNetwork().to(device)
print(model)


# Define loss fn and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


epochs = 5
for e in range(epochs):
    print(f"Epoch {e + 1} -----------------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")
