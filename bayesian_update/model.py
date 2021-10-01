import torch
from torch import nn


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


# Define training loop
def train(dataloader, model, loss_fn, optimizer, device):
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
def test(dataloader, model, loss_fn, device):
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
