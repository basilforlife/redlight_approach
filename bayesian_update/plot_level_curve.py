import matplotlib.pyplot as plt
import numpy as np
import torch

from redlight_approach.bayesian_update.model import NeuralNetwork
from redlight_approach.bayesian_update.train import function_thing


# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load("model.parameters"))
model.eval()  # Set to inference mode


# Make data to plot
x = np.arange(1, 10, 0.1)
y = 3

X = [[a, y] for a in x]  # Vary x, hold y constant
input = torch.Tensor(X)
model_output = model(input)

truth = [function_thing(a, y) for a in x]


# Plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, truth, c="b", marker="o", label="Truth")
ax.scatter(x, model_output.detach().numpy(), c="r", marker="+", label="Model")
plt.legend()
plt.show()
