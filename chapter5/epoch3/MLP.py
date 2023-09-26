import torch
import core

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
core.plot()