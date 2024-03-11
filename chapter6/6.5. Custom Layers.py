
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


"""
layer without parameters

深度学习最灵活的一点在于可以在forward方法中结合torch.nn.functional中的函数来实现自定义层。
"""
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.tensor([1.0, 2, 3, 4, 5]))

net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())


"""

6.5.2. Layers with Parameters
"""
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(type(linear.weight))


linear(torch.rand(2, 5))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))