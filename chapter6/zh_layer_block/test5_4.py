'''
5.4 自定义层

小结:
- 我们可以通过基本层类设计⾃定义层。这允许我们定义灵活的新层，其⾏为与深度学习框架中的任何
现有层不同。
- 在⾃定义层定义完成后，我们就可以在任意环境和⽹络架构中调⽤该⾃定义层。
- 层可以有局部参数，这些参数可以通过内置函数创建。

目录：
在forward中增加计算
自定义一个线性层
'''

import torch
import torch.nn.functional as F
from torch import nn


"""
(1) 在forward中增加计算

"""
# 增加减去均值的计算
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))


# 增加更复杂的计算
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())


"""
(2) 自定义一个线性层

"""
# 自定义Linear()
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)

# 正向传播
linear(torch.rand(2, 5))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))