
import torch
from torch import nn


"""

7.2.1. The Cross-Correlation Operation
回忆深度学习中卷积层的计算，其实我们能够发现严格意义上讲这并不是卷积计算，而是互相关运算。我们使用
卷积核将对应位置的元素相乘后再相加得到一个元素，再将卷积核移动，计算下一个元素。
"""
# 接下来，我们手动实现卷积层的计算。X是Input输入向量，K是Kernel卷积核
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


"""

7.2.2. Convolutional Layers
卷积层的完整计算过程是：计算互相关后再加上偏置
"""
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    

"""

7.2.3. Object Edge Detection in Images
卷积层的一个简单应用在于：通过查找像素变化的位置来检测图像中对象的边缘
"""
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

# kernel ( 1, 2 )
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
print(Y)


"""

7.2.4. Learning a Kernel
使用可学习参数来学习权重矩阵； 更新梯度
"""
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))   # 会发现非常接近于我们前面定义的权重矩阵的元素的值
"""
就卷积本身而言，它们可用于多种目的，例如检测边缘和线条、模糊图像或锐化图像。
最重要的是，统计学家（或工程师）没有必要发明合适的过滤器。相反，我们可以简单地从数据中学习它们。
这用基于证据的统计数据取代了特征工程启发法。
"""