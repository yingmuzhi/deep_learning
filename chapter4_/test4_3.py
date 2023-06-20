
'''

使用MLP进行多项式回归

我们能够发现由于 MLP 引入了 非线性的原因， 它能够拟合更多的表达式，例如次数大于1的幂函数等。

在下面我们将通过 引入参数次数多少来体现`过拟合`和`欠拟合`。

在数据量方面，数据量过少会引起过拟合，数据量过多会引起欠拟合。
过拟合：当数据量太少时，模型无法完成充分的训练，模型过度拟合用于训练的少量数据的信息，对测试数据效果不好，泛化能力差；
欠拟合：数据量很多，但是模型太简单没有充分利用数据信息模型不够准确。
'''

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


"""
(1) 生成多项式

"""
max_degree = 20 # 多项式的最⼤阶数
n_train, n_test = 100, 100 # 训练和测试数据集⼤⼩
true_w = np.zeros(max_degree) # 分配⼤量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)


# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

print(features[:2], poly_features[:2, :], labels[:2])


"""
(2) 对模型增加评估标准

"""

def evaluate_loss(net, data_iter, loss): #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2) # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


"""
(3) 定义训练函数

"""
from animator import Animator
from test4_1 import train_epoch_ch3
from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(
    train_features, 
    test_features, 
    train_labels, 
    test_labels,
    num_epochs=400):

    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1] # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
        batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
        batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
        legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 三阶多项式: 正常拟合
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
    labels[:n_train], labels[n_train:])


# 一阶多项式: 欠拟合
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
    labels[:n_train], labels[n_train:])


# ⾼阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
    labels[:n_train], labels[n_train:], num_epochs=1500)