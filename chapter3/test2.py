'''
线性回归简洁表示

这其实是一个feature=2, n=1000, label.shape=1的二元线性回归问题y = a * x_1 + b * x_2 + c: 用1000个样本(x_1, x_2)来拟合出a, b, c.
线性回归的简洁实现
'''

import numpy as np
import torch
from torch.utils import data


# 生成n = 1000组数据, label 1维, features 2维; => weight [2, 1]
# 初始化 weight 和 bias 的初始值
def synthetic_data(w, b, num_examples): 
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b 
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 构造一个Dataset和一个DataLoader对象, 并将DataLoader对象返回
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


print(next(iter(data_iter)))
pass


# 构造神经网络
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))


# 手动初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# cost func
loss = nn.MSELoss()


# optimizer
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# 开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
