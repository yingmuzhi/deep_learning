'''
多层感知机的简单实现，调用高级API
'''
import torch
from torch import nn
from d2l import torch as d2l


"""
(1) 搭建网络
"""
# 搭建网络
net = nn.Sequential(nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10))


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)


"""
(2) 开始训练

"""

# 配置超参数
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# 开始训练
from test4_1 import load_data_fashion_mnist, train_ch3
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)