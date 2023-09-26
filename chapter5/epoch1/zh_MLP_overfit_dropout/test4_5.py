'''

dropout

Tikhonov正则化 

减少过拟合的暂退法dropout: 通过注入噪声实现, dropout=0.4时即随机失活40%的神经元.

任何运气不好，生成位置的值低于dropout的值都将被丢弃; 任何运气好的值都将被保留且**值被放大**.

暂退法在前向传播过程中，计算每⼀内部层的同时注⼊噪声，这已经成为
训练神经⽹络的常⽤技术。这种⽅法之所以被称为暂退法，因为我们从表⾯上看是在训练过程中丢弃（drop
out）⼀些神经元。

这里的dropout需要人为指定, 它也是一个超参数.

'''

import torch
from torch import nn
from d2l import torch as d2l


"""
(1) 定义dropout层

生成数据并根据不同的dropout值进行失活

"""
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1 # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))


"""
(2) 定义模型参数

并且使得暂退法只在训练期间使用

"""
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
            is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使⽤dropout
        if self.training == True: 
            # 在第⼀个全连接层之后添加⼀个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True: 
            # 在第⼆个全连接层之后添加⼀个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)


"""
(4) 开始训练

"""
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


"""
(5) 调用API简单实现dropout

通过在搭建模型的时候添加nn.Dropout(dropout1)实现

"""
# 搭建模型并初始化权重
net = nn.Sequential(nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第⼀个全连接层之后添加⼀个dropout层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第⼆个全连接层之后添加⼀个dropout层
    nn.Dropout(dropout2),
    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


# 开始训练
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)