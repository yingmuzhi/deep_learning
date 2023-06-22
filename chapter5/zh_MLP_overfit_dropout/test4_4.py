'''
减少过拟合的方式主要三种方式：

1. 更多数据
2. 早停
3. 正则化( L1, L2 weight decay, Tikhonov( 增加噪声 , 即dropout暂退法) )

若因为数据过少引起过拟合，我们可以使用K折交叉验证；数据量过多而模型过于简单，或者训练epoch过少才会引起欠拟合。

下面主要讲解L2正则化: weight decay权重衰减. 我们通过减少权重的占比来取消一些权重在模型中的制衡值。

常使用L2正则化，通过权重衰减来减少过拟合。L2正则化线性模型构成岭回归，L1正则化构成
lasso；L2正则化常适用于大量特征上均匀分布权重的模型；L1正则化使得模型权重集中在一小
部分特征，而将其他权重清除为0.

PyTorch 封装的 API 调用，运行速度更快, 其中weight_decay指的是L_2损失的lambda.

这里的weight_decay是人为指定的, 他也是一个超参数.
'''

import torch
from torch import nn
from d2l import torch as d2l


"""
(1) 生成训练集, 验证集

"""
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


"""
(2) 初始化模型参数

"""
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True) 
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


"""
(3) 定义L2范数惩罚

"""
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


"""
(4) 开始训练

训练的唯一区别在于增加了惩罚项

"""
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
        xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # ⼴播机制使l2_penalty(w)成为⼀个⻓度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

# 忽略正则化直接训练
train(lambd=0)
pass

# 增加正则化进行训练
train(lambd=3)


"""
(5) 使用封装API简单实现，即指定weight_decay参数

"""
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
        xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                (d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

train_concise(0)
train_concise(3)