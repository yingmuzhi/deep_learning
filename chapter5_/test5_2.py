'''
5.2 参数管理

小结:
- 我们有⼏种⽅法可以访问、初始化和绑定模型参数。
- 我们可以使⽤⾃定义初始化⽅法。

目录:
参数访问
从嵌套块收集参数
参数初始化
参数共享

'''


"""
(1) 参数访问

1. Sequential()的访问就如同list, 用.state_dict()访问其中的参数

2. 参数底层值的访问。
但是参数是复合的对象，包含值、梯度和额外信息。我们需要访问底层的值需要用.data 相当于调用detach()去处梯度获得了实际值。

"""
#  构造一个简单的MLP
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))


# method 1 --- 访问底层数值
# 参数访问，访问表层
print(net[2].state_dict())

# 访问底层数值
print(type(net[2].bias))    
print(net[2].bias)  # 参数是复合的对象，包含值、梯度和额外信息。
print(net[2].bias.data) # .data 相当于调用detach()去处了梯度
# 在上⾯这个⽹络中，由于我们还没有调⽤反向传播，所以参数的梯度处于初始状态。 => 只要不调用backward(), 梯度矩阵中的值就不会更新; 不调用step(), 参数矩阵中的值就不会更新
net[2].weight.grad == None


# method 2 --- 访问底层数值
# 访问第⼀个全连接层的参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

# 访问所有层
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 这为我们提供了另⼀种访问⽹络参数的⽅式
print(net.state_dict()['2.bias'].data)  # .data 相当于调用detach()去处了梯度


"""
(2) 从嵌套块收集参数

1. 直接nn.Sequential(list)中传入list

2. 除了传入list, 还能传入name, Module 来传入带名称的layer.
    使用 net = Sequential(); net.add_module()来增加嵌套块。

3. () 又叫函数调用符
"""
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套block
        net.add_module("block {}".format(i), block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# 展示model
print(rgnet)


# 访问权重bias底层值
print(rgnet[0][1][0].bias.data)


"""
(3) 参数初始化

使用内置的初始化器
我们可以将权重变为 高斯分布，全0，全常值，xavier初始化，任意分布，直接设置参数等
我们还可以将Sequential中特定的层的权重值初始化

使用net.apply(函数名)来初始化权重值

"""

# 高斯分布的weight 和 全为0的bias
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print(net[0].weight.data, net[0].bias.data)


# 全1的weight
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data, net[0].bias.data)


# 对不同layer初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data)
print(net[2].weight.data)


# 任意分布
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
            for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10) 
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])


# 我们始终可以直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]


"""
(4) 参数共享

实例化一个对象，将这个对象传入nn.Sequential中，他们会共享参数，但是梯度相加
"""
# 我们需要给共享层⼀个名称，以便可以引⽤它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
    shared, nn.ReLU(),
    shared, nn.ReLU(),
    nn.Linear(8, 1))
    
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同⼀个对象，⽽不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])









