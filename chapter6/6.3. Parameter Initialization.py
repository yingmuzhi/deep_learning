

import torch
from torch import nn


net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
print(net(X).shape)


"""

6.3.1. Built-in Initialization
using built-in func to init.

- `nn.init.normal_(module.weight, mean=0, std=0.01)`
- `nn.init.zeros_(module.bias)`
- `nn.init.constant_(module.weight, 1)`
- `nn.init.zeros_(module.bias)`
- `nn.init.xavier_uniform_(module.weight)`
- `nn.init.kaiming_uniform_(module.weight)` # default one for Linear, and the type is Leaky_ReLU
- `nn.init.uniform_(module.weight, -10, 10)`

net.apply(init_normal)方法会将model中的所有层都执行init_normal方法，用于进行模型参数初始化
"""
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
print(net[0].weight.data[0]) 
print(net[0].bias.data[0])


def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
print(net[0].weight.data[0]); print(net[0].bias.data[0])

def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier); net[2].apply(init_42)
print(net[0].weight.data[0]); print(net[2].weight.data)


"""

6.3.1.1. Custom Initialization
"""
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

def _weights_init(m):
    """
    intro:
        weights init.
        finish these:
            - torch.nn.Linear
    >>> version 1.0.0
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])    # linear - param - weight
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            print("Init", *[(name, param.shape) for name, param in m.named_parameters()][1])    # linear - param - bias
            nn.init.zeros_(m.bias)
    
    args:
        :param torch.parameters m: nn.Module
    """
    classname = m.__class__.__name__

    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])    # linear - param - weight
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            print("Init", *[(name, param.shape) for name, param in m.named_parameters()][1])    # linear - param - bias
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

net.apply(_weights_init)
    
net.apply(my_init)
net[0].weight[:2]

# set param directly
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]