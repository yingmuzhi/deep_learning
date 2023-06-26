'''Lazy initialization = 延后初始化'''
import torch
from torch import nn
from d2l import torch as d2l


net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net[0].weight)    # won't init

net2 = nn.Linear(2, 2)  # will init using kaiming_uniform_
print(net2.weight)


# The following method passes in dummy inputs through the network for a dry run to infer all parameter shapes and subsequently initializes the parameters. 
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)


"""another test"""
import random, numpy as np, torch

# set random seed
seed = 416
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import math
def init_param(my_module):
    if type(my_module) == nn.Linear:
        nn.init.kaiming_uniform_(my_module.weight, a=math.sqrt(5))
        if my_module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(my_module.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(my_module.bias, -bound, bound)


model = nn.Sequential(nn.Linear(2, 2))
model.apply(init_param)
# check param
print(model[0].weight, '\n', model[0].bias)


"""init param"""
# !!write above on the first line!!
import random, numpy as np, torch

# set random seed
seed = 416
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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
    elif classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    >>> version 1.0.1
        refer https://blog.csdn.net/guofei_fly/article/details/105109883
        finish nn.Linear, nn.Conv
    
    args:
        :param torch.parameters m: nn.Module
    """
    classname = m.__class__.__name__

    if type(m) == nn.Linear or classname.startswith("Conv"):
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])    # linear - param - weight
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
        if m.bias is not None:
            print("Init", *[(name, param.shape) for name, param in m.named_parameters()][1])    # linear - param - bias
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
         
net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1), nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(1, 3, 224, 224))   # [ batch_size, channel, height, width ]
Y = net(X)
net.apply(_weights_init)

# check param
print(net[0].weight, '\n', net[0].bias)