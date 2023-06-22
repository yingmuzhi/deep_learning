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