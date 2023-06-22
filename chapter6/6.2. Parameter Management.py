import torch
from torch import nn


net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4)) # 2 means the number of sample. Linear() just chage the last dimension's data.
print(net(X).shape)


"""

6.2.1. Parameter Access
"""
# print model
print(net)

# print parameters
print(net.state_dict())
print(net[2].state_dict())  # (8+1) * 1


"""

6.2.1.1. Targeted Parameters
"""
print(type(net[2].bias))    # <class 'torch.nn.parameter.Parameter'> an object, including data, grad and other information.
print(net[2].bias.data)     # just data.
print(net[2].weight.grad)   # no loss.backward(), such output is None.


"""

6.2.1.2. All Parameters at Once
"""


net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4)) # 2 means the number of sample. Linear() just chage the last dimension's data.
Y = net(X)

# type
print(type(net.state_dict()))
print(type(next(net.named_parameters())))
print(type(net.named_parameters()))

# value
print(net.state_dict())
print([(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)


"""

6.2.2. Tied Parameters
shared parameters
"""
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
