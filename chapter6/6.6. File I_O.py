
import torch
from torch import nn
from torch.nn import functional as F


""" save torch.Tensor"""
x = torch.arange(4)
torch.save(x, '/home/yingmuzhi/_learning/d2l/data/_legacy/x-file')

y = torch.load("/home/yingmuzhi/_learning/d2l/data/_legacy/x-file")
print(y)

# list
y = torch.zeros(4)
torch.save([x, y],'/home/yingmuzhi/_learning/d2l/data/_legacy/x-file')
x2, y2 = torch.load('/home/yingmuzhi/_learning/d2l/data/_legacy/x-file')
(x2, y2)

# dict
mydict = {'x': x, 'y': y}
torch.save(mydict, '/home/yingmuzhi/_learning/d2l/data/_legacy/x-file')
mydict2 = torch.load('/home/yingmuzhi/_learning/d2l/data/_legacy/x-file')
mydict2


""" save model parameters """
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), '/home/yingmuzhi/_learning/d2l/data/_legacy/mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('/home/yingmuzhi/_learning/d2l/data/_legacy/mlp.params'))
clone.eval()

Y_clone = clone(X)
Y_clone == Y