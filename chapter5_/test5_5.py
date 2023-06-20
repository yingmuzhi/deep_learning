'''
5.5 读写文件

小结:
- save和load函数可⽤于张量对象的⽂件读写。
- 我们可以通过参数字典保存和加载⽹络的全部参数。
- 保存架构必须在代码中完成，⽽不是在参数中完成。

目录:
- 存储张量信息
- 存储整个模型的params 

'''
import torch
from torch import nn
from torch.nn import functional as F 


"""
(1) 存储张量信息

"""
# 存储单个张量
x = torch.arange(4)
torch.save(x, '/home/yingmuzhi/_learning/d2l/data/x-file')


# 读取张量
x2 = torch.load('/home/yingmuzhi/_learning/d2l/data/x-file')
print(x2)


# 存储list张量, dict张量同理。使用torch.load()和torch.save()
# >>> (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
# >>> {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}


"""
(2) 存储整个模型的params

"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


# 接下来，我们将模型的参数存储在⼀个叫做“mlp.params”的⽂件中
torch.save(net.state_dict(), '/home/yingmuzhi/_learning/d2l/data/mlp.params')


# 直接读取⽂件中存储的参数。
clone = MLP()
clone.load_state_dict(torch.load('/home/yingmuzhi/_learning/d2l/data/mlp.params'))
clone.eval()
print(clone)


# 由于两个实例具有相同的模型参数，验证一下输出是否相同
Y_clone = clone(X)
print(Y_clone == Y)
