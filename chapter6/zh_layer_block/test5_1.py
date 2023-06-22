'''
5.1 层和块

layer层组成block块。layer是一个torch封装好的高级API，而block则是一个继承了nn.Module的子类。用nn.Module也能自定义layer.
nn.Module的子类是一个block块.在__init__中定义layer, 在__forward__中任意书写代码.

- ⼀个块可以由许多层组成；⼀个块可以由许多块组成
- 块可以包含代码。块负责⼤量的内部处理，包括参数初始化和反向传播。
- 层和块的顺序连接由Sequential块处理。

torch.zero((shape), required_grad=True)需要指定是否需要梯度; 而nn.Linear(2, 4)就不需要指定是否需要梯度，nn.的对象自动会有梯度

1. nn.Sequential()
nn.Sequential()**已经实现了内部的forward()方法**，即将列表中的每个块连接在⼀起，将每个块的输出作为下⼀个块的输⼊。
如nn.Sequential定义了⼀种特殊的Module，即在PyTorch中表⽰⼀个块的类，它维护了⼀个由Module组成
的有序列表。net(X)调⽤我们的模型来获得模型的输出。这实际上是net.__call__(X)的简写


2. 每个block块的基本功能
    - 输入数据为foward()函数的参数。输出数据为经过foward()函数后的输出。
    - 计算输出关于输入的梯度，是通过反向传播进行的，这是自动发生的，
    - block能够存储和访问前向传播所需要的参数
    - 可以根据需要初始化模型参数


3. 手撕一个block块, 包含一个MLP

4. 手撕一个nn.Sequential块

5. 在forward前向传播中可以任意执行代码

'''
import torch
from torch import nn
from torch.nn import functional as F


"""
(3) 手撕一个MLP

"""
class MLP(nn.Module):
    # ⽤模型参数声明层。这⾥，我们声明两个全连接的层
    def __init__(self):
        # 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
    
    # 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
    def forward(self, X):
        # 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

X = torch.rand(2, 20)

net = MLP()
print(net(X))

# 注意，上面的block和用下面nn.Sequential()定义的block实现的效果相同
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net.__call__(X))


"""
(4) nn.Sequential块

一个下划线是程序员默认的私有属性 PEP 8
`https://www.runoob.com/w3cnote/python-5-underline.html`

两个下划线是强制私有属性
`https://blog.csdn.net/liuskyter/article/details/80387726`

nn.Sequential本质上是个字典形式: key - value, 而 `for ... in ... .values()`则是按顺序取出了对应字典的键值对的值

"""
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
                self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


"""
(5) 在forward中任意执行代码

"""
# 在forward中进行运算
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使⽤创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1) # 复⽤全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1: X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))


# 随便定义层
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))