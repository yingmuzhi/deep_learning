'''torch.Tensor类型的基本使用'''

# 集中常见的初始化torch.Tensor的方法

import torch

# one dimension
x = torch.arange(12)
print(x, x.shape, x.dtype, x.numel(), )

# reshape
y = x.reshape(3, -1)
print(y)

# multi dimensions
a = torch.zeros((2, 3, 4))
b = torch.ones((2, 3, 4))
c = torch.randn((2, 3, 4))
d = torch.tensor([[2, 2, 2], [3, 3, 3]])
print(a, '\n', b, '\n', c, '\n', d)


# 运算符: 运算符的运算是按照元素进行的
e = a + b

# concat 拼接，指定拼接的dim
f = torch.cat((a, b), dim=0)
print(f)

# sum计算求和统计量
g = b.sum()
print(g)

# 判断位置是否相同，常用于做mask
k = a == b
print(k)

# 广播机制，在进行四则运算的时候，当矩阵形状不同时，会先扩展为形状相同再进行计算。但是很多情况下我并不知道怎么使用。常用的是矩阵加一个标量。
h = torch.arange(6).reshape(3, 2)
i = torch.arange(2).reshape(1, 2)
print(h)
print(i)
j = h+i
print(j)

# 切片和索引。同python中的list使用
a[0, 0, 0:3] = 7
print(a)

# 我们使用id()进行内存定位。Y=[expresion Y]会重新分配Y的内存；Y[:]=和Y+=都不会重新分配内存。建议多用后者，这样做可以减少内存开销。
before = id(y)
y = y + 1
print(id(y) == before)

before = id(y)
y += 1
print(id(y) == before)

before = id(y)
y[:] = y + 1
print(id(y) == before)

# 类型转换
l = x.numpy()
m = torch.tensor(l)
print(type(l), type(m))

# 提出tensor 的标量
n = torch.tensor([[1.]])
print(type(n.item()), type(float(n)))