'''一些数学知识，关于线性代数的'''

# 统计学中一行数据叫一个样本，一列数据叫一个特征feature，最后一列往往是标签label

import torch

# 标量 - 指0维数据
a = torch.tensor(1.)
print(a)

# 向量 - 指1维数据
b = torch.arange(12)
print(b, b[7], len(b), b.shape)

# 矩阵 - 指多维数据
c = torch.arange(12).reshape(3, 4)
print(c)
# 矩阵转置
# d = torch.transpose(c, 1, 0)
d = c.transpose(1, 0)
print(d)

# 两个矩阵的哈马达积用*
e = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# f = e         # id 相同
# f = e.clone() # id 不同
f = e + 0       # id 不同
print(id(e) == id(f))
g = f * e   # 哈马达积
print(g)

# 矩阵求和sum(); 求均值mean(); 
# 你除了知道这两个API的作用外，你还应该知道这两个函数是用来降维的。
h = g.sum(axis=[0, 1])
i = g.mean(axis=[0, 1])
print(h, i)

# 两个向量的点积用.dot()
j = torch.ones(4, dtype=torch.float32)
k = j.clone()
l = j.dot(k)
print(l)

# 矩阵-向量乘法用.mv
m = torch.arange(12).reshape(6, 2)
n = torch.ones(2, dtype=torch.long) # shpae=[2, 1, 1, 1, ...]补全是往后补1
print(n)
o = m.mv(n)
print(o)

# 矩阵-矩阵乘法用.mm
p = m.transpose(1, 0)
q = m.mm(p)
print(q)

# 向量二范数（又叫L2范数）就是平方和的平方根, 用norm()二范数的英文
r = torch.tensor([3., 4.])
s = r.norm()
print(s)

# 向量一范数（又叫L1范数）就是绝对值的和
t = r.abs().sum()
print(t)

# 矩阵的Frobenius范数类似向量的二范数，就是各个元素平方和的平方根
u = torch.randn(3, 4)
v = u.norm()
print(v)
