'''

4.8 数值稳定性和模型初始化

模型权重初始化 可以影响 非线性激活函数 的选择。这两者可以决定优化算法的收敛速度.

模型权重初始化 和 非线性激活函数 这两者如果选的不好会引起 梯度消失 和 梯度爆炸 问题.

由于链式法则的使用，当链太长的时候，多个概率相乘会引起数值下溢问题。除此之外，不稳定的梯度相乘还会引起梯度消失和爆炸
梯度爆炸（gradient exploding）问题：参数更新过⼤，破坏了模型的稳定收敛；
梯度消失（gradient vanishing）问题：参数更新过⼩，在每次更新时⼏乎不会移动，导致模型⽆法学习。

故我们需要对权重进行初始化(如Xavier) 并且 选择合适的激活函数(如ReLU激活函数)

'''

"""
(1) 梯度消失问题

梯度消失问题很有可能是因为激活函数的选择（如sigmoid激活函数）。

梯度消失问题的例子：如sigmoid引起的梯度消失问题。由于sigmoid函数的特殊性，只有当你的输入为0的时候才能很好的防止梯度消失，
而当你的输入在两侧的时候，梯度都很接近于0。当网络深度一加深后（整个乘积的梯度将消失），由此带来的梯度消失将是非常严重的（
所以人们使用ReLU激活函数更多，虽然这违反了神经元失活）。

"""
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) 
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
    legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))


"""
(2) 梯度爆炸

由于深度网络的初始化导致的。

"""
M = torch.normal(0, 1, size=(4,4))
print('⼀个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))
    
print('乘以100个矩阵后\n', M)