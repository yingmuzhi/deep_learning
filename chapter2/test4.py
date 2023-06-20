'''关于微积分的学习'''

# 我们往往使用python进行参数的训练。在使用c/c++将模型部署到硬件/软件上。

# 微分。用导数定义求导。
def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}') 
    h *= 0.1


# torch或者tf相比较于numpy好用的地方有两点， 1. 分别是可以传入GPU进行计算 2. 可以自动微分（automatic differentiation）
# 其中torch会根据搭建的模型构建一个计算图(computational graph)来跟踪计算; 而反向传播(backpropagate)则是根据计算图计算每个参数的偏导数。
import torch

x = torch.arange(4.)
x.requires_grad_(True)  # 对x开启梯度跟踪
y = 2 * x.dot(x)    # 搭建好了computational graph
y.backward()    # backpropagate
print(x.grad)   # 输出梯度
print(x.grad == 4 * x)  # 判断梯度对不对

# 将张量的梯度设置为0用.grad.zero_()。否则会梯度爆炸。而这个梯度清零经常包围在backward()前后，即：
# optimizer.zero_grad()
# loss.backward()   # 这里的loss相当于你的目标函数y = 2 * x.dot(x)
# optimizer.step()
x.grad.zero_()  # 梯度清零
y = x.sum()
y.backward()
print(x.grad)

# 将张量从计算图中分离出来使用.detach()
x.grad.zero_()
y = x * x 
u = y.detach()
z = u * x   

z.sum().backward() # 这里在x的backpropagate，即计算z关于x的偏导数时候，因为.detach()的存在，到u就截止了。
print(x.grad == u)
# --- 这里你一定觉得很奇怪为什么要加sum()，这是因为backward()这个函数只对标量进行计算。当你的计算结果是向量的时候，就得使用sum()或者mean()进行降维，常使用sum().
# 参考链接:https://zhuanlan.zhihu.com/p/427853673