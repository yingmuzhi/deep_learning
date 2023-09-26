'''
不可运行

softmax的简洁表示



'''
import torch.nn as nn
import torch, d2l
"""(1). 初始化权重参数 """
# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


""" (2). 设置损失函数 """
loss = nn.CrossEntropyLoss(reduction='none')


""" (3). 设置优化函数 """
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


""" (4). 训练 """
# 开始训练
num_epochs = 10
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 不用d2l