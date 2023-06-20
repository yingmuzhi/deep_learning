'''
MLP

MLP由于引入了隐藏层+激活函数，使得增加了非线性性质。这样就能表达更多的多项式, 例如次数大于1的幂函数


本章主要手撕了一个 MLP 
'''

import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import  data
from d2l import torch as d2l


"""
(1) DataLoader
"""
batch_size = 256

def get_dataloader_workers(): #@save
    """使⽤4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/home/yingmuzhi/_learning/d2l/data/Fashion_MNIST", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/home/yingmuzhi/_learning/d2l/data/Fashion_MNIST", train=False, transform=trans, download=False)
    return (data.DataLoader(
        mnist_train, batch_size, shuffle=True,
        num_workers=get_dataloader_workers()),  # 设置num_workers的个数，使得DataLoader加载的数据更多，更快
        data.DataLoader(mnist_test, batch_size, shuffle=False,
        num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(batch_size)


"""
(2) 初始化模型参数
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


"""
(3) 激活函数
"""
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


"""
(4) 定义模型
"""
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1) # 这⾥“@”代表矩阵乘法
    return (H@W2 + b2)


"""
(5) 使用损失函数
交叉熵损失函数
"""
loss = nn.CrossEntropyLoss(reduction='none')


"""
(6) 训练过程
同softmax训练过程
"""


# 用Accuracy做评价指标
def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator: #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# 训练一个epoch
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    """训练模型⼀个迭代周期（定义⻅第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3) # 累计计算
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使⽤PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:# 使⽤定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 开始训练
import os, sys
sys.path.append(os.path.dirname(__file__))
from animator import Animator
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    """训练模型（定义⻅第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


"""
(5) 进行预测
"""
def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 做预测
def predict_ch3(net, test_iter, n=6): #@save
    """预测标签（定义⻅第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    print(titles)

# predict_ch3(net, test_iter)