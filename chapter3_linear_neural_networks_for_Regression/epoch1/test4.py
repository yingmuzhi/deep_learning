'''
从零开始手撸一个 softmax 分类问题

逻辑回归， softmax回归都是分类问题， 他们是强监督学习的另一个大类。对此他们采用的损失函数(cross-entropy loss交叉熵损失, 交叉熵损失可以通过信息论了解)和最后一层的激活函数(softmax)是不一样的。损失函数同样是由极大似然估计(MLE)得到的。

分类问题，网络模型最终的输出会有多个channel且引入one-hot编码的概念。使用Fashion-MNIST

手撸一个 softmax 分类问题
'''


import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from timer import Timer
from d2l import torch as d2l


# 1. 下载数据，并了解transforms.ToTensor()方法，查看一下数据等操作
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0〜1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="/home/yingmuzhi/_learning/d2l/data/Fashion_MNIST", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="/home/yingmuzhi/_learning/d2l/data/Fashion_MNIST", train=False, transform=trans, download=False)

print(len(mnist_train), len(mnist_test))

print(mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]



# # 展示图片
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
#     """绘制图像列表"""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图⽚张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图⽚
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes


batch_size = 256
def get_dataloader_workers(): #@save
    """使⽤4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
num_workers=get_dataloader_workers())


# 查看一下读取时间
timer = Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')




# 2. 整合上面所有的组件 - 形成一个加载dataloader的脚本
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

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break


""" 
(3). 开始softmax分类，包括:

1. 加载DataLoader数据
2. 初始化权重
3. 定义激活函数softmax
4. 定义网络模型
5. 定义损失函数交叉熵损失
6. 用Accuracy做评价指标
"""

# 从0开始softmax 分类（逻辑回归）
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


# 初始化权重，正态分布weight 和 为0的bias
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) 
b = torch.zeros(num_outputs, requires_grad=True)


# 使用sum()手动定义激活函数softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)


# 定义模型Y = softmax(Xw + b)
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 定义损失函数：还是由极大似然估计得到的**交叉熵损失**作为 损失函数
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)


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

print(evaluate_accuracy(net, test_iter))


"""
(4). 开始训练

"""

# 开始训练
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

# optimizer to minimize cost function
def sgd(params, lr, batch_size):
    """mini batchsize SGD"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    """训练模型（定义⻅第3章）"""
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        # legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


"""
(5). 预测

"""

# 做预测
def predict_ch3(net, test_iter, n=6): #@save
    """预测标签（定义⻅第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    print(titles)

predict_ch3(net, test_iter)