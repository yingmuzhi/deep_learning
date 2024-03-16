"""
**这一章节很重要**

批量归一化，BN层

训练深度神经网络的困难之处在于可能很难在合理时间内让网络收敛，这里我们引入BN层，
使得网络深度可以超过100层，BN层还有好处在于其固有的正则化。

在BN中，FC层和CNN层是不一样的，
对于FC层他将计算整个的mu和sigma；而对于CNN层，将根据channel数量计算每个channel的mu和sigma；

在批量归一化中，训练阶段和测试阶段他起到的作用是不一样的；
在训练阶段，其均值和方差都是针对一个小的batch_size进行计算的；
在测试阶段，均值和方差是针对整个数据集来计算的；BN这个特性和dropout层一样。

除了BN层的归一化还有LN层的归一化。

当BN层应用到具体的网络结构时，需要记住的是他是在卷积层或者全连接层之后，相应的激活层之前，即CNN-BN-AF

高级 API 变体运行速度要快得多，因为它的代码已编译为 C++ 或 CUDA，而我们的自定义实现必须由 Python 解释

它的工作原理：减少内部协变量偏移
"""
import torch
from torch import nn
import core


"""
BN层的手动实现
"""
# torch.no_grad()
# nn.Dropout().eval()
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    # `torch.is_grad_enabled()` 主要用于判断当前是否启用了梯度计算, 一般在评估模型时, 往往不需要计算梯度。
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data



class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y

# test
temp_builtin = [i for i in torch.nn.BatchNorm2d(4).parameters()]
temp_diy = [i for i in BatchNorm(num_features=4, num_dims=4).parameters()]
pass
# `self.moving_mean` and `self.moving_var` is not nn.Parameters(), not learnable.
# 只会记录gamma和beta，而其余都是根据数据计算得到的，nn.Parameters()所学习得到的也是这两个参数


"""
LeNet

在LeNet中使用手动搭建的BN层
"""
class BNLeNetScratch(core.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))

trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
data = core.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn)  # 因为使用了Lazy init所以要先forward一个Tensor初始化模型
trainer.fit(model, data)

# learnable gamma and beta
print(model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,)))


"""
LeNet - Concise Implementation

在LeNet中使用简洁的BN层, 他不需要指定输出的channel数量和输入的BN层维度（二维FC层或者四维2D-CNN层）
"""
class BNLeNet(core.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes))

trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
data = core.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn)
trainer.fit(model, data)