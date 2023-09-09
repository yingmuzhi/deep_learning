import math
import time
import numpy as np
import torch
import inspect
import collections
from IPython import display
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import torch.nn as nn

# set seed
seed = 588
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# region Foundation APIs
def add_to_class(Class):  #@save
    """
    intro:
        Register functions as methods in created class.
    example:
    >>> # test
    >>> class A:
    >>>     def __init__(self):
    >>>         self.b = 1
    >>> 
    >>> a = A()
    >>> 
    >>> @add_to_class(A)
    >>> def do(self):
    >>>     print('Class attribute "b" is', self.b)
    >>> 
    >>> a.do()
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    """
    intro:
        The base class of hyperparameters.
    example:
    >>> # test
    >>> class B(HyperParameters):
    >>>     def __init__(self, a, b, c):
    >>>         self.save_hyperparameters(ignore=['c'])
    >>>         print('self.a =', self.a, 'self.b =', self.b)
    >>>         print('There is no self.c =', not hasattr(self, 'c'))
    >>> 
    >>> b = B(a=1, b=2, c=3)
    """
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

class ProgressBoard(HyperParameters):
    """
    intro:
        The board that plots data points in animation.
        Defined in :numref:`sec_oo-design`
    example:
    >>> board = ProgressBoard('x')
    >>> for x in np.arange(0, 10, 0.1):
    >>>     board.draw(x, np.sin(x), 'sin', every_n=2)
    >>>     board.draw(x, np.cos(x), 'cos', every_n=10)
    """
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils` 
        这里的(x,y)是二维的点坐标，而label是绘制曲线的名称，every_n是绘制的频率即每几个点后更新图形"""
        Point = collections.namedtuple('Point', ['x', 'y']) # 创建一个名为Point的命名元组 p1 = Point(x=1, y=2); print(p1.x)  # 输出 1
        if not hasattr(self, 'raw_points'): # 如果对象中没有self.raw_points属性
            self.raw_points = collections.OrderedDict() # 创建一个有序字典，即保持遍历顺序与插入顺序一致。在python3.7前普通字典是无序的，不会记住插入顺序。
            self.data = collections.OrderedDict()   
        if label not in self.raw_points:    # 如果现在添加的label不在self.raw_points中则添加，每一个label只添加一次
            self.raw_points[label] = []     # raw_points中存取的是暂时的数据，用于计算均值后存入data中
            self.data[label] = []           # data中存取的是永久的均值，绘图也是根据这个来的
        points = self.raw_points[label] #   point存储了raw_points中特定label的指针
        line = self.data[label] #           line存储了data中特定label的指针；就和c语言中list是常量指针一样，python中liist，dict，tuple都是指针，你这里赋予的是地址，改变a就会改变b
        points.append(Point(x, y))  #       points添加该点，即raw_points中会添加该点
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)    # 定义了匿名函数lambda，其中x是输入变量，冒号后面的是函数的具体定义。这句话是当使用mean函数传入一个列表时，会计算列表中所有函数的均值y并返回y。
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points]))) # line添加every_n的均值，即data中添加均值。这句话计算了points中x和y的均值，并生成一个Point对象，存储在line的list中，也存储在了data的list中
        points.clear()                      # points清空，即raw_points清空
        if not self.display:
            return
        use_svg_display()   # 用于设置在 Jupyter Notebook 中显示图形时使用 SVG 格式, 确保图形在高分辨率显示中具有良好的清晰度。
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize) # 创建新的matplotlib图形对象，并设置figsize
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])    # 在每次迭代中，这一行代码使用 plt.plot 函数创建一个线条。
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()    # 这行代码用于获取要在图形中绘制的坐标轴对象，并将其赋值给变量 axes。
        if self.xlim: axes.set_xlim(self.xlim)  # 如果没有横坐标范围，则设置横坐标范围
        if self.ylim: axes.set_ylim(self.ylim)  # 纵坐标
        if not self.xlabel: self.xlabel = self.x    # 设置横坐标标签值
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)    # 设置横坐标比例尺度
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)  # 添加图例
        display.display(self.fig)
        display.clear_output(wait=True)
# endregion


# region APIs
"""
Name: 
    Model module

Intro: 
    The Module class is the base class of all models we will implement. It inherits from `nn.Module`

    __init__, stores the learnable parameters

    training_step method accepts a data batch to return the loss value

    configure_optimizers returns the optimization method, or a list of them, that is used to update the learnable parameters

    validation_step to report the evaluation measures.
"""
def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')

class Module(nn.Module, HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError


"""
Name:
    Data Module

Intro:
    The DataModule class is the base class for data.

    __init__ method is used to prepare the data. This includes downloading and preprocessing if needed.

    train_dataloader returns the data loader for the training dataset. A data loader is a (Python) generator that yields a data batch each time it is used. This batch is then fed into the training_step method of Module to compute the loss.

    There is an optional val_dataloader to return the validation dataset loader.
"""
class DataModule(HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    # add
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
"""
Name:
    Training Module

Intro:
    The Trainer class trains the learnable parameters in the Module class with data specified in DataModule. 

    The key method is fit, which accepts two arguments: model, an instance of Module, and data, an instance of DataModule. It then iterates over the entire dataset max_epochs times to train the model. 
"""
class Trainer(HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        try:
            self.num_train_batches = len(self.train_dataloader) if len(self.train_dataloader) else 0
            self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        except:
            print("not using torch dataset")
    
    # add
    def prepare_batch(self, batch):
        raise NotImplementedError
    
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
# endregion


if __name__=="__main__":
    # 1. test for `add to class`
    # 作为装饰器，只能作用于方法，即将方法作为属性使用
    class A:
        def __init__(self):
            self.b = 1
    a = A()
    # @add_to_class(A)
    # b = 3
    @add_to_class(A)
    def do(self):
        print('Class attribute "b" is', self.b)
    a.do()

    # 2. test for `HyperParameters`
    # 只是将形参变为self.中的值；且需要在子类中的__init__方法中调用父类的save_hyperparameters方法
    class B(HyperParameters):
        def __init__(self, a, b, c):
            self.save_hyperparameters(ignore=['c'])
            print('self.a =', self.a, 'self.b =', self.b)
            print('There is no self.c =', not hasattr(self, 'c'))
    b = B(a=1, b=2, c=3)

    # 3. test for `ProgressBoard` 就是一个绘制loss的曲线
    board = ProgressBoard('x')
    for x in np.arange(0, 10, 0.1):
        board.draw(x, np.sin(x), 'sin', every_n=2)
        board.draw(x, np.cos(x), 'cos', every_n=10)

    # 4. test for Data module
    class SyntheticRegressionData(DataModule):  #@save
        """Synthetic data for linear regression."""
        def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                    batch_size=32):
            super().__init__()
            self.save_hyperparameters()
            n = num_train + num_val
            self.X = torch.randn(n, len(w))
            noise = torch.randn(n, 1) * noise
            self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise   
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    print('features:', data.X[0],'\nlabel:', data.y[0])
    print(data.X.shape, data.y.shape)
