'''
Version: 1.0

Time: 202403013

intro: core.py for machine learning.

Author : YMZ
              ii.                                         ;9ABH,          
             SA391,                                    .r9GG35&G          
             &#ii13Gh;                               i3X31i;:,rB1         
             iMs,:,i5895,                         .5G91:,:;:s1:8A         
              33::::,,;5G5,                     ,58Si,,:::,sHX;iH1        
               Sr.,:;rs13BBX35hh11511h5Shhh5S3GAXS:.,,::,,1AG3i,GG        
               .G51S511sr;;iiiishS8G89Shsrrsh59S;.,,,,,..5A85Si,h8        
              :SB9s:,............................,,,.,,,SASh53h,1G.       
           .r18S;..,,,,,,,,,,,,,,,,,,,,,,,,,,,,,....,,.1H315199,rX,       
         ;S89s,..,,,,,,,,,,,,,,,,,,,,,,,....,,.......,,,;r1ShS8,;Xi       
       i55s:.........,,,,,,,,,,,,,,,,.,,,......,.....,,....r9&5.:X1       
      59;.....,.     .,,,,,,,,,,,...        .............,..:1;.:&s       
     s8,..;53S5S3s.   .,,,,,,,.,..      i15S5h1:.........,,,..,,:99       
     93.:39s:rSGB@A;  ..,,,,.....    .SG3hhh9G&BGi..,,,,,,,,,,,,.,83      
     G5.G8  9#@@@@@X. .,,,,,,.....  iA9,.S&B###@@Mr...,,,,,,,,..,.;Xh     
     Gs.X8 S@@@@@@@B:..,,,,,,,,,,. rA1 ,A@@@@@@@@@H:........,,,,,,.iX:    
    ;9. ,8A#@@@@@@#5,.,,,,,,,,,... 9A. 8@@@@@@@@@@M;    ....,,,,,,,,S8    
    X3    iS8XAHH8s.,,,,,,,,,,...,..58hH@@@@@@@@@Hs       ...,,,,,,,:Gs   
   r8,        ,,,...,,,,,,,,,,.....  ,h8XABMMHX3r.          .,,,,,,,.rX:  
  :9, .    .:,..,:;;;::,.,,,,,..          .,,.               ..,,,,,,.59  
 .Si      ,:.i8HBMMMMMB&5,....                    .            .,,,,,.sMr
 SS       :: h@@@@@@@@@@#; .                     ...  .         ..,,,,iM5
 91  .    ;:.,1&@@@@@@MXs.                            .          .,,:,:&S
 hS ....  .:;,,,i3MMS1;..,..... .  .     ...                     ..,:,.99
 ,8; ..... .,:,..,8Ms:;,,,...                                     .,::.83
  s&: ....  .sS553B@@HX3s;,.    .,;13h.                            .:::&1
   SXr  .  ...;s3G99XA&X88Shss11155hi.                             ,;:h&,
    iH8:  . ..   ,;iiii;,::,,,,,.                                 .;irHA  
     ,8X5;   .     .......                                       ,;iihS8Gi
        1831,                                                 .,;irrrrrs&@
          ;5A8r.                                            .:;iiiiirrss1H
            :X@H3s.......                                .,:;iii;iiiiirsrh
             r#h:;,...,,.. .,,:;;;;;:::,...              .:;;;;;;iiiirrss1
            ,M8 ..,....,.....,,::::::,,...         .     .,;;;iiiiiirss11h
            8B;.,,,,,,,.,.....          .           ..   .:;;;;iirrsss111h
           i@5,:::,,,,,,,,.... .                   . .:::;;;;;irrrss111111
           9Bi,:,,,,......                        ..r91;;;;;iirrsss1ss1111


Usage:
    1. `def add_to_class`: add later function/method to class.
    2. `def HyperParameters.save_hyperparameters`: save hyper params in self.param.
    3. `def ProgressBoard.draw`: like Tensorboard, drawing loss value while training.


Comments: 
# ============  ============
# ============ preparing data ... ============

#*********** -----------------
#            | 
#*********** -----------------

# --- |
# --- | 1.  | ---
# --- |

# --- |
# --- | 6. change script's MEDIAN and IQR to yours in script `tile_image_multi_thread.py` | ---
# --- |


BlockComments:
# --- TODO::..
# ---


Checkpoint:
        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            caler_state = self.scaler.state_dict(),         # may not use
            scheduler_state = self.scheduler.state_dict(),  # may not use
            epoch = self.current_epoch,
        )

        
Checkpoint:
        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),  
            caler_state = self.scaler.state_dict(),         # may not use
            scheduler_state = self.scheduler.state_dict(),  # may not use
            _best_validation_loss = self._best_validation_loss  # add
            epoch = self.current_epoch,
        )

        
logger:
        self.logger(
            # df,
            train_mean_loss,
            lr,
            validation_mean_loss,
            metrics,
            training_time,
        )

        
'''
import math
import time
import sys
import numpy as np
import torch
import inspect
import collections
from IPython import display
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.nn import functional as F

# Const
CURRENT_VERSION=1.0
SEED = seed = 3407

# set seed
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("The version of core.py is {}\n".format(CURRENT_VERSION) + \
      "The seed is {}::if you want to change seed, change it in core.py".format(SEED))

core = sys.modules[__name__]

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
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        # raise NotImplemented
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
        # raise NotImplementedError
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
        """
        intro:
            read the data path. 
            init self.train == Dataset_train
            init self.val == Dataset_val
            num_workers is used in DataLoader.
        """
        self.save_hyperparameters()

    def train_dataloader(self):
        """
        intro:
            return training dataloader
        """
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        """
        intro:
            return validation dataloader
        """
        return self.get_dataloader(train=False)
    
    def get_dataloader(self, train):
        """
        intro:
            return train / validation dataloader depend on train == True / False
        """
        # raise NotImplementedError
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

    # add
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """
        intro:
            get dataset through `class DIYDataset` inherit `Dataset` using `torch.utils.data.Dataset`. then return dataloader
        """
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
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
class Module(nn.Module, HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        """
        intro:
            init self.net to get model parameters. 
            init self.board to get ProgressBar, that is, TensorBoard.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = None
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        """
        intro:
            init loss(fn), then calculate the loss between (y_hat, y).
        """
        # raise NotImplementedError
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def forward(self, X):
        """
        intro:
            how to calculate the model learnable parameters using forward.
        """
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """
        intro:
            plot loss. Plot a point in animation.
        """
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
        """
        intro:
            how to calculate loss in one batch.
        """
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        """
        intro:
            how to calculate loss in one batch.
        """
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        """
        intro:
            return optimizer
        """
        # raise NotImplementedError
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def apply_init(self, inputs, init=None):
        """
        intro:
            Judge whether the dataset's batch can through the model.
            Init the `self.net` parameter using `init`
        """
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
"""
Name:
    Training Module

Intro:
    The Trainer class trains the learnable parameters in the Module class with data specified in DataModule. 

    The key method is fit, which accepts two arguments: model, an instance of Module, and data, an instance of DataModule. It then iterates over the entire dataset max_epochs times to train the model. 
"""
class Trainer(HyperParameters):  
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        intro:
            init trainer
        """
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        """
        intro:
            get self.train_dataloader, self.val_dataloader
        """
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
        return batch
    
    def prepare_model(self, model):
        """
        intro:
            get self.model
        """
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        """
        intro:
            fit.
        """
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        """
        intro:
            fit per epoch
        """
        # 1. training time
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            # 2. grad clip
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return 
        # 3. (optional) validation time
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
    
    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
"""
Name:
    Trainer_GPU

Intro:
    Trainer on GPU.
"""
def cpu():
    """
    intro:
        Get the CPU device.
    """
    return torch.device('cpu')
def gpu(i=0):
    """
    intro:
        Get a GPU device.
    """
    return torch.device("cuda:{}".format(i))
def num_gpus():
    """
    intro:
        Get the number of available GPUs.
    """
    return torch.cuda.device_count()
class Trainer_GPU(Trainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        intro:
            using GPU.
        """
        self.save_hyperparameters()
        self.gpus = [gpu(i) for i in range(min(num_gpus, core.num_gpus()))]
        pass
    
    def prepare_batch(self, batch):
        """
        intro:
            passing batch to device.
        """
        if self.gpus:
            batch = [to(a, self.gpus[0]) for a in batch]
        return batch
    
    def prepare_model(self, model):
        """
        intro:
            passing model to device.
        """
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model
"""
Name:
    Trainer_DDP

Intro:
    Trainer on DDP.
"""
class Trainer_DDP():
    pass
# endregion


# region Appendix
normal = torch.normal
zeros = torch.zeros
matmul = torch.matmul
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
float32 = torch.float32
randn = torch.randn
# region Chapter 3
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
    
    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

class LinearRegressionScratch(Module):
    """The linear regression model implemented from scratch.

    Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = zeros(1, requires_grad=True)

    def forward(self, X):
        """Defined in :numref:`sec_linear_scratch`"""
        return matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_scratch`"""
        l = (y_hat - y) ** 2 / 2
        return reduce_mean(l)

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_scratch`"""
        return SGD([self.w, self.b], self.lr)

class SGD(HyperParameters):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        """Defined in :numref:`sec_linear_concise`"""
        return self.net(X)

    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_concise`"""
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_concise`"""
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        """Defined in :numref:`sec_linear_concise`"""
        return (self.net.weight.data, self.net.bias.data)
# endregion
# region Chapter 4
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = core.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = core.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28), download=False, root="/home/yingmuzhi/_learning/d2l/data"):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=download)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=download)

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)

class Classifier(Module):
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
    
        Defined in :numref:`sec_classification`"""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
# endregion
# region Chapter 5
def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
# endregion 
# region Chapter 7
def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = reduce_sum((X[i: i + h, j: j + w] * K))
    return Y   
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
