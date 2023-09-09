'''
讲解一下Softmax Regression中需要使用的数据。 Fashion-Minist

'''
import sys, os
sys.path.append(__file__)
sys.path.append(os.path.dirname(__file__))

import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


"""

4.2 dataset
"""
# load dataset
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28), root="/home/yingmuzhi/_learning/d2l/data"):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=False)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=False)

data = FashionMNIST(resize=(32, 32))
print(len(data.train), len(data.val))


# explore data
print(data.train[0][0].shape)  # 1, 32, 32
# transform index to text
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]


# dataloader
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
# test
X, y = next(iter(data.train_dataloader()))  # 使用next(iter())来查看dataloader
print(X.shape, X.dtype, y.shape, y.dtype)
# timer
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'


# visualize data
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)