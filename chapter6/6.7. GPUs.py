
# nvidia-smi -l 2
"""
"""

import torch
from torch import nn

# 使用os中的环境变量来指定最终可以看到的GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
os.environ["NCCL_DEBUG"] = "INFO"

# 使用`net.to`, `tensor.to`将Torch.Tensor传到特定硬件上。

# 使用`nvidia-smi -l 2`来一直查看显卡挂载

print(torch.device("cpu"))
print(torch.device("cuda:1"))
print(torch.cuda.is_available())
print(torch.cuda.device_count())

def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)


def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

num_gpus()


def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()


# tensor
X = torch.zeros((2, 3), device=torch.device("cuda:1"))
X = X.to(device=torch.device("cuda:0"))
print(X.device)

# model
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=torch.device("cuda:0"))
print(net[0].weight.data.device)