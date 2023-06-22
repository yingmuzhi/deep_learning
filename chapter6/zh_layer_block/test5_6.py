'''
设备指定

在跑多GPU时候，要确定你要计算的数据在同一张卡上。
'''

"""
(1) 指定设备

这个地方要在os.environ环境变量中指定CUDA可见参数的原因在于: 如果你不指定，则Pytorch会在其余可见的显卡上占用2~3MB的显存
"""
# 指定设备
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" # 会将实际上0, 2显卡设定为可见：即在该脚本中0 -> 0, 2 -> 1

import torch
print(torch.cuda.device_count())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
a = torch.tensor([2, 2]).to(device=device)
pass


"""
(2) 直接在GPU上创建torch.tensor

但是基本上用不到，因为基本上我们的数据流向是 硬盘 -> 内存 -> CPU(transform) -> 内存 -> GPU
"""
# 在GPU上直接创建
Y = torch.rand(2, 3, device=device)
pass