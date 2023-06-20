'''
经常存储的会有

- 模型参数, 包括权重weights和偏置bias, 即model.params.weight和model.params.bias
- optimizer
- epoch迭代次数
- ...

我们的处理流程往往是：
先用torch.load()导入, 读成checkpoint
将checkpoint分步骤解析, 如有的可能需要使用load_state_dict()来读取
将需要存储的东西以一个字典的形式组装成checkpoint, 其中可能有些需要存储的东西要使用state_dict()来存储
使用torch.save()将checkpoint存储

... 表示无实意的pass或者: 
参考`https://zhuanlan.zhihu.com/p/264896206`

'''
import os, torch


SAVE_DICTIONARY: dict = {
    "model": {},
    "optimizer": {},
    "start_epoch": 1,
    "args": [],
}



def main(

):
    ...
    
    resume = '',
    model = None,
    optimizer = None,
    # load pre-trained model
    if os.path.exists(resume):
        checkpoint = torch.load(resume, map_location="cpu") # 先用torch.load()全部导入, 读成checkpoint再后续分解
        model.load_state_dict(checkpoint["model"])          # 有些要存储成torch的字典形式的, 就必须使用.load_state_dict()来将dict解压 和 用.state_dict()来压缩成dict 
        optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print("load pre-trained model successfully!")
    else:
        print("load pre-trained model failed.")
    
    ...

    epoch = 1
    args = []
    # save model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch, 
        "args": args
    }                                                       # 先组装成字典checkpoint, 再使用torch.save()全部存储
    torch.save(checkpoint, resume)

    ...