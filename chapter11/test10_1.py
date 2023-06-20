'''
注意力机制

query(查询), key(键), value(值)

- 注意⼒机制与全连接层或者汇聚层的区别源于增加的⾃主提⽰。
- 注意⼒机制通过注意⼒汇聚使选择偏向于值（感官输⼊），其中包含查询（⾃主性提⽰）和键（⾮⾃主
性提⽰）。键和值是成对的。
- 查询（⾃主提⽰）和键（⾮⾃主提⽰）之间的交互形成
了注意⼒汇聚attention pooling；注意⼒汇聚有选择地聚合了值（感官输⼊）以⽣成最终的输出。


'''

"""
(1) 使用heatmap对注意力可视化

注意力权重图，即当Query和Key匹配到某一特定值后，热力图的地方颜色会变深，且热力图越深表示的是：**注意力权重越大**

仅当query==key的时候，注意力权重=1，否则=0.
"""
import torch
from d2l import torch as d2l

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
    cmap='Reds'
):
    """显⽰矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
        sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.savefig("/home/yingmuzhi/_learning/d2l/transformer/fig1.jpg")

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')