'''
10.5 多头注意力

需要注意的是query, key, value都是在乘以W矩阵后，再分为多头（将d进行拆分，被分后多出的矩阵增加到batch上，[batch, number_of_query, dimension] 
**这里的batch指的是batch句话（同CNN中batch个图），number_of_query（一句话中有number_of_query个词），dimension指生成词向量有dimension个维度**）

将多头进行attention计算后，再concat拼接成一个头output_concat

将output_concat乘以矩阵W_o得到最终输出。

Transformer编码器中的任何层都不会改变其输⼊的形状。
FFN是基于位置的前馈⽹络, 它的本质上是一个MLP.
'''
import math, os
import torch
from torch import nn
from d2l import torch as d2l

"""
(-1) 实现10.3.1 掩蔽softmax操作

即实现Mask掩码操作，将超出的query查询强行赋值为负无穷，以使得其经过softmax处理后值为0，直接忽略掉。

为了仅将有意义的词元作为值来获取注意⼒汇聚，可以指定⼀个有效序列⻓度（即词元的个数），
以便在计算softmax时过滤掉超出指定范围的位置。

masked_softmax函数实现了这样的掩蔽softmax操
作（masked softmax operation），其中任何超出有效⻓度的位置都被掩蔽并置为0
"""
def sequence_mask(X, valid_len, value=0):
    """将值置为负无穷

    Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1) 
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0 
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 为了演⽰此函数是如何⼯作的，考虑由两个2 × 4矩阵表⽰的样本，这两个样本的有效⻓度分别为2和3。经过
# 掩蔽softmax操作，超出有效⻓度的值都被掩蔽为0。
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

# 同样，也可以使⽤⼆维张量，为矩阵样本中的每⼀⾏指定有效⻓度。
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))
pass


"""
(0) 实现10.3.3 缩放点积注意力

"""
# 显示注意力权重热力图
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
    
    save_path = "/home/yingmuzhi/_learning/d2l/transformer/fig1.jpg" 
    while os.path.exists(save_path):
        part1, part2 = save_path.split(".")
        
        part1+='_'
    
        save_path = part1 + '.' + part2

    d2l.plt.savefig(save_path)


# 点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意⼒"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1] 
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = d2l.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 指定参数
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# 我们令查询的特征维度与键的特征维度⼤⼩相同。
queries = torch.normal(0, 1, (2, 1, 2))
# values的⼩批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])


# 进行点积attention
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))


# 显示注意力权重热力图
show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
    xlabel='Keys', ylabel='Queries')
pass


"""
(1) transpose_qkv

"""
def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


"""
10.5.2 (2) 实现Multi-Head Attention

我们选择将 缩放点积注意力 作为每一个注意力头

我们设定pq = pk = pv = po/ho 

我们需要注意的是 如果将查询、键和值的线性变换的输出数量设置为 pqh = pkh = pvh = po，则可以并⾏计算h个头
"""


# 设定pq = pk = pv = po/ho 
class MultiHeadAttention(nn.Module):
    """多头注意⼒"""
    def __init__(self, 
        key_size, 
        query_size, 
        value_size, 
        num_hiddens,
        num_heads, 
        dropout, 
        bias=False, 
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens 的形状:
        # (batch_size，)或(batch_size，查询的个数) # 经过变换后，输出的queries，keys，values 的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)  # 乘以权重以后再多头
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None: 
            # 在轴0，将第⼀项（标量或者⽮量）复制num_heads次，
            # 然后如此复制第⼆项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)  # 对每个多头进行attention计算

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)  #output还要乘以一个权重


# 打印模型
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
    num_hiddens, num_heads, 0.5)
print(attention.eval()) # 打印模型的时候，不能填写print(model); 应该填写print(model.eval())


# 输出
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).shape)











'''
10.6 ⾃注意⼒和位置编码


本节将使⽤⾃注意⼒进⾏序列编码，以及如何使⽤序列
的顺序作为补充信息。

self-attention 能够提高并行计算，且能轻松学到长距离的依赖关系
'''
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
    num_hiddens, num_heads, 0.5)
print(attention.eval())


batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)


"""
(1) 位置编码

"""
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

    
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
    figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])

# d2l.plt.savefig("/home/yingmuzhi/_learning/d2l/transformer/fig2.jpg")


"""
(2) 打印绝对位置信息

"""
for i in range(8):
    print(f'{i}的⼆进制是：{i:>03b}')

P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
    ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')










'''
10.7 Transfomer

'''

import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

 
"""
(1) FFN

基于位置的前馈⽹络对序列中的所有位置的表⽰进⾏变换时使⽤的是同⼀个多层感知机（MLP），这就是称
前馈⽹络是基于位置的（positionwise）的原因。
"""
class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
        **kwargs
    ):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # self.scj = nn.Conv2d(2, 2, 2)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 打印网络，基于位置的前馈网络
ffn = PositionWiseFFN(4, 4, 8)
print(ffn.eval())


input = torch.ones((2, 3, 4))
output = ffn(input)
print(output)
print(ffn(torch.ones((2, 3, 4)))[0])


"""
(2) 加法和规范化（add&norm）组件

"""
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2) 
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算X的均值和⽅差
print('layer norm:', ln(X), '\nbatch norm:', bn(X))


# 使⽤残差连接和层规范化来实现AddNorm类。暂退法也被作为正则化⽅法使⽤。
class AddNorm(nn.Module):
    """残差连接后进⾏层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5)
print(add_norm.eval())
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)


"""
(3) 编码器

EncoderBlock类包含
两个⼦层：多头⾃注意⼒和基于位置的前馈⽹络，这两个⼦层都使⽤了残差连接和紧随的层规范化。
"""
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        dropout, use_bias=False, **kwargs
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # a = self.attention(X, X, X, valid_lens)
        # Y = self.addnorm1(X, a)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


# Transformer编码器中的任何层都不会改变其输⼊的形状。
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
print(encoder_blk.eval())
print(encoder_blk(X, valid_lens).shape)


"""
(4) 堆叠了N个block的编码器

"""
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
        num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
        num_heads, num_layers, dropout, use_bias=False, **kwargs
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, use_bias)
                )

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌⼊值乘以嵌⼊维度的平⽅根进⾏缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

    
# 创建一个堆叠了N=2个 layers 的编码器
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
print(encoder.eval())
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)


"""
(5) Transformer Decorder

运用了 auto-regression 自回归属性
"""
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        dropout, i, **kwargs
    ):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
            num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1] # 训练阶段，输出序列的所有词元都在同⼀时间处理，
        # 因此state[2][self.i]初始化为None。 # 预测阶段，输出序列是通过词元⼀个接着⼀个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表⽰
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每⼀⾏是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # ⾃注意⼒
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意⼒。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)


# 构建了由num_layers个DecoderBlock实例组成的完整的Transformer解码器
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
        num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
        num_heads, num_layers, dropout, **kwargs
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器⾃注意⼒权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”⾃注意⼒权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# d2l.predict_seq2seq
"""
(6) 开始训练Transformer

训练时使用teacher forcing

预测时就正常预测，由信号得到输出
"""
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout
)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout
)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ', 
        f'bleu {d2l.bleu(translation, fra, k=2):.3f}')


# 可视化Transformer的注意⼒权重
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
    -1, num_steps))
print(enc_attention_weights.shape)

show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5)
)


# --- after
dec_attention_weights_2d = [head[0].tolist()
for step in dec_attention_weight_seq
for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape

# Plusonetoincludethebeginning-of-sequencetoken
d2l.show_heatmaps(
dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
xlabel='Key positions', ylabel='Query positions',
titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

d2l.show_heatmaps(
dec_inter_attention_weights, xlabel='Key positions',
ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
figsize=(7, 3.5))