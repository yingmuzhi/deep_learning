import torch
from torch import nn

"""
以2D为例，在卷积计算过程中最重要的就是一下几个参数，

- input_size: Sin   (也即输入图像大小)
- output_size: Sout (也即输出图像大小)
- input_channel: Cin
- output_channel: Cout
- kernel_size: k1*k2 (一般卷积核大小为正方形，令k = k1 = k2)
- kernel_padding: padding
- kernel_stride: stride
- parameters_number: Params (参数个数)

计算过程：在每一次计算时，会进行output_channel轮次的互相关计算；而每一轮，会有Cin个kernel去跟图片的每个channel去进行互相关计算，最终将Cin个结果相加，得到output中的一个channel值；

经过卷积计算后**输出特征矩阵的尺寸大小**计算公式：Sout = (Sin - k + 2*padding) / stride + 1

**卷积层参数个数**计算: Params = Cout * ( Cin * k * k + 1 )      # 1 是卷积的bias

**全连接层参数个数**计算: Params = (Cin + 1) * Cout         # 1 是全连接层的bias, 全连接层会全部展平, 不要再讲什么多channel了, 多channel都没展平
"""
"""padding"""
# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)


# We use a convolution kernel with height 5 and width 3. The padding on either
# side of the height and width are 2 and 1, respectively
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)


"""stride"""
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)


conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)