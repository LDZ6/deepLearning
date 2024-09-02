import torch
from torch import nn

# 为了方便起见，我们定义了一个函数来计算卷积层的输出。
# 此函数会初始化卷积层的权重，并对输入和输出进行维度调整。
def comp_conv2d(conv2d, X):
    # 这里的 (1, 1) 表示批量大小和通道数都是 1
    X = X.reshape((1, 1) + X.shape)
    # 对输入X应用卷积层conv2d
    Y = conv2d(X)
    # 去除批量大小和通道数的维度，返回结果
    return Y.reshape(Y.shape[2:])

# 1. 填充操作
# 创建一个卷积层，输入通道数为1，输出通道数为1，卷积核大小为3x3，填充大小为1
# 这样卷积后输出的高度和宽度与输入相同
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# 创建一个随机输入张量，大小为8x8
X = torch.rand(size=(8, 8))
# 计算卷积输出的形状
print(comp_conv2d(conv2d, X).shape)  # torch.Size([8, 8])

# 当卷积核的高度和宽度不同时，我们可以分别对高度和宽度进行不同的填充，使输出和输入具有相同的高度和宽度
# 在此示例中，我们使用高度为5，宽度为3的卷积核，高度和宽度的填充分别为2和1
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)  # torch.Size([8, 8])

# 2. 步幅操作
# 在卷积计算中，卷积窗口从输入张量的左上角开始，向下、向右滑动。
# 这里我们设置了步幅（stride）为2，即每次滑动两个元素
# 设置卷积层，步幅为2，卷积核大小为3x3，填充大小为1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)  # torch.Size([4, 4])

# 3. 更复杂的例子
# 这里我们设置卷积核大小为3x5，填充为(0,1)，步幅为(3,4)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)  # torch.Size([2, 2])
