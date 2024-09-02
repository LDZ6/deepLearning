import torch
from torch import nn
from d2l import torch as d2l  # d2l库需要提前安装

# 1. 实现一个2D汇聚层函数
# 该函数支持最大汇聚（max pooling）和平均汇聚（average pooling）模式
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size  # 获取汇聚窗口的高度和宽度
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))  # 初始化输出张量
    for i in range(Y.shape[0]):  # 遍历输出张量的每一个元素
        for j in range(Y.shape[1]):
            # 获取输入张量中与汇聚窗口对应的子区域
            X_sub = X[i: i + p_h, j: j + p_w]
            if mode == 'max':  # 如果是最大汇聚
                Y[i, j] = X_sub.max()  # 取子区域中的最大值
            elif mode == 'avg':  # 如果是平均汇聚
                Y[i, j] = X_sub.mean()  # 取子区域中的平均值
    return Y

# 2. 构建输入张量X并测试2D最大汇聚层
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])

# 测试最大汇聚
print("Max pooling result:\n", pool2d(X, (2, 2)))
# 输出应为 tensor([[4., 5.],
#                 [7., 8.]])

# 测试平均汇聚
print("Average pooling result:\n", pool2d(X, (2, 2), 'avg'))
# 输出应为 tensor([[2., 3.],
#                 [5., 6.]])

# 3. 使用PyTorch中的内置2D最大汇聚层演示填充和步幅的使用
# 创建一个4x4的输入张量X，形状为 (1, 1, 4, 4)
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("Input tensor X:\n", X)
# 输出应为:
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]]]])

# 使用形状为(3, 3)的汇聚窗口，步幅为(3, 3)
pool2d = nn.MaxPool2d(3)
print("Max pooling with 3x3 window:\n", pool2d(X))
# 输出应为 tensor([[[[10.]]]])

# 指定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("Max pooling with padding and stride:\n", pool2d(X))
# 输出应为 tensor([[[[ 5.,  7.],
#                   [13., 15.]]]])

# 使用不同大小的矩形汇聚窗口，指定不同的填充和步幅
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print("Max pooling with rectangular window, different padding and stride:\n", pool2d(X))
# 输出应为 tensor([[[[ 5.,  7.],
#                   [13., 15.]]]])

# 4. 处理多个通道的输入数据
# 构建一个具有两个通道的输入张量X
X = torch.cat((X, X + 1), 1)
print("Input tensor with two channels:\n", X)
# 输出应为:
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]],
#          [[ 1.,  2.,  3.,  4.],
#           [ 5.,  6.,  7.,  8.],
#           [ 9., 10., 11., 12.],
#           [13., 14., 15., 16.]]]])

# 使用2D最大汇聚层，指定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("Max pooling on multiple channels:\n", pool2d(X))
# 输出应为:
# tensor([[[[ 5.,  7.],
#           [13., 15.]],
#          [[ 6.,  8.],
#           [14., 16.]]]])
