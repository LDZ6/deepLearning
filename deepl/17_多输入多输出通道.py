import torch
from d2l import torch as d2l


# 1. 实现多输入通道的2D互相关运算
# 该函数对每个输入通道执行2D互相关运算，然后将所有结果相加
def corr2d_multi_in(X, K):
    # 对输入X和核K的每个通道执行2D互相关，并将结果相加
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


# 2. 测试多输入通道的互相关运算
# 创建输入张量X，有2个通道，每个通道的大小为3x3
X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],

                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])

# 创建核张量K，有2个通道，每个通道的大小为2x2
K = torch.tensor([[[0.0, 1.0],
                   [2.0, 3.0]],

                  [[1.0, 2.0],
                   [3.0, 4.0]]])

# 执行多输入通道的互相关运算
result = corr2d_multi_in(X, K)
print(result)


# 输出应为 tensor([[ 56.,  72.],
#                 [104., 120.]])

# 3. 实现多输出通道的2D互相关运算
# 该函数对每个输出通道的核执行多输入通道的互相关运算
def corr2d_multi_in_out(X, K):
    # 对核张量K的每个输出通道执行多输入通道的互相关运算，并将结果堆叠在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 测试多输出通道的互相关运算
# 构造具有3个输出通道的卷积核张量K
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)  # 输出应为 torch.Size([3, 2, 2, 2])

# 执行多输出通道的互相关运算
result = corr2d_multi_in_out(X, K)
print(result)


# 输出应为 tensor([[[ 56.,  72.],
#                   [104., 120.]],

#                  [[ 76., 100.],
#                   [148., 172.]],

#                  [[ 96., 128.],
#                   [192., 224.]]])

# 4. 实现1x1卷积层的多输入多输出运算
# 该函数将1x1卷积层的计算转化为矩阵乘法
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape  # 输入通道数，高度，宽度
    c_o = K.shape[0]  # 输出通道数
    X = X.reshape((c_i, h * w))  # 将输入X展平成 (ci, h*w)
    K = K.reshape((c_o, c_i))  # 将核K展平成 (co, ci)

    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))  # 将输出Y重塑为 (co, h, w)


# 测试1x1卷积层的运算与多输入多输出通道的互相关运算结果一致性
X = torch.normal(0, 1, (3, 3, 3))  # 创建一个随机输入张量X
K = torch.normal(0, 1, (2, 3, 1, 1))  # 创建一个随机核张量K
Y1 = corr2d_multi_in_out_1x1(X, K)  # 使用1x1卷积运算
Y2 = corr2d_multi_in_out(X, K)  # 使用多输入多输出通道的互相关运算
#
# # 验证两者结果是否一致
# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6  # 如果断言失败则会抛出异常
