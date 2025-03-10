import torch
from torch import nn
from d2l import torch as d2l

# 二维互相关运算
def corr2d(X, K):  # @save
    """计算二维互相关运算

    参数:
    X -- 输入张量 (二维张量)
    K -- 卷积核张量 (二维张量)

    返回:
    Y -- 输出张量，表示X和K的二维互相关结果
    """
    # 获取卷积核的高度和宽度
    h, w = K.shape

    # 初始化输出张量Y，其大小取决于输入张量X和卷积核K的大小
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    # 对输出张量Y的每一个元素进行计算
    for i in range(Y.shape[0]):  # 遍历输出张量Y的行
        for j in range(Y.shape[1]):  # 遍历输出张量Y的列
            # 计算输出张量Y的每个元素，等于输入张量X的一个子区域与卷积核K按元素相乘的和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()

    return Y



# 示例输入张量X和卷积核张量K
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])

K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])

# 计算二维互相关运算结果
result = corr2d(X, K)

print(result)

# 定义一个二维卷积类，继承自nn.Module
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        """
        初始化二维卷积层

        参数:
        kernel_size -- 卷积核的大小 (tuple)，如 (2, 2)
        """
        super().__init__()
        # 定义卷积核的权重，使用随机初始化
        self.weight = nn.Parameter(torch.rand(kernel_size))
        # 定义偏置项，初始化为零
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播函数，用于计算卷积层的输出

        参数:
        x -- 输入张量 (二维张量)

        返回:
        x 与卷积核的二维互相关结果，加上偏置
        """
        # 计算输入与卷积核的互相关结果，并加上偏置
        return corr2d(x, self.weight) + self.bias



# 边缘检测示例
# 构造一个6x8像素的黑白图像，其中中间四列为黑色(0)，其余像素为白色(1)
X = torch.ones((6, 8))  # 全部初始化为白色(1)
X[:, 2:6] = 0  # 将中间的四列设置为黑色(0)
print("输入图像 X：\n", X)

# 构造一个高度为1、宽度为2的卷积核 K
K = torch.tensor([[1.0, -1.0]])
print("\n卷积核 K：\n", K)

# 对输入 X 和卷积核 K 执行互相关运算
Y = corr2d(X, K)
print("\n互相关运算结果 Y：\n", Y)

# 将输入 X 转置，再进行互相关运算
Y_transposed = corr2d(X.t(), K)
print("\n转置后的输入 X 进行互相关运算结果：\n", Y_transposed)



# 卷积层示例
# 构造一个6x8像素的黑白图像，其中中间四列为黑色(0)，其余像素为白色(1)
# 将 X 和 Y 的形状调整为四维（批量大小、通道、高度、宽度）
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

# 构造一个二维卷积层，具有1个输出通道，卷积核形状为 (1, 2)，不使用偏置
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 学习率
lr = 3e-2

# 迭代进行训练，使用梯度下降来优化卷积核权重
for i in range(10):
    # 前向传播
    Y_hat = conv2d(X)
    # 计算损失（平方误差）
    l = (Y_hat - Y) ** 2
    # 梯度清零
    conv2d.zero_grad()
    # 反向传播计算梯度
    l.sum().backward()
    # 更新卷积核权重
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    # 每隔2轮输出一次损失
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

# 查看学到的卷积核
print("学习到的卷积核：", conv2d.weight.data.reshape((1, 2)))