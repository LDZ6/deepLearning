import torch
import torch.nn.functional as F
from torch import nn

# 定义一个没有参数的自定义层 CenteredLayer
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的初始化函数

    def forward(self, X):
        """
        前向传播函数: 从输入 X 中减去均值。
        Args:
            X (Tensor): 输入的张量
        Returns:
            Tensor: 返回减去均值后的张量
        """
        return X - X.mean()

# 创建 CenteredLayer 实例并进行测试
layer = CenteredLayer()
output = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print("CenteredLayer输出:", output)

# 将 CenteredLayer 作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

# 测试新网络的输出均值是否为零
Y = net(torch.rand(4, 8))
print("新网络输出的均值:", Y.mean())

# 定义一个带参数的自定义层 MyLinear
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        """
        自定义全连接层的初始化函数。
        Args:
            in_units (int): 输入维度
            units (int): 输出维度
        """
        super().__init__()  # 调用父类的初始化函数
        # 初始化权重和偏置参数，使用均值为0，标准差为1的正态分布
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        """
        前向传播函数: 计算线性变换并应用ReLU激活函数。
        Args:
            X (Tensor): 输入的张量
        Returns:
            Tensor: 应用ReLU后的输出张量
        """
        # 计算线性变换：X @ W + b
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        # 应用ReLU激活函数并返回
        return F.relu(linear)

# 实例化 MyLinear 类并访问其模型参数
linear = MyLinear(5, 3)
print("自定义层的权重参数:", linear.weight)

# 使用自定义层直接执行前向传播计算
output = linear(torch.rand(2, 5))
print("自定义层的输出:", output)

# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
output = net(torch.rand(2, 64))
print("自定义网络的输出:", output)
