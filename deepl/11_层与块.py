# 多层感知机（MLP）是神经网络的一种常见类型，它由多个全连接层组成，每层之间都有激活函数。
# import torch
# from torch import nn
# from torch.nn import functional as F
#
# # 定义神经网络模型
# net = nn.Sequential(
#     nn.Linear(20, 256),  # 输入层到隐藏层
#     nn.ReLU(),           # 激活函数
#     nn.Linear(256, 10)   # 隐藏层到输出层
# )
#
# # 生成输入数据
# X = torch.rand(2, 20)
#
# # 前向传播
# output = net(X)
# print(output)



# 定义MLP模型
# import torch
# from torch import nn
# from torch.nn import functional as F
#
# # 定义MLP模型
# class MLP(nn.Module):
#     # 模型的初始化
#     def __init__(self):
#         # 调用父类的初始化方法
#         super().__init__()
#         # 定义隐藏层
#         self.hidden = nn.Linear(20, 256)  # 输入层到隐藏层
#         # 定义输出层
#         self.out = nn.Linear(256, 10)  # 隐藏层到输出层
#
#     # 定义前向传播过程
#     def forward(self, X):
#         # 使用ReLU激活函数
#         hidden_output = F.relu(self.hidden(X))  # 隐藏层的输出
#         return self.out(hidden_output)  # 输出层的输出
#
# # 实例化模型
# net = MLP()
#
# # 生成输入数据
# X = torch.rand(2, 20)
#
# # 调用模型进行前向传播并输出结果
# output = net(X)
# print(output)



#`MySequential`类是自定义的`nn.Module`子类，它可以用来构建顺序模型。
import torch
from torch import nn


from torch.nn import functional as F

# 定义自定义的 MySequential 类
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 将每个传入的模块添加到有序字典中
        for idx, module in enumerate(args):
            # module是Module子类的一个实例，保存到'Module'类的成员变量_modules中
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

# 使用MySequential重新实现多层感知机
net = MySequential(
    nn.Linear(20, 256),  # 全连接层：输入20维，输出256维
    nn.ReLU(),           # ReLU激活函数
    nn.Linear(256, 10)   # 全连接层：输入256维，输出10维
)

# 生成输入数据
X = torch.rand(2, 20)

# 前向传播并输出结果
output = net(X)
print(output)



# 定义自定义的 FixedHiddenMLP 类
# 这个类包含一个隐藏层，权重固定，且不参与训练。
# 它还包含一个控制流，如果输出向量的 L1 范数大于 1，则将其除以 2，直到它满足条件。
import torch
from torch import nn
from torch.nn import functional as F

# 定义自定义的 FixedHiddenMLP 类
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        # 定义一个全连接层
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        # 应用线性变换
        X = self.linear(X)
        # 使用固定的权重进行矩阵乘法，并应用 ReLU 激活函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 再次使用全连接层，相当于共享参数
        X = self.linear(X)
        # 控制流：如果 L1 范数大于 1，将输出向量除以 2，直到它满足条件
        while X.abs().sum() > 1:
            X /= 2
        # 返回输出张量中所有元素的和
        return X.sum()

# 实例化模型
net = FixedHiddenMLP()

# 生成输入数据
X = torch.rand(2, 20)

# 调用模型进行前向传播并输出结果
output = net(X)
print(output)



# 定义自定义的嵌套多层感知机类 NestMLP
import torch
from torch import nn
from torch.nn import functional as F

# 定义自定义的嵌套多层感知机类 NestMLP
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 嵌套的神经网络部分
        self.net = nn.Sequential(
            nn.Linear(20, 64),  # 输入层到隐藏层1
            nn.ReLU(),          # 隐藏层1的激活函数
            nn.Linear(64, 32),  # 隐藏层1到隐藏层2
            nn.ReLU()           # 隐藏层2的激活函数
        )
        # 额外的线性层
        self.linear = nn.Linear(32, 16)  # 隐藏层2到输出层

    def forward(self, X):
        # 首先通过嵌套的网络 self.net，然后通过线性层 self.linear
        return self.linear(self.net(X))

# 实例化嵌套模型 NestMLP 和其他模型
chimera = nn.Sequential(
    NestMLP(),                    # 嵌套的 MLP
    nn.Linear(16, 20),            # 线性层
    FixedHiddenMLP()              # 固定隐藏层的 MLP
)

# 生成输入数据
X = torch.rand(2, 20)

# 调用组合模型 chimera 进行前向传播并输出结果
output = chimera(X)
print(output)
