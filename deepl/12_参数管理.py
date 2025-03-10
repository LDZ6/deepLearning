import torch
from torch import nn

# 定义一个具有单隐藏层的多层感知机（MLP）模型
net = nn.Sequential(
    nn.Linear(4, 8),  # 第一层，全连接层，将输入的4个特征映射到8个特征
    nn.ReLU(),        # 激活函数ReLU
    nn.Linear(8, 1)   # 第二层，全连接层，将8个特征映射到1个输出
)

# 生成一个随机输入张量，大小为 (2, 4)
X = torch.rand(size=(2, 4))
print(net(X))  # 前向传播，通过模型得到输出

# 5.2.1 参数访问

# 访问第二个全连接层的参数
print("第二个全连接层的参数:", net[2].state_dict())

# 提取偏置参数，并查看其类型和数值
print("偏置参数类型:", type(net[2].bias))
print("偏置参数内容:", net[2].bias)
print("偏置参数数值:", net[2].bias.data)

# 查看权重梯度（未经过反向传播时梯度为None）
print("权重梯度状态:", net[2].weight.grad)

# 一次性访问所有参数
print("第一个全连接层的参数:", *[(name, param.shape) for name, param in net[0].named_parameters()])
print("所有层的参数:", *[(name, param.shape) for name, param in net.named_parameters()])

# 使用 state_dict 方法访问参数
print("第二个全连接层偏置的数值:", net.state_dict()['2.bias'].data)

# 从嵌套块收集参数

# 定义一个包含两个线性层的块
def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

# 将多个块组合成更大的块
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())  # 嵌套块
    return net

# 使用嵌套块创建新的网络
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))  # 前向传播

# 查看嵌套网络的结构
print("嵌套网络的结构:", rgnet)

# 访问嵌套块的参数
print("嵌套块中第一个主要块的第二个子块的第一个层的偏置项:", rgnet[0][1][0].bias.data)

# 5.2.2 参数初始化
# 使用自定义初始化方法初始化模型参数
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 正态分布初始化权重
        nn.init.zeros_(m.bias)                      # 将偏置初始化为0

net.apply(init_normal)  # 将初始化方法应用于网络
print("初始化后的权重和偏置:", net[0].weight.data[0], net[0].bias.data[0])

# 使用常数初始化
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)  # 将权重初始化为常数1
        nn.init.zeros_(m.bias)          # 将偏置初始化为0

net.apply(init_constant)
print("常数初始化后的权重和偏置:", net[0].weight.data[0], net[0].bias.data[0])

# 使用 Xavier 初始化和常数初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)  # 将权重初始化为常数42

net[0].apply(init_xavier)
net[2].apply(init_42)
print("Xavier 初始化后的权重:", net[0].weight.data[0])
print("常数42初始化后的权重:", net[2].weight.data)

# 自定义初始化方法
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)  # 均匀分布初始化权重
        m.weight.data *= m.weight.data.abs() >= 5  # 保留绝对值大于等于5的部分

net.apply(my_init)
print("自定义初始化后的权重:", net[0].weight[:2])

# 直接设置参数值
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print("手动设置后的权重:", net[0].weight.data[0])

# 5.2.3 参数绑定

# 创建一个共享的层
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,  # 使用共享的层
    nn.ReLU(),
    shared,  # 再次使用共享的层
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 检查参数是否相同
print("检查共享层的权重是否相同:", net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print("修改共享层的权重后再检查:", net[2].weight.data[0] == net[4].weight.data[0])
