import torch
from torch import nn
from torch.nn import functional as F

# 1. 加载和保存张量
# 定义一个简单的张量
x = torch.arange(4)
# 保存张量到文件 'x-file'
torch.save(x, 'x-file')

# 从文件 'x-file' 中加载张量数据
x2 = torch.load('x-file')
print("加载的张量 x2:", x2)  # 输出: tensor([0, 1, 2, 3])

# 保存一个张量列表到文件 'x-files'
y = torch.zeros(4)
torch.save([x, y], 'x-files')

# 从文件 'x-files' 中加载张量列表
x2, y2 = torch.load('x-files')
print("加载的张量列表 (x2, y2):", (x2, y2))  # 输出: (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))

# 保存一个字典到文件 'mydict'
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')

# 从文件 'mydict' 中加载字典
mydict2 = torch.load('mydict')
print("加载的字典 mydict2:", mydict2)  # 输出: {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}


# 2. 加载和保存模型参数
# 定义一个简单的多层感知机(MLP)模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义隐藏层和输出层
        self.hidden = nn.Linear(20, 256)  # 隐藏层: 输入 20, 输出 256
        self.output = nn.Linear(256, 10)  # 输出层: 输入 256, 输出 10

    def forward(self, x):
        """
        前向传播函数: 通过隐藏层和输出层进行计算。
        Args:
            x (Tensor): 输入的张量
        Returns:
            Tensor: 网络的输出
        """
        # 使用ReLU激活函数进行计算
        return self.output(F.relu(self.hidden(x)))

# 创建MLP模型实例并生成一些随机输入数据
net = MLP()
X = torch.randn(size=(2, 20))  # 生成一个 2x20 的随机输入张量
Y = net(X)  # 使用模型进行前向传播计算
print("模型的输出 Y:", Y)

# 保存模型参数到文件 'mlp.params'
torch.save(net.state_dict(), 'mlp.params')

# 恢复模型：先创建一个相同结构的MLP模型实例
clone = MLP()

# 从文件 'mlp.params' 加载模型参数到克隆的模型实例中
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()  # 将模型设置为评估模式

print("加载的模型 clone:", clone)

# 验证两个模型在相同输入下的输出是否相同
Y_clone = clone(X)
print("两个模型的输出是否相同:", Y_clone == Y)  # 逐元素比较输出
