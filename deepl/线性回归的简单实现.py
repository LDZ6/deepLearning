import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l



def load_array(data_arrays, batch_size, is_train=True):
    """
    构造⼀个PyTorch数据迭代器
    参数:
        data_arrays: 包含输入特征和标签的数据数组元组
        batch_size: 批量大小
        is_train: 是否为训练数据集，默认为True
    返回:
        数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)  # 创建数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回数据迭代器



true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
#定义模型
net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 从均值为0、标准差为0.01的正态分布中随机初始化权重参数
net[0].bias.data.fill_(0)  # 初始化偏置参数为零
# 定义损失函数
loss = nn.MSELoss()#计算均⽅误差使⽤的是MSELoss类，也称为平⽅L2范数。默认情况下，它返回所有样本损失的平均值。
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算当前批次的损失
        trainer.zero_grad()  # 清空梯度
        l.backward()  # 反向传播，计算梯度
        trainer.step()  # 更新模型参数
    l = loss(net(features), labels)  # 计算整个数据集的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)




