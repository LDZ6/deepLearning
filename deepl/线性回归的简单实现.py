import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 1. 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2. 数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 3. 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 4. 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 5. 定义损失函数
loss = nn.MSELoss()

# 6. 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 7. 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算预测值和真实值之间的损失
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播，计算梯度
        trainer.step()  # 更新模型参数
    l = loss(net(features), labels)  # 计算整个数据集上的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

# 8. 评估模型
w = net[0].weight.data
b = net[0].bias.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)
