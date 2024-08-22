import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import train_ch3, predict_ch3

# 定义神经网络
net = nn.Sequential(
    nn.Flatten(),         # 将输入展平为一维
    nn.Linear(784, 256), # 第一个全连接层，输入784维，输出256维
    nn.ReLU(),           # ReLU 激活函数
    nn.Linear(256, 10)   # 第二个全连接层，输入256维，输出10维
)

# 权重初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

# 应用权重初始化
net.apply(init_weights)
# 定义超参数
batch_size, lr, num_epochs = 256, 0.1, 10
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 定义优化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# 加载数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 训练模型
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 进行预测
predict_ch3(net, test_iter)
