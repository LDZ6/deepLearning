import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from d2l import torch as d2l

import timer


def synthetic_data(w, b, num_examples):
    """
    生成y=Xw+b+噪声的合成数据集
    参数:
        w: 权重张量
        b: 偏置值
        num_examples: 数据集中的样本数量

    返回:
        X: 输入特征矩阵，形状为(num_examples, len(w))
        y: 标签向量，形状为(num_examples, 1)
    """
    # 生成服从标准正态分布的随机数作为特征矩阵 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算标签向量 y
    y = torch.matmul(X, w) + b
    # 添加服从均值为0，标准差为0.01的正态分布噪声
    y += torch.normal(0, 0.01, y.shape)

    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """
    生成一个迭代器，用于按批次读取数据
    参数:
        batch_size: 每个批次的大小
        features: 输入特征张量
        labels: 输出标签张量

    返回:
        返回一个迭代器，每次迭代产生一个批次的特征和标签数据
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    """
    线性回归模型
    参数:
        X: 输入特征张量
        w: 权重张量
        b: 偏置张量

    返回:
        返回线性回归模型的输出结果
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """
    均方损失
    参数:
        y_hat: 预测值张量
        y: 实际值张量

    返回:
        返回均方损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    参数:
        params: 待优化参数列表
        lr: 学习率
        batch_size: 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



# 定义真实的权重和偏差
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 调用函数生成数据集
features, labels = synthetic_data(true_w, true_b, 1000)



#通过⽣成第⼆个特征features[:, 1]和labels的散点图，直观观察到两者之间的线性关系。
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
# plt.show()



# 初始化权重 w，采用正态分布随机初始化，均值为 0，标准差为 0.01
# size=(2,1) 表示 w 是一个形状为 (2,1) 的张量，即一个二维列向量，其中每个元素都是从均值为 0、标准差为 0.01 的正态分布中随机采样得到的
# requires_grad=True 表示我们会在训练过程中自动求导计算梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# 初始化偏置 b 为零
# requires_grad=True 表示我们会在训练过程中自动求导计算梯度
b = torch.zeros(1, requires_grad=True)



#训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使⽤参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')