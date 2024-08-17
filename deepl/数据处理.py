import numpy as np
from matplotlib import pyplot as plt
import torch

# def f(x):
#     return 3 * x ** 2 - 4 * x
#
# def numerical_lim(f, x, h):
#     return (f(x + h) - f(x)) / h
#
# def set_figsize(figsize=(3.5, 2.5)): #@save
#     """设置matplotlib的图表大小"""
#     plt.rcParams['figure.figsize'] = figsize
#
# #@save
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     """设置matplotlib的轴"""
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()
#
# # @save
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []
#
#     set_figsize(figsize)
#     axes = axes if axes else plt.gca()
#
#     # 如果X有一个轴，输出True
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1) or (isinstance(X, list) and not hasattr(X[0], "__len__"))
#
#     if has_one_axis(X):
#         X = [X]
#
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#
#     if len(X) != len(Y):
#         X = X * len(Y)
#
#     axes.cla()
#
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#     plt.show()

# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])






#自动微分
import torch
#
# # 创建一个张量并设置 requires_grad=True 以追踪其梯度
# x = torch.arange(4.0)
# x.requires_grad_(True)
#
# # 打印张量 x
# print("x:", x)
#
# # 计算 y = 2 * torch.dot(x, x)
# y = 2 * torch.dot(x, x)
# print("y = 2 * torch.dot(x, x):", y)
#
# # 计算 y 关于 x 的梯度
# y.backward()
# print("x.grad after y.backward():", x.grad)
#
# # 验证梯度是否正确，梯度应该是 4 * x
# print("x.grad == 4 * x:", x.grad == 4 * x)
#
# # 计算新的函数 y = x.sum()
# x.grad.zero_()  # 清除之前的梯度
# y = x.sum()
# y.backward()
# print("x.grad after y.sum().backward():", x.grad)
#
# # 计算 y = x * x
# x.grad.zero_()  # 清除之前的梯度
# y = x * x
# print(y)
# # 对非标量张量 y 调用 backward 需要传递一个 gradient 参数
# # 这里我们传递一个全 1 的张量
# y.backward(torch.ones_like(x))  # 计算偏导数的和
# print("x.grad after y.sum().backward() with gradient:", x.grad)





#概率论
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# fair_probs = torch.ones([6]) / 6
# multinomial.Multinomial(1, fair_probs).sample()
#
# # 将结果存储为32位浮点数以进⾏除法
# counts = multinomial.Multinomial(1000, fair_probs).sample()
# print(counts / 1000)  # 相对频率作为估计值

import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt

# 定义一个公平的六面骰子的概率分布
fair_probs = torch.ones(6) / 6

# 抽取500组，每组投掷10次的样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))

# 计算累计次数
cum_counts = counts.cumsum(dim=0)


# 估计每个点的概率
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

# 设置图像大小
plt.figure(figsize=(6, 4.5))

# 绘制每个点的概率估计
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=f'P(die={i + 1})')

# 绘制真实概率的水平线
plt.axhline(y=1/6, color='black', linestyle='dashed')

# 设置图表标签
plt.xlabel('Groups of experiments')
plt.ylabel('Estimated probability')

# 显示图例
plt.legend()

# 展示图表
plt.show()
