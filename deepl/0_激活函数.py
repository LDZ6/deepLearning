import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图

# 创建一个从 -8 到 8，步长为 0.1 的张量，并启用自动求导
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# 1. ReLU 函数及其梯度
# 应用 ReLU 激活函数
y = torch.relu(x)
# 使用 d2l 的 plot 函数来设置绘图格式，显示 ReLU 函数的图像
d2l.use_svg_display()  # 使得图像显示更清晰
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
plt.show()
# 计算 ReLU 函数的梯度
y.backward(torch.ones_like(x), retain_graph=True)
# 绘制 ReLU 函数梯度的图形
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
plt.show()

# pReLU(x) = max(0, x) + α min(0, x)

# 2. Sigmoid 函数及其梯度
# 清除之前的梯度（以免影响后续计算）
x.grad.data.zero_()
# 应用 Sigmoid 激活函数
y = torch.sigmoid(x)
# 显示 Sigmoid 函数的图像
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
plt.show()
# 计算 Sigmoid 函数的梯度
y.backward(torch.ones_like(x), retain_graph=True)
# 绘制 Sigmoid 函数梯度的图形
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
plt.show()




# 3. Tanh 函数及其梯度
# 清除之前的梯度
x.grad.data.zero_()
# 应用 Tanh 激活函数
y = torch.tanh(x)
# 显示 Tanh 函数的图像
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
plt.show()
# 计算 Tanh 函数的梯度
y.backward(torch.ones_like(x), retain_graph=True)
# 绘制 Tanh 函数梯度的图形
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
plt.show()
