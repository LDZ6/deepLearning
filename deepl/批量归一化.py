import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import train_ch6


# 批量规范化函数
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断是否处于训练模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式，使用移动平均的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:  # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:  # 卷积层
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        # 训练模式下标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和平移
    return Y, moving_mean.data, moving_var.data


# 自定义 BatchNorm 层
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:  # 全连接层
            shape = (1, num_features)
        else:  # 卷积层
            shape = (1, num_features, 1, 1)

        # 可训练参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 非模型参数 moving_mean 和 moving_var
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 将 moving_mean 和 moving_var 复制到 X 所在设备
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 调用批量规范化函数
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# net = nn.Sequential(
#      nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
#      nn.AvgPool2d(kernel_size=2, stride=2),
#      nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
#      nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#      nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
#      nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
#      nn.Linear(84, 10))



if __name__ == '__main__':
    # 构建带有批量规范化的 LeNet 网络
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    # 设置训练参数
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 开始训练
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    # 查看第一个批量规范化层学到的 gamma 和 beta 参数
    print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
