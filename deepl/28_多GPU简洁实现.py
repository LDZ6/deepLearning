import torch
from torch import nn
from d2l import torch as d2l


# 自定义的 Residual 类
class Residual(nn.Module):
    """Residual block used in ResNet"""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()

        if use_1x1conv:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the shortcut connection
        out = self.relu(out)
        return out


# 12.6.1 简单网络：稍加修改的 ResNet-18 模型
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))

    return net


# 12.6.2 网络初始化
net = resnet18(10)

# 获取 GPU 列表
devices = d2l.try_all_gpus()


# 网络初始化函数
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, std=0.01)


# 初始化网络权重
net.apply(init_weights)


# 12.6.3 训练
def train(net, num_gpus, batch_size, lr):
    # 加载数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 将模型转移到多个 GPU 上
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    net = nn.DataParallel(net, device_ids=devices)  # 多 GPU 模型并行

    # 定义优化器和损失函数
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()

    # 计时器和其他训练参数
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])

    # 训练过程
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])  # 将数据移动到第一个 GPU
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()

        # 每个 epoch 结束后评估精度
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
        print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
              f'在{str(devices)}')


# 训练网络
# 在单个GPU上训练网络进行预热
train(net, num_gpus=1, batch_size=256, lr=0.1)

# 使用2个GPU进行训练
train(net, num_gpus=2, batch_size=512, lr=0.2)
