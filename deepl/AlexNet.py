import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import train_ch6

# 定义 AlexNet
net = nn.Sequential(
    # 使用一个 11x11 的大卷积核来捕捉对象
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 减小卷积窗口，增加输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 使用三个连续的卷积层和较小的卷积窗口
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(),

    # 全连接层和 Dropout
    nn.Linear(256 * 5 * 5, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    # 输出层
    nn.Linear(4096, 10)
)

# 打印每一层的输出形状
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

if __name__ == '__main__':
    # 读取 Fashion-MNIST 数据集
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # 训练 AlexNet
    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
