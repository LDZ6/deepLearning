import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('../img/catdog.jpg')
d2l.plt.imshow(img);

# 定义辅助函数apply用于可视化增广效果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    plt.show()

# 左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变颜色
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 多种增广组合
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 13.1.2 使用图像增广进行训练

# 加载CIFAR-10数据集
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.show()

# 图像增广：训练集进行随机左右翻转，测试集仅转换为tensor
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

# 定义加载CIFAR-10数据集的辅助函数
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                            transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

# 13.1.3 多GPU训练

# 定义批量训练函数，支持多GPU
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """ 用多GPU进行小批量训练 """
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

# 定义多GPU训练函数
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    """ 用多GPU进行模型训练 """
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # 使用多GPU并行训练
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)  # 4个维度：训练损失、训练准确率、实例数、特征数
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(devices)}')

# 定义训练函数，使用图像增广训练ResNet-18
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size=256)
    test_iter = load_cifar10(False, test_augs, batch_size=256)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs=10, devices=d2l.try_all_gpus())
# 创建一个虚拟输入，进行前向传播，初始化 Lazy 模块

dummy_input = torch.zeros(1, 3, 32, 32)  # CIFAR-10 的输入大小
# ResNet-18模型
net = d2l.resnet18(10, 3)
net(dummy_input)  # 执行一次前向传播，初始化参数
# 初始化网络参数

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

# 使用基于随机左右翻转的图像增广来训练模型
train_with_data_aug(train_augs, test_augs, net)
