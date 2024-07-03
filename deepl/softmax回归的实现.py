import torch
from IPython import display
from d2l import torch as d2l



def softmax(X):#@save
    """
    对输入进行softmax运算
    Args:
        X: 输入张量，形状为(batch_size, num_classes)
    Returns:
        输出张量，形状为(batch_size, num_classes)，每个元素取值范围在(0, 1)之间且每一行元素和为1
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):#@save
    """
    定义softmax回归模型
    Args:
        X: 输入特征张量，形状为(batch_size, num_features)
    Returns:
        输出张量，形状为(batch_size, num_classes)，每个元素取值范围在(0, 1)之间且每一行元素和为1
    """
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):#@save
    """
    计算交叉熵损失
    Args:
        y_hat: 模型的预测结果张量，形状为(batch_size, num_classes)，每一行代表一个样本的预测概率分布
        y: 真实标签张量，形状为(batch_size,)
    Returns:
        交叉熵损失张量，形状为(batch_size,)，其中每个元素是对应样本的交叉熵损失值
    """
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):#@save
    """
    计算预测正确的数量
    Args:
        y_hat: 模型的预测结果张量，形状为(batch_size, num_classes)，每一行代表一个样本的预测概率分布
        y: 真实标签张量，形状为(batch_size,)
    Returns:
        预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):#@save
    """
    计算在指定数据集上模型的精度
    Args:
        net: 神经网络模型
        data_iter: 数据集迭代器
    Returns:
        模型在指定数据集上的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 创建一个累加器来保存正确预测数和总预测数
    with torch.no_grad():  # 禁用梯度跟踪
        for X, y in data_iter:  # 遍历数据集
            metric.add(accuracy(net(X), y), y.numel())  # 使用累加器记录正确预测数和总预测数
    return metric[0] / metric[1]  # 返回模型在指定数据集上的精度

def train_epoch_ch3(net, train_iter, loss, updater):#@save
    """
    训练模型一个迭代周期
    Args:
        net: 模型
        train_iter: 训练数据集迭代器
        loss: 损失函数
        updater: 更新参数的函数或者优化器
    Returns:
        训练损失和训练精度的元组，(train_loss, train_acc)
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)

    for X, y in train_iter:
        # 计算模型的预测值
        y_hat = net(X)

        # 计算损失
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        # 统计训练损失、训练精度和样本数量
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):#@save
    """
    Parameters:
        net : torch.nn.Module
            要训练的神经网络模型
        train_iter : torch.utils.data.DataLoader
            训练数据集的数据迭代器
        test_iter : torch.utils.data.DataLoader
            测试数据集的数据迭代器
        loss : torch.nn.Module
            损失函数
        num_epochs : int
            训练的轮数
        updater : callable
            更新模型参数的函数

    Returns:
        None
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc

class Accumulator:#@save
    """
    在n个变量上累加
    """
    def __init__(self, n):
        """
        初始化累加器对象
        Args:
            n: 变量数量
        """
        self.data = [0.0] * n  # 创建一个长度为n的列表，初始值全部为0.0

    def add(self, *args):
        """
        向累加器中添加数据，进行累加
        Args:
            *args: 要添加到累加器中的数据，数量必须与初始化时的变量数量相同
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 对累加器中的每个变量进行累加

    def reset(self):
        """重置累加器，将所有变量清零"""
        self.data = [0.0] * len(self.data)  # 将累加器中的所有变量重置为0.0

    def __getitem__(self, idx):
        """
        获取累加器中指定索引位置的变量的值
        Args:
            idx: 变量的索引
        Returns:
            累加器中指定索引位置的变量的值
        """
        return self.data[idx]  # 返回累加器中指定索引位置的变量的值

class Animator:#@save
    """
    在动画中绘制数据
    """

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """
        初始化函数
        Args:
            xlabel: x轴标签
            ylabel: y轴标签
            legend: 图例
            xlim: x轴范围
            ylim: y轴范围
            xscale: x轴刻度的缩放
            yscale: y轴刻度的缩放
            fmts: 线条的样式
            nrows: 子图的行数
            ncols: 子图的列数
            figsize: 图形尺寸
        """
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """
        向图表中添加多个数据点
        Args:
            x: x轴数据点
            y: y轴数据点
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)




batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 实现softmax由三个步骤组成：
# 1. 对每个项求幂（使⽤exp）；
# 2. 对每⼀⾏求和（⼩批量中每个样本是⼀⾏），得到每个样本的规范化常数；
# 3. 将每⼀⾏除以其规范化常数，确保结果的和为1。

# 训练
lr = 0.1


def updater(batch_size):#@save
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):#@save
    """
    Parameters:
        net : torch.nn.Module
            训练好的神经网络模型
        test_iter : torch.utils.data.DataLoader
            测试数据集的数据迭代器
        n : int
            预测并展示的样本数量，默认为6

    Returns:
        None
    """
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
