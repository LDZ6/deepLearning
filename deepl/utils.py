import re
import random
import torch
from d2l import torch as d2l
import collections
import re
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)  # 稳定计算softmax
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

# 定义模型

# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
# 定义数据集 URL 和哈希
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

def read_time_machine(): #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 清理数据：去除非字母字符，转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 定义分类精度计算
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 定义模型评估函数
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上的准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 累加器类，用于累计损失和精度
class Accumulator:
    """在n个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        """向图表中添加多个数据点"""
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
        self.axes[0].cla()  # 清除当前图形
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)
        self.config_axes()


# 训练模型一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
    plt.show()



def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    # 获取第一个批次的图像和标签
    for X, y in test_iter:
        break

    # 获取真实标签
    trues = d2l.get_fashion_mnist_labels(y)

    # 获取预测标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))

    # 生成标题
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    # 显示图像
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
    plt.tight_layout()
    plt.show()  # 在 PyCharm 中显示图像

import hashlib
import os
import tarfile
import zipfile
import requests

# 数据中心，用于存储数据集的下载链接和对应的sha-1哈希值
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 如果文件存在，且sha1值匹配，则使用缓存文件
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存

    # 下载文件
    print(f'正在从 {url} 下载 {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)

    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'

    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    plt.show()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 定义数据集 URL 和哈希
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

# 读取《时间机器》文本并清理数据
def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 清理数据：去除非字母字符，转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 词元化函数：支持按单词或字符进行分词
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 统计词频
def count_corpus(tokens):
    """统计词元的频率"""
    # 处理二维列表或一维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 定义词表类
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频并排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元索引为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 添加满足最小词频要求的词元
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """根据词元返回其索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """根据索引返回词元"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引为 0"""
        return 0

    @property
    def token_freqs(self):
        """返回词元频率"""
        return self._token_freqs


# 加载和处理《时间机器》数据集
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 使用字符进行词元化
    vocab = Vocab(tokens)  # 构建词表
    # 将所有文本行展平为一个词元列表
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# 随机采样生成小批量子序列
def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。

    Args:
        corpus (list or array): 输入的文本序列（通常是整数索引形式）。
        batch_size (int): 每个小批量的样本数量。
        num_steps (int): 每个序列的时间步长。

    Yields:
        tuple: 包含输入序列 (X) 和目标序列 (Y) 的小批量。
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps - 1)  # 确保偏移量小于 num_steps
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    # 切分 corpus 为 Xs 和 Ys
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # 调整为 (batch_size, -1) 的形状
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)

    # 计算每个批量中包含的时间步长
    num_batches = Xs.shape[1] // num_steps

    # 生成小批量
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """
        初始化数据加载器
        参数：
        batch_size: 每个小批量的大小
        num_steps: 每次采样序列的长度
        use_random_iter: 是否使用随机采样
        max_tokens: 词汇表的最大长度
        """
        # 根据是否随机迭代，选择对应的数据采样函数
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential

        # 加载语料库和词汇表
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """返回数据迭代器"""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):  # @save
    """
    返回时光机器数据集的迭代器和词表
    参数：
    batch_size: 每个小批量的大小
    num_steps: 每次采样序列的长度
    use_random_iter: 是否使用随机采样
    max_tokens: 词汇表的最大长度
    返回：
    data_iter: 数据迭代器
    vocab: 词表
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


class RNNModel(nn.Module):
    """循环神经网络模型（RNN）。"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """初始化 RNN 模型。

        参数：
            rnn_layer: 一个 RNN 层（如 nn.LSTM 或 nn.GRU）。
            vocab_size: 词汇表的大小。
        """
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer  # 循环神经网络层（LSTM 或 GRU）
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_hiddens = self.rnn.hidden_size  # 隐藏层大小

        # 判断是否是双向 RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1  # 单向 RNN
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)  # 全连接层
        else:
            self.num_directions = 2  # 双向 RNN
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)  # 双向时需要乘以 2

    def forward(self, inputs, state):
        """前向传播函数。

        参数：
            inputs: 输入数据，形状为 (batch_size, time_steps)。
            state: 初始隐藏状态。

        返回：
            output: RNN 输出，形状为 (time_steps * batch_size, vocab_size)。
            state: 更新后的隐藏状态。
        """
        # 将输入转换为 one-hot 编码并转为 float32 类型
        X = F.one_hot(inputs.T.long(), self.vocab_size)  # 转置输入
        X = X.to(torch.float32)  # 转为 float32

        # 通过 RNN 层计算输出和新的隐藏状态
        Y, state = self.rnn(X, state)

        # 将 RNN 的输出 Y 从 (time_steps, batch_size, hidden_size) 转换为 (time_steps * batch_size, hidden_size)
        # 然后通过全连接层，输出 (time_steps * batch_size, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """初始化隐藏状态。

        参数：
            device: 计算设备（'cpu' 或 'cuda'）。
            batch_size: 每批数据的样本数。

        返回：
            初始隐藏状态。
        """
        if not isinstance(self.rnn, nn.LSTM):
            # 对于 GRU，隐藏状态是一个张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # 对于 LSTM，隐藏状态是一个元组
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))



def get_params(vocab_size, num_hiddens, device):
    """
    初始化 RNN 的参数。
    参数：
    - vocab_size: 词汇表大小（输入和输出的维度）。
    - num_hiddens: 隐藏单元的数量。
    - device: 使用的设备（如 'cpu' 或 'cuda'）。
    返回：
    - params: 包含所有模型参数的列表。
    """
    num_inputs = num_outputs = vocab_size
    # 初始化参数的函数
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏层的权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏层到隐藏层的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层的偏置

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层到输出层的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层的偏置

    # 将所有参数放入列表并附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """
    简单的 RNN 前向传播。
    参数：
    - inputs: 输入序列的张量，形状为 (时间步数量, 批量大小, 词表大小)。
    - state: 初始隐藏状态，形状为 (1, 批量大小, 隐藏单元数)。
    - params: 包含 RNN 所有参数的列表，包含 (W_xh, W_hh, b_h, W_hq, b_q)。
    返回：
    - outputs: 经过 RNN 层处理后的输出序列，形状为 (时间步数量, 批量大小, 词表大小)。
    - state: 最终的隐藏状态，形状为 (1, 批量大小, 隐藏单元数)。
    """
    # 解包参数
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 隐藏状态 H，形状为 (批量大小, 隐藏单元数)

    outputs = []  # 用于存储每个时间步的输出

    # 遍历每个时间步的输入 X
    for X in inputs:
        # 计算当前时间步的隐藏状态 H
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)

        # 计算当前时间步的输出 Y
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)

    # 将所有时间步的输出连接成一个大张量
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        """
        初始化 RNN 模型。

        参数：
        - vocab_size: 词汇表的大小。
        - num_hiddens: 隐藏层单元的数量。
        - device: 使用的设备（如 'cpu' 或 'cuda'）。
        - get_params: 用于获取模型参数的函数。
        - init_state: 用于初始化隐藏状态的函数。
        - forward_fn: 用于前向传播的函数。
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)  # 获取模型参数
        self.init_state, self.forward_fn = init_state, forward_fn  # 保存初始化状态和前向传播函数

    def __call__(self, X, state):
        """
        定义模型的前向传播。

        参数：
        - X: 输入数据。
        - state: 隐藏状态。
   返回：
        - 模型输出和新的隐藏状态。
        """
        # 将输入转化为独热编码（one-hot encoding），并转置
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """
        初始化隐藏状态。

        参数：
        - batch_size: 批量大小。
        - device: 使用的设备（如 'cpu' 或 'cuda'）。

        返回：
        - 初始的隐藏状态。
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """.
    在给定的前缀（prefix）后生成新字符。

    参数：
    - prefix: 生成文本的前缀字符串。
    - num_preds: 需要生成的字符数。
    - net: 训练好的 RNN 模型。
    - vocab: 词汇表，提供字符到索引和索引到字符的映射。
    - device: 计算设备（如 'cpu' 或 'cuda'）。

    返回：
    - 生成的完整字符串（包含 prefix 和生成的字符）。
    """
    # 初始化隐藏状态，batch_size=1 表示单个样本
    state = net.begin_state(batch_size=1, device=device)

    # 初始化输出列表，并将 prefix 的第一个字符对应的索引添加到输出中
    outputs = [vocab[prefix[0]]]

    # 获取输入的 lambda 函数，用于生成下一个时间步的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 根据 prefix 进行“预热”，逐步更新隐藏状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 输入当前字符，更新隐藏状态
        outputs.append(vocab[y])  # 将当前字符对应的索引添加到输出中

    # 生成 num_preds 个字符
    for _ in range(num_preds):
        # 根据当前输入生成下一个字符
        y, state = net(get_input(), state)

        # 获取预测的字符索引，并添加到输出列表中
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    # 将索引转换为字符，并拼接成字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度以避免梯度爆炸。

    参数:
        net (nn.Module 或其他自定义网络对象): 神经网络模型。
        theta (float): 阈值，若梯度范数超过该值则进行裁剪。

    """
    # 如果网络是 PyTorch 的 nn.Module 类型
    if isinstance(net, nn.Module):
        # 获取需要计算梯度的参数
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 如果是其他类型（例如自定义实现），获取其参数
        params = net.params

    # 计算梯度的 L2 范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 如果梯度范数超过阈值，则进行裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（适用于循环神经网络）。

    参数:
        net: 循环神经网络模型。
        train_iter: 训练数据的迭代器。
        loss: 损失函数。
        updater: 优化器或自定义更新函数。
        device: 计算设备（如 'cpu' 或 'cuda'）。
        use_random_iter: 是否使用随机采样。

    返回:
        perplexity: 困惑度（语言模型常用指标）。
        speed: 每秒处理的词元数量。
    """
    # 初始化隐藏状态和计时器
    state, timer = None, d2l.Timer()
    # 用于累加训练损失和词元数量
    metric = d2l.Accumulator(2)  # [训练损失之和, 词元数量]

    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次迭代或使用随机采样时初始化隐藏状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 如果是 nn.Module 并且状态是张量，则直接 detach
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                # 对于 nn.LSTM 或自定义实现，状态可能是元组，需要逐一 detach
                for s in state:
                    s.detach_()

        # 转换标签形状并移动数据到指定设备
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)

        # 前向传播计算预测值和新的隐藏状态
        y_hat, state = net(X, state)
        # 计算损失并取平均值
        l = loss(y_hat, y.long()).mean()

        # 梯度清零、反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)  # 裁剪梯度
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)  # 裁剪梯度
            # 使用自定义的更新函数，指定 batch_size=1
            updater(batch_size=1)

        # 累加损失和词元数量
        metric.add(l * y.numel(), y.numel())

    # 计算困惑度和每秒处理的词元数量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（定义见第8章）。

    参数:
        net: 神经网络模型。
        train_iter: 训练数据的迭代器。
        vocab: 词汇表。
        lr: 学习率。
        num_epochs: 训练的迭代周期数。
        device: 计算设备（'cpu' 或 'cuda'）。
        use_random_iter: 是否使用随机采样。

    """
    loss = nn.CrossEntropyLoss()  # 损失函数
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])

    # 初始化优化器
    if isinstance(net, nn.Module):
        # 对于 PyTorch 模型，使用 SGD 优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 对于自定义模型，使用自定义的 SGD 更新函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    # 定义预测函数
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    # 训练和预测
    for epoch in range(num_epochs):
        # 训练一个迭代周期
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)

        # 每10个epoch打印一次预测结果
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))

        # 更新动画图
        animator.add(epoch + 1, [ppl])

        # 打印困惑度和每秒处理的词元数量
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')

        # 打印模型的预测
        print(predict('time traveller'))
        print(predict('traveller'))