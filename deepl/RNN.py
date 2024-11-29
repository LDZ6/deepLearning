import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from deepl.utils import load_data_time_machine


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


# 设置批处理大小和序列长度
batch_size, num_steps = 32, 35
# 加载时间机器数据集
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

# 设置超参数
num_epochs = 500  # 训练的迭代周期数
lr = 1  # 学习率
# 开始训练
# 开始训练
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
plt.show()
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
plt.show()




#简洁实现
rnn_layer = nn.RNN(len(vocab), num_hiddens)
state = torch.zeros((1, batch_size, num_hiddens))

import torch
import torch.nn as nn
import torch.nn.functional as F


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


device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(predict_ch8('time traveller', 10, net, vocab, device))

train_ch8(net, train_iter, vocab, lr, num_epochs, device)