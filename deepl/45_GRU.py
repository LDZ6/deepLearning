import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import load_data_time_machine, RNNModelScratch, train_ch8, RNNModel


def get_params(vocab_size, num_hiddens, device):
    # 输入和输出维度与词汇表大小相同
    num_inputs = num_outputs = vocab_size

    # 定义正态分布的初始化函数
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 返回三组参数：门控参数
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 更新门参数（门控网络的参数）
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 输出层权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出偏置

    # 参数列表，包含所有需要训练的参数
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    # 设置所有参数的梯度追踪
    for param in params:
        param.requires_grad_(True)

    return params

def gru(inputs, state, params):
    # 解包参数
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params

    # 从state中获取H，H是上一时刻的隐藏状态
    H, = state

    # 用来存储每个时间步的输出
    outputs = []

    # 遍历输入序列中的每个时间步
    for X in inputs:
        # 更新门的计算
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)  # 更新门 (Z)

        # 重置门的计算
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)  # 重置门 (R)

        # 候选隐状态的计算
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)  # 候选隐状态 (H_tilda)

        # 更新隐状态：根据更新门Z来加权当前隐状态和候选隐状态
        H = Z * H + (1 - Z) * H_tilda  # 新的隐状态

        # 输出层计算
        Y = H @ W_hq + b_q  # 当前时刻的输出

        # 将输出加入到结果中
        outputs.append(Y)

    # 将所有时间步的输出连接起来，返回整个序列的输出
    return torch.cat(outputs, dim=0), (H,)

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)



# 简洁实现
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)