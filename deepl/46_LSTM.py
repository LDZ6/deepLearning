import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import load_data_time_machine, RNNModelScratch, train_ch8


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # 初始化权重
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 初始化LSTM门的权重
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 输入门参数
    W_xi, W_hi, b_i = three()
    # 遗忘门参数
    W_xf, W_hf, b_f = three()
    # 输出门参数
    W_xo, W_ho, b_o = three()
    # 候选记忆元参数
    W_xc, W_hc, b_c = three()

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 将所有参数放入列表
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    # 设置梯度计算
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    # 解包参数
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params

    # 初始状态
    (H, C) = state

    # 存储输出
    outputs = []

    # 遍历输入序列
    for X in inputs:
        # 输入门、遗忘门、输出门、候选记忆元
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)

        # 更新记忆单元状态 C 和隐藏状态 H
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)

        # 输出层
        Y = (H @ W_hq) + b_q
        outputs.append(Y)

    # 拼接输出
    return torch.cat(outputs, dim=0), (H, C)

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)