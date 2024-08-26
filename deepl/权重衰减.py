# import torch
# from matplotlib import pyplot as plt
# from torch import nn
# from d2l import torch as d2l
#
# from deepl.utils import Animator
#
#
# # 生成数据
# def synthetic_data(w, b, num_examples):
#     """生成 y = Xw + b + 噪声"""
#     X = torch.normal(0, 1, (num_examples, w.shape[0]))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     return X, y.reshape((-1, 1))
#
# # 设置超参数和生成数据
# n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# train_data = synthetic_data(true_w, true_b, n_train)
# train_iter = d2l.load_array(train_data, batch_size)
# test_data = synthetic_data(true_w, true_b, n_test)
# test_iter = d2l.load_array(test_data, batch_size, is_train=False)
#
# # 初始化模型参数
# def init_params():
#     w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
#     b = torch.zeros(1, requires_grad=True)
#     return [w, b]
#
# # L2 范数惩罚（正则化项）
# def l2_penalty(w):
#     return torch.sum(w.pow(2)) / 2
#
# # 训练函数
# def train(lambd):
#     w, b = init_params()
#     net = lambda X: d2l.linreg(X, w, b)
#     loss = d2l.squared_loss
#     num_epochs, lr = 100, 0.003
#     animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                             xlim=[5, num_epochs], legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             # 增加 L2 范数惩罚项，广播机制使 l2_penalty(w) 成为一个长度为 batch_size 的向量
#             l = loss(net(X), y) + lambd * l2_penalty(w)
#             l.sum().backward()
#             d2l.sgd([w, b], lr, batch_size)
#         if (epoch + 1) % 5 == 0:
#             animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
#                                      d2l.evaluate_loss(net, test_iter, loss)))
#     print('w 的 L2 范数是：', torch.norm(w).item())
#     plt.show()
#
# # 忽略正则化直接训练
# train(lambd=0)
# # 使用权重衰减
# train(lambd=3)


import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

from deepl.utils import Animator

# 超参数设置
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 简洁实现权重衰减
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}  # bias 不进行衰减
    ], lr=lr)

    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
    plt.show()

# 训练和测试
train_concise(0)  # 不使用权重衰减
train_concise(3)  # 使用权重衰减
