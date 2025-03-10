# nvidia-smi 显卡信息powershell
import torch
from torch import nn

# 1. 检查可用的 GPU 设备
def try_gpu(i=0):
    """如果存在，则返回 gpu(i)，否则返回 cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的 GPU，如果没有 GPU，则返回 [cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 打印可用的设备
print("设备:", try_gpu(), try_gpu(10), try_all_gpus())

# 2. 张量与 GPU 的操作
# 创建张量并指定设备
X = torch.ones(2, 3, device=try_gpu())
print("X 张量:", X)

# 创建另一个张量并指定不同的 GPU 设备
Y = torch.rand(2, 3, device=try_gpu(1))
print("Y 张量:", Y)

# # 张量间的操作需要在同一设备上
# # 将 X 张量复制到 GPU 1
# Z = X.cuda(1)
# print("X 张量 (GPU 0):", X)
# print("Z 张量 (GPU 1):", Z)
#
# # 在相同的 GPU 上进行张量操作
# result = Y + Z
# print("张量加法结果:", result)
#
# # 确保 Z 张量已经在 GPU 1 上
# print("Z 张量是否仍然在 GPU 1 上:", Z.cuda(1) is Z)

# 3. 神经网络与 GPU
# 定义一个简单的神经网络
net = nn.Sequential(nn.Linear(3, 1))

# 将模型参数放到 GPU 上
net = net.to(device=try_gpu())
print("网络参数设备:", net[0].weight.data.device)

# 使用模型进行预测
X = torch.ones(2, 3, device=try_gpu())
output = net(X)
print("模型输出:", output)
