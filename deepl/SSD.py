import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                    kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

# Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                            kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 20, 20)))
# print(down_sample_blk(3, 10)).shape)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    if i == 0:
        blk = base_net()  # 基础网络，使用前面定义的base_net
    elif i == 1:
        blk = down_sample_blk(64, 128)  # 从64通道到128通道的下采样块
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))  # 自适应池化层，将输出大小调整为1x1
    else:
        blk = down_sample_blk(128, 128)  # 128通道的下采样块，保持输入输出通道相同
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
        [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes, num_anchors=5, sizes=None, ratios=None, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        idx_to_in_channels = [64, 128, 128, 128, 128]

        for i in range(5):
            # 动态赋值 blk, cls_predictor 和 bbox_predictor
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 动态访问 blk, cls 和 bbox
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )

        # 将所有层的结果拼接
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)

        return anchors, cls_preds, bbox_preds

# net = TinySSD(num_classes=1)
# X = torch.zeros((32, 3, 256, 256))
# anchors, cls_preds, bbox_preds = net(X)
# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)


batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # (continues on next page)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                    bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
# 由于类别预测结果放在最后⼀维，argmax需要指定最后⼀维。
    return float((cls_preds.argmax(dim=-1).type(
            cls_labels.dtype) == cls_labels).sum())
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)

for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)  # 用于累计分类准确度和边界框误差
    net.train()

    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)

        # 生成多尺度的锚框，预测类别和边界框偏移量
        anchors, cls_preds, bbox_preds = net(X)

        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)

        # 计算损失
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()

        # 累加分类误差和边界框误差
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())

    # 计算分类误差和边界框MAE
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]

    # 动态显示训练过程
    animator.add(epoch + 1, (cls_err, bbox_mae))

    # 输出每个epoch的训练结果
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')


X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()  # 切换到评估模式，不使用dropout和batch norm
    anchors, cls_preds, bbox_preds = net(X.to(device))  # 模型生成锚框、类别预测和边界框偏移量
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  # 计算类别概率并调整维度
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)  # 多框检测，得到边界框
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]  # 移除置信度低的预测
    return output[0, idx]


def display(img, output, threshold):
    d2l.set_figsize((5, 5))  # 设置图像尺寸
    fig = d2l.plt.imshow(img)  # 显示图像
    for row in output:
        score = float(row[1])  # 提取置信度
        if score < threshold:
            continue  # 如果置信度小于阈值，则跳过
        h, w = img.shape[0:2]  # 获取图像的高和宽
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]  # 计算边界框相对图像的实际位置
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')  # 显示边界框


output = predict(X)  # 运行预测函数，得到输出边界框
display(img, output.cpu(), threshold=0.9)  # 显示预测结果，只保留置信度大于0.9的边界框


