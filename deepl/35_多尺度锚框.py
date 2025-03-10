import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 加载图片
img = plt.imread('../img/catdog.jpg')

# 获取图片的高度和宽度
h, w = img.shape[:2]


# 定义显示锚框的函数
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()

    # 创建一个特征图，前两个维度不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))

    # 生成锚框
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])

    # 缩放锚框到图片大小
    bbox_scale = torch.tensor((w, h, w, h))

    # 显示锚框
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)


# 显示不同特征图尺度的锚框
# 小目标：特征图大小为4x4，锚框大小为0.15
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
plt.show()
# 中目标：特征图大小为2x2，锚框大小为0.4
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
plt.show()
# 大目标：特征图大小为1x1，锚框大小为0.8
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()
