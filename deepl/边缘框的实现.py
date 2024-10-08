import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 设置图像显示大小
d2l.set_figsize()

# 加载示例图像
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img)

# 边界框转换函数
# 从（左上，右下）转换到（中间，宽度，高度）
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

# 从（中间，宽度，高度）转换到（左上，右下）
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 定义狗和猫的边界框（左上x, 左上y, 右下x, 右下y）
dog_bbox = [60.0, 45.0, 378.0, 516.0]
cat_bbox = [400.0, 112.0, 655.0, 493.0]

# 将边界框转换为张量
boxes = torch.tensor([dog_bbox, cat_bbox])

# 验证边界框转换函数的正确性
assert torch.all(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 边界框格式转换为 matplotlib 格式的辅助函数
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
                             fill=False, edgecolor=color, linewidth=2)

# 在图像上画出边界框
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))  # 绘制狗的边界框
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))   # 绘制猫的边界框
plt.show()
