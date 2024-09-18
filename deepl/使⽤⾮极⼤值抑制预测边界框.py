import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
from d2l.torch import show_bboxes

def box_iou(boxes1, boxes2):
    """
    计算两个锚框或边界框列表中成对的交并比 (IoU)

    参数:
    - boxes1: (Tensor) 边界框列表 1，形状为 (N, 4)
    - boxes2: (Tensor) 边界框列表 2，形状为 (M, 4)

    返回:
    - IoU: (Tensor) 交并比矩阵，形状为 (N, M)
    """

    # 计算边界框的面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    # boxes1, boxes2 的面积
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # 计算交集的左上角和右下角
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算交集区域的宽和高
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    # 交集面积
    inter_areas = inters[:, :, 0] * inters[:, :, 1]

    # 联合面积
    union_areas = areas1[:, None] + areas2 - inter_areas

    # 计算 IoU
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框

    参数:
    - ground_truth: (Tensor) 真实边界框列表，形状为 (M, 4)
    - anchors: (Tensor) 锚框列表，形状为 (N, 4)
    - device: 设备 (例如 'cpu' 或 'cuda')
    - iou_threshold: (float) IOU 阈值，用于确定是否分配真实边界框

    返回:
    - anchors_bbox_map: (Tensor) 每个锚框对应的真实边界框的索引
    """

    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算每个锚框与真实边界框的 IoU
    jaccard = box_iou(anchors, ground_truth)

    # 初始化锚框对应的真实边界框的索引，初始值为 -1 表示未分配
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # 对每个锚框，找到与其 IoU 最大的真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)

    # 根据 IoU 阈值，分配真实边界框
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j

    # 用于处理多对一分配的辅助张量
    col_discard = torch.full((num_anchors,), -1, device=device)
    row_discard = torch.full((num_gt_boxes,), -1, device=device)

    # 为每个真实边界框分配最匹配的锚框
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx // num_gt_boxes).long()

        anchors_bbox_map[anc_idx] = box_idx

        # 使已经分配的锚框和边界框无效
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """
    使用真实边界框标记锚框，并计算锚框的偏移量和类别标签

    参数:
    - anchors: (Tensor) 锚框张量，形状为 (1, num_anchors, 4)
    - labels: (Tensor) 真实边界框和类别标签，形状为 (batch_size, num_objects, 5)，其中每个真实边界框包括 (类别, xmin, ymin, xmax, ymax)

    返回:
    - bbox_offset: (Tensor) 偏移量张量，形状为 (batch_size, num_anchors * 4)
    - bbox_mask: (Tensor) 掩码张量，形状为 (batch_size, num_anchors * 4)
    - class_labels: (Tensor) 类别标签张量，形状为 (batch_size, num_anchors)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]  # 当前 batch 中的标签

        # 分配锚框到真实边界框
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)

        # 构建用于偏移量计算的掩码，形状为 (num_anchors, 4)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        # 初始化类别标签和边界框坐标
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        # 处理分配的真实边界框，并标记锚框的类别
        indices_true = torch.nonzero(anchors_bbox_map >= 0).squeeze(-1)
        bb_idx = anchors_bbox_map[indices_true]

        # 类别标签为真实类别标签加 1，背景标签为 0
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # 计算偏移量并应用掩码
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask

        # 记录偏移量、掩码和类别标签
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    # 将每个 batch 的结果堆叠起来
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    return bbox_offset, bbox_mask, class_labels


def offset_inverse(anchors, offset_preds):
    """
    根据锚框和预测的偏移量计算出预测的边界框。

    参数:
        anchors (Tensor): 锚框的坐标（xmin, ymin, xmax, ymax）。
        offset_preds (Tensor): 对应锚框的预测偏移量（Δx, Δy, Δw, Δh）。

    返回:
        Tensor: 预测的边界框（xmin, ymin, xmax, ymax）。
    """
    # 将锚框从角点坐标转换为中心坐标 (x_center, y_center, width, height)
    anc = d2l.box_corner_to_center(anchors)

    # 根据预测的偏移量计算预测的中心坐标 x 和 y
    # 其中，offset_preds[:, :2] 表示偏移的 Δx 和 Δy
    # anc[:, 2:] 表示锚框的宽和高
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]

    # 根据预测的偏移量计算预测的宽度和高度
    # 其中，offset_preds[:, 2:] 表示宽度和高度的偏移量 Δw 和 Δh
    # 使用 exp 函数来计算相对的变化率
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]

    # 将预测的中心坐标和尺寸组合成中心格式的边界框
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)

    # 将预测的边界框从中心坐标转换为角点坐标
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)

    return predicted_bbox


# @save
def nms(boxes, scores, iou_threshold):
    """
    使用非极大值抑制（NMS）从多个边界框中筛选出最佳边界框。

    参数:
        boxes (Tensor): 形状为 (n, 4) 的张量，表示 n 个边界框，每个边界框由 (xmin, ymin, xmax, ymax) 表示。
        scores (Tensor): 形状为 (n,) 的张量，表示每个边界框的置信度分数。
        iou_threshold (float): 用于决定两个边界框是否过于重叠的 IOU 阈值。

    返回:
        Tensor: 保留下来的边界框的索引。
    """
    # 根据置信度分数进行排序，得到降序排列的索引
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 用于存储保留下来的边界框的索引

    # 循环直到处理完所有边界框
    while B.numel() > 0:
        i = B[0]  # 当前置信度最高的边界框索引
        keep.append(i)  # 保留该边界框
        if B.numel() == 1:
            break  # 如果只剩下一个边界框，结束循环

        # 计算当前选中的框与剩余框的 IOU
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)

        # 找出与当前框的 IOU 小于阈值的边界框索引
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)

        # 更新 B，只保留那些 IOU 小于阈值的框
        B = B[inds + 1]

    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使⽤⾮极⼤值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)  # 压缩维度
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []

    # 遍历每个样本
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)  # 从第1类开始找最大概率（忽略背景类）
        # 将偏移量还原为预测的边界框
        predicted_bb = offset_inverse(anchors, offset_pred)

        # 执行非极大值抑制，保留置信度较高的框
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有未保留的锚框
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))  # 将保留的和所有的锚框索引拼接
        uniques, counts = combined.unique(return_counts=True)  # 找出唯一的索引及其出现次数
        non_keep = uniques[counts == 1]  # 找出未保留的索引

        # 排序所有的索引
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1  # 对非保留框设为背景类

        # 对置信度和预测边界框进行重新排序
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]

        # 过滤低于阈值的置信度框
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1  # 设置低于阈值的框为背景类
        conf[below_min_idx] = 1 - conf[below_min_idx]  # 调整置信度

        # 将预测信息拼接为 (class_id, conf, predicted_bb)
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)

        out.append(pred_info)  # 将结果添加到输出列表中

    return torch.stack(out)  # 将列表转换为张量

bbox_scale = torch.tensor(( 728, 561, 728, 561))  # 边界框比例
img = plt.imread('../img/catdog.jpg')
print(img.shape[0:2])# 替换为你使用的图片路径

# 测试multibox_target
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],  # 狗
                             [1, 0.55, 0.2, 0.9, 0.88]])  # 猫

anchors = torch.tensor([[0, 0.1, 0.2, 0.3],   # A0
                        [0.15, 0.2, 0.4, 0.4],  # A1
                        [0.63, 0.05, 0.88, 0.98],  # A2
                        [0.66, 0.45, 0.8, 0.8],  # A3
                        [0.57, 0.3, 0.92, 0.9]])  # A4

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

print(labels[0])
print(labels[1])
print(labels[2])

# 给定的锚框，偏移预测和类别概率
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.zeros(anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫预测概率

# 显示图像和预测结果

fig, ax = plt.subplots()
ax.imshow(img)

# 显示预测边界框和置信度
show_bboxes(ax, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

# 执行非极大值抑制
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)

# 过滤背景类边界框并显示结果
output = output[0].detach().numpy()
fig, ax = plt.subplots()
ax.imshow(img)

for i in output:
    if i[0] == -1:
        continue  # 跳过背景类
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(ax, [torch.tensor(i[2:]) * bbox_scale], label)
plt.show()