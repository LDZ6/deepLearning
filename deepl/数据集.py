import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

# 注册数据集链接
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)

# 读取香蕉检测数据集
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname).set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        # 读取图像
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')
        ))
        # 保存边界框和标签
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

# 自定义 Dataset 类
class BananasDataset(torch.utils.data.Dataset):
    """用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('读取 ' + str(len(self.features)) + (f' 训练样本' if is_train else f' 验证样本'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)

# 加载数据集
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter

# 读取一个小批量并打印图像和标签的形状
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

# 展示带有真实边界框的图像
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

# 显示图像
d2l.plt.show()
