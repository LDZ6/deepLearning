import os
import torch
import torchvision
from d2l import torch as d2l
from torch.utils.data import Dataset

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                              '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像及其标注

    参数:
        voc_dir (str): VOC数据集的根目录
        is_train (bool): 是否读取训练集数据，默认为True，读取训练集；False则读取验证集

    返回:
        features (list): 包含图像数据的列表
        labels (list): 包含对应标注的列表
    """

    # 根据是否训练集，选择对应的文件（train.txt或val.txt）
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')

    # 图像读取模式，RGB表示读取彩色图像
    mode = torchvision.io.image.ImageReadMode.RGB

    # 打开文件，获取所有图像文件名
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    # 用于存放图像和标注的列表
    features, labels = [], []

    # 逐个读取图像和对应的标注
    for i, fname in enumerate(images):
        # 读取图像文件（JPEG格式）
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))

        # 读取对应的标注文件（PNG格式），模式为RGB
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))

    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n)

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
[0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
[64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
[64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
[0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
[0, 64, 128]]


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
'diningtable', 'dog', 'horse', 'motorbike', 'person',
'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射

    返回:
        colormap2label (Tensor): RGB值到类别索引的映射张量
    """
    # 构建一个全0张量，大小为256^3 (表示可能的RGB组合数量)，数据类型为long
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)

    # 将VOC的colormap RGB值映射到对应的类别索引
    for i, colormap in enumerate(VOC_COLORMAP):
        # 将RGB值编码为一个唯一索引，映射到类别索引i
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引

    参数:
        colormap (Tensor): VOC标签图像的RGB值
        colormap2label (Tensor): RGB值到类别索引的映射张量

    返回:
        idx (Tensor): 类别索引的张量
    """
    # 将colormap的维度从 (C, H, W) 转为 (H, W, C) 并转换为numpy数组的int32类型
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')

    # 根据RGB值计算唯一索引
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])

    # 将这些索引映射到对应的类别索引
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
print(y[105:115, 130:140], VOC_CLASSES[1])


def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像

    参数:
        feature (Tensor): 输入的特征图像（例如原始图片）
        label (Tensor): 输入的标签图像（例如对应的分割标签）
        height (int): 裁剪后的高度
        width (int): 裁剪后的宽度

    返回:
        feature (Tensor): 裁剪后的特征图像
        label (Tensor): 裁剪后的标签图像
    """

    # 获取随机裁剪参数（起始位置和裁剪区域）
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))

    # 对特征图像进行裁剪
    feature = torchvision.transforms.functional.crop(feature, *rect)

    # 对标签图像进行相同区域的裁剪
    label = torchvision.transforms.functional.crop(label, *rect)

    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)


class VOCSegDataset(Dataset):
    """⾃定义⽤于加载VOC数据集的语义分割数据集

    参数:
        is_train (bool): 指定是否加载训练数据
        crop_size (tuple): 裁剪图像的大小 (height, width)
        voc_dir (str): VOC数据集的根目录
    """

    def __init__(self, is_train, crop_size, voc_dir):
        # 图像标准化转换 (归一化均值和标准差)
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 保存裁剪的尺寸
        self.crop_size = crop_size

        # 读取特征图像和标签
        features, labels = read_voc_images(voc_dir, is_train=is_train)

        # 筛选符合裁剪尺寸要求的图像并进行标准化
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)

        # 生成RGB到类别索引的映射
        self.colormap2label = voc_colormap2label()

        # 打印加载的样本数量
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """将图像归一化"""
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        """筛选出满足裁剪大小的图像

        参数:
            imgs (list): 输入的图像列表

        返回:
            list: 经过筛选的图像列表
        """
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and  # 检查图像高度是否大于裁剪高度
                img.shape[2] >= self.crop_size[1]  # 检查图像宽度是否大于裁剪宽度
        )]

    def __getitem__(self, idx):
        """根据索引返回裁剪后的图像和标签"""
        # 随机裁剪图像和标签
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # 将标签从RGB映射到类别索引
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)



crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)


batch_size = 64

# 创建DataLoader用于批量加载数据
train_iter = torch.utils.data.DataLoader(
    voc_train,            # 训练集
    batch_size=batch_size,  # 每个批次的样本数
    shuffle=True,          # 随机打乱数据
    drop_last=True,        # 如果最后一个批次不足batch_size，则丢弃
    num_workers=d2l.get_dataloader_workers()  # 使用的进程数
)

# 迭代DataLoader，获取批次数据
for X, Y in train_iter:
    print(X.shape)  # 打印输入特征的形状
    print(Y.shape)  # 打印标签的形状
    break  # 打印一次后退出循环


def load_data_voc(batch_size, crop_size):
    """加载VOC2012语义分割数据集

    参数:
        batch_size (int): 每个批次的样本数量
        crop_size (tuple): 裁剪图像的大小 (height, width)

    返回:
        train_iter (DataLoader): 训练数据集的 DataLoader
        test_iter (DataLoader): 测试数据集的 DataLoader
    """

    # 下载并解压缩VOC2012数据集
    voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))

    # 获取用于DataLoader的并行加载数据的进程数
    num_workers = d2l.get_dataloader_workers()

    # 创建训练集的 DataLoader
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(is_train=True, crop_size=crop_size, voc_dir=voc_dir),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    # 创建测试集的 DataLoader
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(is_train=False, crop_size=crop_size, voc_dir=voc_dir),
        batch_size=batch_size, drop_last=True, num_workers=num_workers)

    return train_iter, test_iter