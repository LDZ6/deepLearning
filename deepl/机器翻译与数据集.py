import os
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 保存数据集链接和校验和
d2l.DATA_HUB['fra-eng'] = (
    d2l.DATA_URL + 'fra-eng.zip',
    '94646ad1522d915e7b0f9296181140edcf86a4f5'
)
# 载入“英语-法语”数据集的函数
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')  # 下载并解压数据集
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

# 读取并展示数据集的前75个字符
raw_text = read_data_nmt()
print(raw_text[:75])


# 预处理“英语－法语”数据集
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    # 判断是否是标点符号且前一个字符不是空格
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格和其他特殊空格字符，转换为小写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]

    return ''.join(out)

# 预处理原始数据并打印处理后的前80个字符
text = preprocess_nmt(raw_text)
print(text[:80])


# 词元化“英语－法语”数据集
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据集"""

    source, target = [], []

    # 按行处理文本，每行代表一个英语-法语的句对
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break  # 如果指定了 num_examples，处理到指定数量后停止
        parts = line.split('\t')  # 使用制表符分割句子
        if len(parts) == 2:
            source.append(parts[0].split(' '))  # 英语句子分词
            target.append(parts[1].split(' '))  # 法语句子分词

    return source, target

# 词元化数据并查看前6个句对
source, target = tokenize_nmt(text)
print(source[:6], target[:6])


# 绘制列表长度对的直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()  # 设置图像大小
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]]  # 计算每个列表的长度
    )
    d2l.plt.xlabel(xlabel)  # 设置x轴标签
    d2l.plt.ylabel(ylabel)  # 设置y轴标签
    for patch in patches[1].patches:
        patch.set_hatch('/')  # 为目标列表的条形图设置不同的填充样式
    d2l.plt.legend(legend)  # 添加图例
    plt.show()


# 示例数据：源语言和目标语言的句子列表
source = [['hello', 'world'], ['how', 'are', 'you'], ['this', 'is', 'a', 'test']]
target = [['bonjour', 'monde'], ['comment', 'ça', 'va'], ['ceci', 'est', 'un', 'test']]

# 使用示例：绘制源语言和目标语言的长度分布直方图
show_list_len_pair_hist(
    ['source', 'target'],
    '# tokens per sequence',
    'count',
    source,
    target
)

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])


# 截断或填充文本序列
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 如果文本超过指定长度，进行截断
    return line + [padding_token] * (num_steps - len(line))  # 如果文本不足，进行填充

# 示例使用：对源语言的第一个句子进行截断或填充
truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])


# 将机器翻译的文本序列转换成小批量
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # 将文本序列转换为词索引列表
    lines = [vocab[l] for l in lines]
    # 在每个句子的末尾添加 <eos>（结束符）
    lines = [l + [vocab['<eos>']] for l in lines]
    # 截断或填充每个句子，使其长度为 num_steps
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # 计算每个句子的有效长度（不包括 <pad> 的部分）
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # 预处理数据，读取并清洗文本
    text = preprocess_nmt(read_data_nmt())

    # 对文本进行分词
    source, target = tokenize_nmt(text, num_examples)

    # 创建源语言和目标语言的词汇表，min_freq=2表示至少出现2次的词才被保留
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 将文本转换为索引数组，并截断或填充到指定的长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # 将源语言和目标语言的数据存储到元组中
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    # 创建数据迭代器，加载数据并按批次分割
    data_iter = d2l.load_array(data_arrays, batch_size)

    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效⻓度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效⻓度:', Y_valid_len)
    break


