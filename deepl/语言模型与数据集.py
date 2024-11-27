import random
import torch
from d2l import torch as d2l
import collections
import re
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 定义数据集 URL 和哈希
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)


# 读取《时间机器》文本并清理数据
def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 清理数据：去除非字母字符，转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 词元化函数：支持按单词或字符进行分词
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# 统计词频
def count_corpus(tokens):
    """统计词元的频率"""
    # 处理二维列表或一维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 定义词表类
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频并排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元索引为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 添加满足最小词频要求的词元
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """根据词元返回其索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """根据索引返回词元"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引为 0"""
        return 0

    @property
    def token_freqs(self):
        """返回词元频率"""
        return self._token_freqs


# 加载和处理《时间机器》数据集
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 使用字符进行词元化
    vocab = Vocab(tokens)  # 构建词表
    # 将所有文本行展平为一个词元列表
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# 读取数据并进行词元化
tokens = d2l.tokenize(read_time_machine())
corpus = [token for line in tokens for token in line]

# 构建词表
vocab = Vocab(corpus)
# 查看词频前10个单词
print(vocab.token_freqs[:10])

# 词频列表
freqs = [freq for token, freq in vocab.token_freqs]
# 绘制词频图
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
plt.show()

# 构建二元语法和三元语法的词元表
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]

bigram_vocab = Vocab(bigram_tokens)
trigram_vocab = Vocab(trigram_tokens)

# 查看二元语法和三元语法词频前10项
print(bigram_vocab.token_freqs[:10])
print(trigram_vocab.token_freqs[:10])

# 获取二元、三元词频
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

# 绘制一元、二元和三元语法的词频图
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
plt.show()

# 随机采样生成小批量子序列
def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

import random
import torch

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。

    Args:
        corpus (list or array): 输入的文本序列（通常是整数索引形式）。
        batch_size (int): 每个小批量的样本数量。
        num_steps (int): 每个序列的时间步长。

    Yields:
        tuple: 包含输入序列 (X) 和目标序列 (Y) 的小批量。
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps - 1)  # 确保偏移量小于 num_steps
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    # 切分 corpus 为 Xs 和 Ys
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # 调整为 (batch_size, -1) 的形状
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)

    # 计算每个批量中包含的时间步长
    num_batches = Xs.shape[1] // num_steps

    # 生成小批量
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# 示例序列
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """
        初始化数据加载器
        参数：
        batch_size: 每个小批量的大小
        num_steps: 每次采样序列的长度
        use_random_iter: 是否使用随机采样
        max_tokens: 词汇表的最大长度
        """
        # 根据是否随机迭代，选择对应的数据采样函数
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential

        # 加载语料库和词汇表
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """返回数据迭代器"""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):  # @save
    """
    返回时光机器数据集的迭代器和词表
    参数：
    batch_size: 每个小批量的大小
    num_steps: 每次采样序列的长度
    use_random_iter: 是否使用随机采样
    max_tokens: 词汇表的最大长度
    返回：
    data_iter: 数据迭代器
    vocab: 词表
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
