# 导入必要的库
import collections
import re
from d2l import torch as d2l

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


# 运行文本处理的示例
if __name__ == "__main__":
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])  # 打印第一行文本
    print(lines[10])  # 打印第十行文本

    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    corpus, vocab = load_corpus_time_machine()
    print(f'词元数量: {len(corpus)}, 词表大小: {len(vocab)}')
    print(f'前 10 个高频词元及其索引: {list(vocab.token_to_idx.items())[:10]}')
