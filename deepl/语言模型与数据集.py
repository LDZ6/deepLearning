import random
import torch
from d2l import torch as d2l

# 读取数据并进行词元化
tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]

# 构建词表
vocab = d2l.Vocab(corpus)

# 查看词频前10个单词
print(vocab.token_freqs[:10])

# 词频列表
freqs = [freq for token, freq in vocab.token_freqs]

# 绘制词频图
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')

# 构建二元语法和三元语法的词元表
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]

bigram_vocab = d2l.Vocab(bigram_tokens)
trigram_vocab = d2l.Vocab(trigram_tokens)

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

# 示例序列
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
