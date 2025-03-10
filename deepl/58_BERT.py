import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import EncoderBlock


# 输入表示处理函数
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# BERT编码器实现
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X += self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# 掩码语言模型任务
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

# 下一句预测任务
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

# 完整BERT模型整合
class BERTModel(nn.Module):
    """BERT完整模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(
            vocab_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout,
            max_len=max_len, key_size=key_size,
            query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        mlm_Y_hat = self.mlm(encoded_X, pred_positions) if pred_positions else None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# 示例用法
if __name__ == "__main__":
    # 初始化参数
    vocab_size, num_hiddens = 10000, 768
    ffn_num_hiddens, num_heads, num_layers, dropout = 1024, 4, 2, 0.2
    norm_shape, ffn_num_input = [768], 768

    # 实例化编码器
    encoder = BERTEncoder(
        vocab_size, num_hiddens, norm_shape, ffn_num_input,
        ffn_num_hiddens, num_heads, num_layers, dropout)

    # 测试输入
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0,0,0,0,1,1,1,1], [0,0,0,1,1,1,1,1]])
    encoded_X = encoder(tokens, segments, None)
    print("Encoded shape:", encoded_X.shape)  # 应为 torch.Size([2, 8, 768])

    # 测试掩码语言模型
    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1,5,2], [6,1,5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print("MLM prediction shape:", mlm_Y_hat.shape)  # 应为 torch.Size([2, 3, 10000])

    # 测试下一句预测
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X.mean(dim=1))
    print("NSP prediction shape:", nsp_Y_hat.shape)  # 应为 torch.Size([2, 2])