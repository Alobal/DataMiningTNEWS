import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, config):
        super(TextCNN, self).__init__()
        self.kernel_sizes = config.kernel_sizes#kernel大小 2，3，4
        self.hidden_dim = config.embed_dim#嵌入层维度 256
        self.num_channel = config.num_channel#通道数 1
        self.num_class = config.num_class #总分类数 15
        self.word_embedding = nn.Embedding(
            vocab_size, config.embed_dim)  # 词向量，这里直接随机， 有vocab_size个词，每个词是embed_dim维向量

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, config.num_kernel, (kernel, config.embed_dim))
             for kernel in self.kernel_sizes])  # 卷积层

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_kernel*3, self.num_class)  # 全连接层，将 核数量的特征 映射变换到 分类维度

    def forward(self, x):
        x = self.word_embedding(x)#嵌入层处理
        x = x.permute(1, 0, 2).unsqueeze(1)#permute置换顺序 unsqueeze在第一维插入一个维度
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]#卷积层
        x = [torch.max_pool1d(h, h.size(2)).squeeze(2) for h in x]#池化层
        x = torch.cat(x, 1)
        x = self.dropout(x)#防止过拟合
        logits = self.fc(x)#返回logits
        return logits
