# -*- coding: utf-8 -*-
import os
import math
import torch
import random
import json
from collections import Counter
import jieba

PAD = '<pad>'  # 0
UNK = '<unk>'  # 1
BOS = '<s>'   # 2
EOS = '</s>'  # 3
# 输入： <s> I eat sth .
# 输出： I eat sth  </s>

# encoding=utf-8
# import jieba

# strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
# for str in strs:
#     seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
#     print("Paddle Mode: " + '/'.join(list(seg_list)))

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))


def read_lines(path):
    """
    {"label": "102",
    "label_desc": "news_entertainment",
    "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物",
    "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    """
    with open(path, 'r',encoding="utf-8") as f:
        for line in f:
            yield eval(line)
    f.close()


class Vocab(object):
    def __init__(self, specials=[PAD, UNK, BOS, EOS], config=None,  **kwargs):
        self.specials = specials
        self.counter = Counter()
        self.stoi = {}
        self.itos = {}
        self.weights = None
        self.min_freq = config.min_freq

    def make_vocab(self, dataset):
        for x in dataset:#，dataset为[[词1，词2],[]]则在counter字典中更新
            if x != [""]:#词不为空, 则录入counter
                self.counter.update(x)
        if self.min_freq > 1:#用最小出现次数对counter进行筛选，counter取为  筛选出 频率大于最小出现次数 的词 的频率字典
            self.counter = {w: i for w, i in filter(
                lambda x: x[1] >= self.min_freq, self.counter.items())}
        self.vocab_size = 0#词汇表大小
        for w in self.specials:#特殊词处理
            self.stoi[w] = self.vocab_size #初始化为前几个递增值
            self.vocab_size += 1  

        for w in self.counter.keys():#对读取的词
            self.stoi[w] = self.vocab_size #编码递增序
            self.vocab_size += 1

        self.itos = {i: w for w, i in self.stoi.items()}#反向映射表

    def __len__(self):
        return self.vocab_size


class DataSet(list):
    def __init__(self, *args, config=None, is_train=True, dataset="train"):
        self.config = config
        self.is_train = is_train
        self.dataset = dataset
        self.data_path = os.path.join(self.config.data_path, dataset + ".json")
        super(DataSet, self).__init__(*args)

    def read(self):
        for items in read_lines(self.data_path):
            #sent = tuple(jieba.cut(items["sentence"], cut_all=False))
            sent = tuple(items["sentence"])
            label = items["label_desc"]
            example = [sent, label]
            self.append(example)

    def _numericalize(self, words, stoi):
        return [1 if x not in stoi else stoi[x] for x in words]

    def numericalize(self, w2id, c2id):#w2id 字的id字典  c2id分类的id字典
        for i, example in enumerate(self):#将self打包成 索引，值 的序列 进行迭代
            sent, label = example
            sent = self._numericalize(sent, w2id)#句子字词转换为数值
            label = c2id[label]#标签转换为数值
            self[i] = (sent, label)#更新迭代对象  原 词组，标签  转换为 词组的数值，标签的数值


class DataBatchIterator(object):
    def __init__(self, config, dataset="train",
                 is_train=True,
                 batch_size=32,
                 shuffle=False,
                 batch_first=False,
                 sort_in_batch=True):
        self.config = config
        self.examples = DataSet(
            config=config, is_train=is_train, dataset=dataset)
        self.vocab = Vocab(config=config)
        self.cls_vocab = Vocab(specials=[], config=config)
        self.is_train = is_train
        self.max_seq_len = config.max_seq_len
        self.sort_in_batch = sort_in_batch
        self.is_shuffle = shuffle
        self.batch_first = batch_first  # [batch_size x seq_len x hidden_size]
        self.batch_size = batch_size
        self.num_batches = 0
        self.device = config.device

    def set_vocab(self, vocab):
        self.vocab = vocab

    def load(self, vocab_cache=None):
        self.examples.read()#[[词]，标记]

        if not vocab_cache and self.is_train:#第一次制作词汇表
            # 0: 分过词的句子， 1: 关键词， 2: 标记
            self.vocab.make_vocab([x[0] for x in self.examples])
            self.cls_vocab.make_vocab([[x[1]] for x in self.examples])
            if not os.path.exists(self.config.save_vocab):#首次保存词汇表
                torch.save(self.vocab, self.config.save_vocab + ".txt")
                torch.save(self.cls_vocab, self.config.save_vocab + ".cls.txt")
        else:#已有词汇表， 载入
            self.vocab = torch.load(self.config.save_vocab + ".txt")
            self.cls_vocab = torch.load(self.config.save_vocab + ".cls.txt")
        assert len(self.vocab) > 0#检查词汇表长度大于0
        self.examples.numericalize(
            w2id=self.vocab.stoi, c2id=self.cls_vocab.stoi)#将examples字词内容转换为数值

        self.num_batches = math.ceil(len(self.examples)/self.batch_size)#按batch大小 计算出有多少个batch

    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):
        if add_bos:
            sentence = [w2id[BOS]] + sentence
        if add_eos:
            sentence = sentence + [w2id[EOS]]
        if len(sentence) < max_L:
            sentence = sentence + [w2id[PAD]] * (max_L-len(sentence))
        return [x for x in sentence]

    def pad_seq_pair(self, samples):
        pairs = [pair for pair in samples]

        Ls = [len(pair[0])+2 for pair in pairs]

        max_Ls = max(Ls)
        sent = [self._pad(
            item[0], max_Ls, self.vocab.stoi, add_bos=True, add_eos=True) for item in pairs]
        label = [item[1] for item in pairs]
        batch = Batch()
        batch.sent = torch.LongTensor(sent).to(device=self.device)

        batch.label = torch.LongTensor(label).to(device=self.device)
        if not self.batch_first:
            batch.sent = batch.sent.transpose(1, 0).contiguous()
        batch.mask = batch.sent.data.clone().ne(0).long().to(device=self.device)
        return batch

    def __iter__(self):
        if self.is_shuffle:
            random.shuffle(self.examples)
        total_num = len(self.examples)
        for i in range(self.num_batches):
            samples = self.examples[i * self.batch_size:
                                    min(total_num, self.batch_size*(i+1))]
            # if self.sort_in_batch:
            # samples = sorted(
            #    samples, key=lambda x: len(x[0]), reverse=True)
            yield self.pad_seq_pair(samples)


class Batch(object):
    def __init__(self):
        self.sent = None
        self.label = None
        self.mask = None
