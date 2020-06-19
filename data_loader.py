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
        for x in dataset:
            if x != [""]:
                self.counter.update(x)
        if self.min_freq > 1:
            self.counter = {w: i for w, i in filter(
                lambda x: x[1] >= self.min_freq, self.counter.items())}
        self.vocab_size = 0
        for w in self.specials:
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        for w in self.counter.keys():
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        self.itos = {i: w for w, i in self.stoi.items()}

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
        for i, example in enumerate(self):
            sent, label = example
            sent = self._numericalize(sent, w2id)#字词转换为数值
            label = c2id[label]
            self[i] = (sent, label)


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

        if not vocab_cache and self.is_train:
            # 0: 分过词的句子， 1: 关键词， 2: 标记
            self.vocab.make_vocab([x[0] for x in self.examples])
            self.cls_vocab.make_vocab([[x[1]] for x in self.examples])
            if not os.path.exists(self.config.save_vocab):
                torch.save(self.vocab, self.config.save_vocab + ".txt")
                torch.save(self.cls_vocab, self.config.save_vocab + ".cls.txt")
        else:#测试数据直接载入做好的词汇表
            self.vocab = torch.load(self.config.save_vocab + ".txt")
            self.cls_vocab = torch.load(self.config.save_vocab + ".cls.txt")
        assert len(self.vocab) > 0
        self.examples.numericalize(
            w2id=self.vocab.stoi, c2id=self.cls_vocab.stoi)#将examples字词内容转换为数值

        self.num_batches = math.ceil(len(self.examples)/self.batch_size)

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
