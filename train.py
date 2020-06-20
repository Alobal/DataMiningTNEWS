# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging


def build_textcnn_model(vocab, config, train=True):
    model = TextCNN(vocab.vocab_size, config)
    if train:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def train_textcnn_model(model, train_data, valid_data, padding_idx, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    params = [p for k, p in model.named_parameters() if p.requires_grad]
    optimizer = Adam(params, lr=config.lr)#优化器
    criterion = CrossEntropyLoss(reduction="sum")#评价函数
    model.train()

    for epoch in range(1, config.epochs + 1):#epoch代表一次完整训练
        train_data_iter = iter(train_data)
        for idx, batch in enumerate(train_data_iter):
            model.zero_grad()#刷新为0梯度
            ground_truth = batch.label
            # batch_first = False
            outputs = model(batch.sent)#输出tensor值
            loss = criterion(outputs, ground_truth)
            loss.backward()#反向传播计算
            optimizer.step()#更新参数

            if idx % 20 == 0:#每20个batch输出一次 loss
                valid_loss = valid_textcnn_model(
                    model, valid_data, criterion, config)
                # 处理
                print("epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, valid_loss))
                model.train()


def valid_textcnn_model(model,  valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()
    total_loss = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, batch.label)
        # loss 打印
        # 处理
        total_loss += loss
    return total_loss


def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()#获得 数值形式的 词组和标签， 制造好了 字词：id 的映射表

    vocab = train_data.vocab#词汇映射表

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="dev",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    # 构建textcnn模型
    model = build_textcnn_model(
        vocab, config, train=True)

    print(model)

    # Do training.
    padding_idx = vocab.stoi[PAD]
    train_textcnn_model(model, train_data,
                        valid_data, padding_idx, config)
    torch.save(model, '%s.pt' % (config.save_model))

    # 测试时
    # checkpoint = torch.load(config.save_model+".pt",
    #                      map_location = config.device)
    # checkpoint
    # model = build_textcnn_model(
    #     vocab, config, train=True)
    # .....
if __name__ == "__main__":
    main()
