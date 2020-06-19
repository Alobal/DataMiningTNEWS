# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
import logging
import pandas as pd

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
    train_data.load()

    vocab = train_data.vocab#词汇映射表

    # 载入测试集合
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.set_vocab(vocab)
    test_data.load()

    # 测试时载入模型
    model = torch.load(config.save_model+".pt",map_location = config.device)

    print(model)

    test(model,test_data)

    #测试
def test(model,test_data):
    model.eval()
    test_data_iter = iter(test_data)
    pre_labels=[]
    true_labels=[]

    for idx, batch in enumerate(test_data_iter):
        # model.zero_grad()
        outputs = model(batch.sent)
        labels=batch.label

        #转换为list方便后续打分函数
        pre=torch.max(outputs,1)[1].data.squeeze()

        pre_labels+=pre.cpu().numpy().tolist()
        true_labels+=labels.cpu().numpy().tolist()

    #生成报告
    report=classification_report(true_labels,pre_labels)

    print(report)
    
if  __name__ == "__main__":
    main()