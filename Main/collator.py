# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 16:40
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : collator.py
# @Software: PyCharm
# @Note    :
import torch
from torch.nn.utils.rnn import pad_sequence


class Collator:
    def __init__(self, pooling, device, output_label=False, padding_value=0):
        self.pooling = pooling
        self.device = device
        self.output_label = output_label
        self.padding_value = padding_value

    def __call__(self, batch):
        sequences = [example["x"] for example in batch]
        if self.pooling == "level1":
            pooling_key = "level1_index"
        elif self.pooling == "mean":
            pooling_key = "pooling_index"
        indexes = [example[pooling_key].to(self.device) for example in batch]

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.padding_value).to(self.device)
        attention_mask = (padded_sequences == 0).all(dim=-1).to(self.device)

        if self.output_label:
            labels = [example["label"] for example in batch]
            labels = torch.tensor(labels).to(self.device)
            return padded_sequences, attention_mask, indexes, labels
        else:
            return padded_sequences, attention_mask, indexes
