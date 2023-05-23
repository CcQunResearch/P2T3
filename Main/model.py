# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 10:49
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from Main.infomax import local_global_loss


class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class P2T3(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, word_embedding_size, pooling, measure='JSD'):
        super(P2T3, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.word_embedding_size = word_embedding_size
        self.measure = measure
        self.pooling = pooling

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )

        self.local_d = FF(self.d_model, self.d_model)
        self.global_d = FF(self.d_model, self.d_model)
    #
    #     self._init_parameters()
    #
    # def _init_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             # 使用 Xavier 初始化
    #             nn.init.xavier_uniform_(p)
    #             # 或使用 He 初始化
    #             # nn.init.kaiming_uniform_(p, nonlinearity='relu')

    def forward(self, padded_sequences, attention_mask, indexes):
        x = self.transformer_encoder(padded_sequences, src_key_padding_mask=attention_mask)

        global_embeddings = x[:, 0, :]

        # level1embeddings = []
        # for i in range(len(indexes)):
        #     level1embedding = torch.mean(x[i][indexes[i]], dim=0)
        #     level1embeddings.append(level1embedding)
        #
        # global_embeddings = torch.stack(level1embeddings)

        predictions = self.lin_class(global_embeddings)
        return F.log_softmax(predictions, dim=-1)

    def unsup_loss(self, padded_sequences, attention_mask, indexes):
        x = self.transformer_encoder(padded_sequences, src_key_padding_mask=attention_mask)

        global_embeddings = x[:, 0, :].squeeze(1)

        if self.pooling == 'level1':
            level1embeddings = []
            batch = []
            for i in range(len(indexes)):
                batch += [i] * (len(indexes[i]) - 1)
                level1embedding = x[i][indexes[i]][1:]
                level1embeddings.append(level1embedding)
            batch = torch.tensor(batch)
            local_embeddings = torch.cat(level1embeddings, dim=0)

            global_embeddings = self.global_d(global_embeddings)
            local_embeddings = self.local_d(local_embeddings)
            loss = local_global_loss(local_embeddings, global_embeddings, batch, self.measure)
            return loss
        elif self.pooling == 'mean':
            mean_embeddings = []
            batch = []
            for i in range(len(indexes)):
                batch += [i] * max(indexes[i])
                mean_embedding = indexed_mean_pooling(x[i], indexes[i])
                mean_embedding = mean_embedding[1:]
                mean_embeddings.append(mean_embedding)
            batch = torch.tensor(batch)
            local_embeddings = torch.cat(mean_embeddings, dim=0)

            global_embeddings = self.global_d(global_embeddings)
            local_embeddings = self.local_d(local_embeddings)
            loss = local_global_loss(local_embeddings, global_embeddings, batch, self.measure)
            return loss


def indexed_mean_pooling(matrix, index):
    index_counts = torch.bincount(index)
    result = torch.zeros(index_counts.shape[0], matrix.shape[1]).to(matrix.device)

    for i, row in enumerate(matrix[:index.shape[0]]):
        result[index[i]] += row
    result /= index_counts.unsqueeze(1)
    return result
