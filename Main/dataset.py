# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 19:17
# @Author  :
# @Email   :
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import os.path as osp
import json
import torch
import random
from torch.utils.data import Dataset, IterableDataset
from Main.utils import get_word2vec_embedding, get_gaussian_orthogonal_random_matrix, get_depth_embedding_matrix, \
    get_type_embedding_matrix


class SequenceTreeDataset(Dataset):
    def __init__(self, data_path, d_model, embedding_method, word2vec, bert_model, content_key, use_chain_identifier,
                 use_depth_embedding, use_type_embedding, max_sequence_length, chunk_size=2048, preload_num=128):
        # word embedding
        self.embedding_method = embedding_method
        self.word2vec = word2vec  # word2vec or plm
        self.bert_model = bert_model
        self.content_key = content_key

        # hyperparameters
        self.d_model = d_model
        self.use_chain_identifier = use_chain_identifier
        self.use_depth_embedding = use_depth_embedding
        self.use_type_embedding = use_type_embedding
        self.max_chain_identifier = self.d_model
        self.max_sequence_length = max_sequence_length
        self.chain_identifier_matrix = get_gaussian_orthogonal_random_matrix(d_model)
        self.depth_embedding_matrix = get_depth_embedding_matrix(d_model)
        self.type_embedding_matrix = get_type_embedding_matrix(d_model)

        # path
        self.data_path = data_path
        self.raw_path = osp.join(data_path, 'raw')
        tokenization_setting = "wc" if content_key == "word content" else "ct"
        prior_setting = "".join(["C" if use_chain_identifier else "", "D" if use_depth_embedding else "",
                                 "T" if use_type_embedding else ""])
        self.processed_data_path = osp.join(data_path, 'processed',
                                            f'{d_model}_{embedding_method}_{tokenization_setting}_{prior_setting}')

        self.chunk_size = chunk_size
        self.preload_num = preload_num
        if not osp.exists(self.processed_data_path):
            self.load_data()

        self.features_files = sorted(os.listdir(self.processed_data_path))
        self.lengths = []
        self.preloaded_features = []

        for file in self.features_files[:preload_num]:
            features = torch.load(os.path.join(self.processed_data_path, file))
            self.lengths.append(len(features))
            self.preloaded_features.append(features)

        for file in self.features_files[preload_num:]:
            features = torch.load(os.path.join(self.processed_data_path, file))
            self.lengths.append(len(features))

    def load_data(self):
        print('loading data...', flush=True)
        os.makedirs(self.processed_data_path)

        self.data = []
        self.chunk_num = 0
        for i, file_name in enumerate(os.listdir(self.raw_path)):
            file_path = osp.join(self.raw_path, file_name)
            post = json.load(open(file_path, 'r', encoding='utf-8'))
            one_data = {'label': post['source']['label']} if 'label' in post['source'].keys() else {}
            chain_identifier_list = []
            depth_list = []
            type_list = []
            level1_index = []

            # process source
            x = get_word2vec_embedding(self.word2vec, post['source'][self.content_key])
            chain_identifier_list.append(post['source']['chain identifier'])
            depth_list.append(post['source']['depth'])
            type_list.append(post['source']['type'])
            level1_index.append(0)

            # process deep conversation
            deep_conversations = post['comment']['deep conversation']
            for chain in deep_conversations:
                if chain['chain identifier'] > self.max_chain_identifier - 1 or len(
                        chain_identifier_list) > self.max_sequence_length:
                    break
                level1_index.append(len(chain_identifier_list))
                for comment in chain['comments'][:40]:
                    x = torch.cat((x, get_word2vec_embedding(self.word2vec, comment[self.content_key])), dim=0)
                    chain_identifier_list.append(chain['chain identifier'])
                    depth_list.append(comment['depth'])
                    type_list.append(chain['type'])

            # process shallow conversation
            shallow_conversations = post['comment']['shallow conversation']
            for comment in shallow_conversations:
                if comment['chain identifier'] > self.max_chain_identifier - 1 or len(
                        chain_identifier_list) > self.max_sequence_length:
                    break
                level1_index.append(len(chain_identifier_list))
                x = torch.cat((x, get_word2vec_embedding(self.word2vec, comment[self.content_key])), dim=0)
                chain_identifier_list.append(comment['chain identifier'])
                depth_list.append(comment['depth'])
                type_list.append(comment['type'])

            if self.use_chain_identifier:
                x = x + self.chain_identifier_matrix[torch.tensor(chain_identifier_list)]
            if self.use_depth_embedding:
                x = x + self.depth_embedding_matrix[torch.tensor(depth_list)]
            if self.use_type_embedding:
                x = x + self.type_embedding_matrix[torch.tensor(type_list)]
            one_data['x'] = x
            one_data['level1_index'] = torch.tensor(level1_index)
            one_data['pooling_index'] = torch.tensor(chain_identifier_list)

            self.data.append(one_data)

            if len(self.data) == self.chunk_size:
                torch.save(self.data, osp.join(self.processed_data_path, f'{self.chunk_num}.pt'))
                self.data = []
                self.chunk_num += 1

        if len(self.data) > 0:
            torch.save(self.data, osp.join(self.processed_data_path, f'{self.chunk_num}.pt'))
            self.chunk_num += 1

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if index < length:
                if i < self.preload_num:
                    features = self.preloaded_features[i]
                else:
                    features_file = os.path.join(self.processed_data_path, self.features_files[i])
                    features = torch.load(features_file)
                return features[index]
            index -= length


class IterableSequenceTreeDataset(IterableDataset):
    def __init__(self, data_path, d_model, embedding_method, word2vec, bert_model, content_key, use_chain_identifier,
                 use_depth_embedding, use_type_embedding, max_sequence_length):
        # word embedding
        self.embedding_method = embedding_method
        self.word2vec = word2vec  # word2vec or plm
        self.bert_model = bert_model
        self.content_key = content_key

        # hyperparameters
        self.d_model = d_model
        self.use_chain_identifier = use_chain_identifier
        self.use_depth_embedding = use_depth_embedding
        self.use_type_embedding = use_type_embedding
        self.max_chain_identifier = self.d_model
        self.max_sequence_length = max_sequence_length
        self.chain_identifier_matrix = get_gaussian_orthogonal_random_matrix(d_model)
        self.depth_embedding_matrix = get_depth_embedding_matrix(d_model)
        self.type_embedding_matrix = get_type_embedding_matrix(d_model)

        # path
        self.data_path = data_path
        self.raw_path = osp.join(data_path, 'raw')

    def __len__(self):
        return len(os.listdir(self.raw_path))

    def __iter__(self):
        file_list = os.listdir(self.raw_path)
        random.shuffle(file_list)
        for i, file_name in enumerate(file_list):
            file_path = osp.join(self.raw_path, file_name)
            post = json.load(open(file_path, 'r', encoding='utf-8'))
            one_data = {'label': post['source']['label']} if 'label' in post['source'].keys() else {}
            chain_identifier_list = []
            depth_list = []
            type_list = []
            level1_index = []

            # process source
            x = get_word2vec_embedding(self.word2vec, post['source'][self.content_key])
            chain_identifier_list.append(post['source']['chain identifier'])
            depth_list.append(post['source']['depth'])
            type_list.append(post['source']['type'])
            level1_index.append(0)

            # process deep conversation
            deep_conversations = post['comment']['deep conversation']
            for chain in deep_conversations:
                if chain['chain identifier'] > self.max_chain_identifier - 1 or len(
                        chain_identifier_list) > self.max_sequence_length:
                    break
                level1_index.append(len(chain_identifier_list))
                for comment in chain['comments'][:40]:
                    x = torch.cat((x, get_word2vec_embedding(self.word2vec, comment[self.content_key])), dim=0)
                    chain_identifier_list.append(chain['chain identifier'])
                    depth_list.append(comment['depth'])
                    type_list.append(chain['type'])

            # process shallow conversation
            shallow_conversations = post['comment']['shallow conversation']
            for comment in shallow_conversations:
                if comment['chain identifier'] > self.max_chain_identifier - 1 or len(
                        chain_identifier_list) > self.max_sequence_length:
                    break
                level1_index.append(len(chain_identifier_list))
                x = torch.cat((x, get_word2vec_embedding(self.word2vec, comment[self.content_key])), dim=0)
                chain_identifier_list.append(comment['chain identifier'])
                depth_list.append(comment['depth'])
                type_list.append(comment['type'])

            if self.use_chain_identifier:
                x = x + self.chain_identifier_matrix[torch.tensor(chain_identifier_list)]
            if self.use_depth_embedding:
                x = x + self.depth_embedding_matrix[torch.tensor(depth_list)]
            if self.use_type_embedding:
                x = x + self.type_embedding_matrix[torch.tensor(type_list)]
            one_data['x'] = x
            one_data['level1_index'] = torch.tensor(level1_index)
            one_data['pooling_index'] = torch.tensor(chain_identifier_list)
            yield one_data


def normalize(vector):
    mean = torch.mean(vector)
    std = torch.std(vector)
    normalized_vector = (vector - mean) / std
    return normalized_vector
