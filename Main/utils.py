# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 11:01
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :
import torch
import json
import os
import shutil


@torch.no_grad()
def get_gaussian_orthogonal_random_matrix(n):
    random_gaussian_matrix = torch.randn((n, n))
    Q, R = torch.linalg.qr(random_gaussian_matrix, mode='reduced')
    return Q


@torch.no_grad()
def get_depth_embedding_matrix(d_model, max_depth=41):
    encoding = torch.zeros(max_depth, d_model)
    pos = torch.arange(0, max_depth)
    pos = pos.float().unsqueeze(dim=1)
    _2i = torch.arange(0, d_model, step=2).float()
    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    return encoding


@torch.no_grad()
def get_type_embedding_matrix(d_model, n_types=3):
    encoding = torch.ones(n_types, d_model)
    for i in range(n_types):
        encoding[i] = i * encoding[i]
    return encoding


def create_log_dict_pretrain(args):
    log_dict = {}

    log_dict['unsup dataset'] = args.unsup_dataset

    log_dict['embedding method'] = args.embedding_method
    log_dict['word embedding size'] = args.word_embedding_size
    log_dict['cn word tokenization'] = args.cn_word_tokenization
    log_dict['unsup train size'] = args.unsup_train_size

    log_dict['use chain identifier'] = args.use_chain_identifier
    log_dict['use depth embedding'] = args.use_depth_embedding
    log_dict['use type embedding'] = args.use_type_embedding
    log_dict['max sequence length'] = args.max_sequence_length

    log_dict['d model'] = args.d_model
    log_dict['num layers'] = args.num_layers
    log_dict['num heads'] = args.num_heads
    log_dict['dim feedforward'] = args.dim_feedforward
    log_dict['measure'] = args.measure
    log_dict['pooling'] = args.pooling

    log_dict['pt batch size'] = args.pt_batch_size
    log_dict['pt acc batch size'] = args.pt_acc_batch_size
    log_dict['pt num epochs'] = args.pt_num_epochs
    log_dict['pt lr'] = args.pt_lr
    log_dict['pt weight decay'] = args.pt_weight_decay
    log_dict['pt warmup ratio'] = args.pt_warmup_ratio

    log_dict['cuda'] = args.cuda
    log_dict['gpu'] = args.gpu
    return log_dict


def create_log_dict_pretrain_parallel(args):
    log_dict = {}

    log_dict['unsup dataset'] = args.unsup_dataset

    log_dict['embedding method'] = args.embedding_method
    log_dict['word embedding size'] = args.word_embedding_size
    log_dict['cn word tokenization'] = args.cn_word_tokenization
    log_dict['unsup train size'] = args.unsup_train_size

    log_dict['use chain identifier'] = args.use_chain_identifier
    log_dict['use depth embedding'] = args.use_depth_embedding
    log_dict['use type embedding'] = args.use_type_embedding
    log_dict['max sequence length'] = args.max_sequence_length

    log_dict['d model'] = args.d_model
    log_dict['num layers'] = args.num_layers
    log_dict['num heads'] = args.num_heads
    log_dict['dim feedforward'] = args.dim_feedforward
    log_dict['measure'] = args.measure
    log_dict['pooling'] = args.pooling

    log_dict['pt batch size'] = args.pt_batch_size
    log_dict['pt acc batch size'] = args.pt_acc_batch_size
    log_dict['pt num epochs'] = args.pt_num_epochs
    log_dict['pt lr'] = args.pt_lr
    log_dict['pt weight decay'] = args.pt_weight_decay
    log_dict['pt warmup ratio'] = args.pt_warmup_ratio

    log_dict['gpus'] = args.gpus
    return log_dict


def create_log_dict_finetuning(args):
    log_dict = {}

    log_dict['record'] = []

    log_dict['dataset'] = args.dataset
    log_dict['unsup dataset'] = args.unsup_dataset

    log_dict['embedding method'] = args.embedding_method
    log_dict['word embedding size'] = args.word_embedding_size
    log_dict['cn word tokenization'] = args.cn_word_tokenization
    log_dict['unsup train size'] = args.unsup_train_size

    log_dict['use chain identifier'] = args.use_chain_identifier
    log_dict['use depth embedding'] = args.use_depth_embedding
    log_dict['use type embedding'] = args.use_type_embedding
    log_dict['max sequence length'] = args.max_sequence_length

    log_dict['d model'] = args.d_model
    log_dict['num layers'] = args.num_layers
    log_dict['num heads'] = args.num_heads
    log_dict['dim feedforward'] = args.dim_feedforward
    log_dict['measure'] = args.measure
    log_dict['pooling'] = args.pooling

    log_dict['pt batch size'] = args.pt_batch_size
    log_dict['pt acc batch size'] = args.pt_acc_batch_size
    log_dict['pt num epochs'] = args.pt_num_epochs
    log_dict['pt lr'] = args.pt_lr
    log_dict['pt weight decay'] = args.pt_weight_decay
    log_dict['pt warmup ratio'] = args.pt_warmup_ratio

    log_dict['ft runs'] = args.ft_runs
    log_dict['ft batch size'] = args.ft_batch_size
    log_dict['ft acc batch size'] = args.ft_acc_batch_size
    log_dict['ft num epochs'] = args.ft_num_epochs
    log_dict['ft lr'] = args.ft_lr
    log_dict['ft weight decay'] = args.ft_weight_decay
    log_dict['split'] = args.split
    log_dict['lamda'] = args.lamda

    log_dict['cuda'] = args.cuda
    log_dict['gpu'] = args.gpu
    return log_dict


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))

    return train_path, val_path, test_path


def get_word2vec_embedding(word2vec, text):
    return word2vec.get_sentence_embedding(text).view(1, -1)
